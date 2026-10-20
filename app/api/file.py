'''文件上传接口'''
from pathlib import Path

from anyio import to_thread
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from loguru import logger
from starlette.responses import JSONResponse

from app.core.auth import Principal, Role, require_role
from app.services.document_converter import SUPPORTED_EXTENSIONS
from app.services.vector_index_service import vector_index_service

router = APIRouter()

# 文件上传后存储的路径
UPLOAD_DIR = Path("./uploads")
# 支持的文件类型（扩展名列表来自 document_converter，唯一真相源）
ALLOWED_EXTENSIONS = sorted(SUPPORTED_EXTENSIONS)
# 单个文件支持最大大小
MAX_FILE_SIZE = 10*1024*1024


@router.post('/upload')
async def upload_file(file:UploadFile = File(...), _user: Principal = Depends(require_role(Role.OPERATOR, Role.ADMIN))):
    '''
    上传文件并自动创建向量索引

    :param file:
        上传的文件

    :return:
        JSONResponse：上传结果
    '''
    try:
        # 1.验证文件
        if not file.filename:
            raise HTTPException(status_code=400,detail = "文件名不能为空")

        # 2.规范化文件名（取出空格，处理Windows上传的文件）
        safe_filename = _sanitize_filename(file.filename)

        # 3.验证文件拓展名
        file_extension = _get_file_extension(safe_filename)
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code = 400,
                detail = f"不支持的文件格式，仅支持：{', '.join(ALLOWED_EXTENSIONS)}",
            )

        # 4.创建上传目录
        UPLOAD_DIR.mkdir(parents = True,exist_ok = True)

        # 5.保存文件（防目录穿越：解析后必须仍在 UPLOAD_DIR 内）
        file_path = UPLOAD_DIR / safe_filename
        try:
            file_path = file_path.resolve()
            file_path.relative_to(UPLOAD_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="非法文件路径")

        # 如果文件已存在，先删除旧文件（实现覆盖更新）
        if file_path.exists():
            logger.info(f"文件已存在，将覆盖：{file_path}")
            file_path.unlink()

        # 读取并保存文件内容
        content = await file.read()

        # 验证文件大小
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400,detail = f"文件大小超过限制(最大{MAX_FILE_SIZE}字节")

        file_path.write_bytes(content)

        logger.info(f"文件上传成功：{file_path}")

        # 6.自动创建向量索引（同步阻塞逻辑放到线程池，避免阻塞事件循环）
        try:
            logger.info(f"开始为上传文件创建向量索引:{file_path}")
            await to_thread.run_sync(vector_index_service.index_single_file, str(file_path))
            logger.info(f"向量索引创建成功：{file_path}")
        except Exception as e:
            logger.error(f"向量索引创建失败：{file_path}，错误:{e}")
            # 注意，即使索引失败，文件上传仍然成功，只是记录错误日志

        # 7.返回响应
        return JSONResponse(
            status_code = 200,
            content = {
                'code':200,
                'message':'success',
                'data':{
                    'file_name':safe_filename,
                    'file_path':str(file_path),
                    'size':len(content),
                },
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败：{e}")
        raise HTTPException(status_code=500,detail=f"文件上传失败：{e}")


@router.post('/index_directory')
async def index_directory(directory_path:str = None, _user: Principal = Depends(require_role(Role.OPERATOR, Role.ADMIN))):
    """
    索引指定目录下的所有文件

    Args:
        directory_path: 目录路径（可选，默认使用 uploads 目录）

    Returns:
        JSONResponse: 索引结果
    """
    try:
        logger.info(f"开始索引目录:{directory_path or 'uploads'}")

        # 执行索引（同步阻塞逻辑放到线程池，避免阻塞事件循环）
        result = await to_thread.run_sync(vector_index_service.index_directory, directory_path)

        return JSONResponse(
            status_code = 200,
            content = {
                'code':200,
                'message':'success' if result.success else 'partial_success',
                'data':result.to_dict(),
            },
        )

    except Exception as e:
        logger.error(f"索引目录失败:{e}")
        raise HTTPException(status_code=500,detail=f'索引目录失败:{e}')


def _get_file_extension(filename):
    """
    获取文件扩展名

    Args:
        filename: 文件名

    Returns:
        str: 扩展名（小写，不含点）
    """
    parts = filename.rsplit('.',1)
    if len(parts) == 2:
        return parts[1].lower()
    return ''

def _sanitize_filename(filename:str) -> str:
    """
    规范化文件名，去除空格和特殊字符，并防目录穿越。

    Args:
        filename: 原始文件名

    Returns:
        str: 规范化后的文件名（可能为空，调用方需校验）
    """
    # 去除空格
    sanitized = filename.replace(" ","_")
    # 取出其他可能导致问题的字符（含路径分隔符）
    for char in ['\\','/',':',"*",'"',"<",">","|"]:
        sanitized = sanitized.replace(char,"_")
    # 去除开头的点，防止 ".." 目录穿越与隐藏文件
    sanitized = sanitized.lstrip('.')
    return sanitized
