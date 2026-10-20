"""文档转换服务：把任意格式（PDF / Word / HTML / PPTX / XLSX / CSV）统一转成 Markdown 文本。

这是「任意格式 → Markdown」的前置一步，转换结果再交给 document_splitter_service 做
结构感知切块（按 Markdown 标题切）。解析由 MarkItDown（微软）负责，纯离线、不额外花
token（与成本主线一致；可选的 LLM 图片描述能力默认不开）。

扩展名列表是**唯一真相源**：file.py 的上传校验与 vector_index_service 的目录 glob 都从
这里 import，避免两处列表漂移。
"""
from pathlib import Path
from typing import Set

from loguru import logger

from app.config import config

# 直接读文本、不转换的格式
TEXT_EXTENSIONS: Set[str] = {"txt", "md"}
# 走 MarkItDown 转换的格式（依赖 pyproject 里的 markitdown[pdf,docx,pptx,xlsx] 提供后端）
CONVERT_EXTENSIONS: Set[str] = {"pdf", "docx", "html", "htm", "pptx", "xlsx", "csv"}
# 全部支持的扩展名（小写、无点），供上传校验与目录 glob 共用
SUPPORTED_EXTENSIONS: Set[str] = TEXT_EXTENSIONS | CONVERT_EXTENSIONS


def _extension(path) -> str:
    return Path(path).suffix.lower().lstrip(".")


def is_supported(path) -> bool:
    """判断文件扩展名是否受支持。"""
    return _extension(path) in SUPPORTED_EXTENSIONS


def load_markdown(path) -> str:
    """把文件读成 Markdown 文本。

    txt/md 直接读原文；其余格式走 MarkItDown 转换。转换失败抛 RuntimeError，由调用方
    （vector_index_service.index_single_file → index_directory）捕获后记为失败文件，
    fail-open 不阻断其余文件。

    非文本格式在 MarkItDown 未安装、或 config.doc_conversion_enabled=False 时给出清晰
    报错（惰性导入 + 开关，绝不 import 期崩溃）。
    """
    path = Path(path)
    ext = _extension(path)

    if ext not in SUPPORTED_EXTENSIONS:
        raise RuntimeError(
            f"不支持的文件格式：.{ext}（支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))}）"
        )

    if ext in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")

    if not config.doc_conversion_enabled:
        raise RuntimeError(
            f"文档转换已关闭（DOC_CONVERSION_ENABLED=false），不支持格式 .{ext}。"
            f"请上传 txt/md，或开启 DOC_CONVERSION_ENABLED"
        )

    try:
        from markitdown import MarkItDown  # 惰性导入：没装时只在真正转换时报错
    except ImportError as e:
        raise RuntimeError(
            f"缺少 markitdown 依赖，无法解析 .{ext}。请 pip install 'markitdown[pdf,docx,pptx,xlsx]'"
        ) from e

    try:
        result = MarkItDown().convert(str(path))
        # 兼容不同版本的返回字段名（markdown / text_content）
        text = getattr(result, "markdown", None) or getattr(result, "text_content", "") or ""
        text = (text or "").strip()
        if not text:
            raise RuntimeError(
                f"文档转换为空（.{ext} 可能为扫描件/纯图片，无文本层，暂不支持 OCR）"
            )
        return text
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001 —— 转成统一错误交给上层记录
        raise RuntimeError(f"文档转换失败（.{ext}）：{e}") from e
