from typing import Any

from fastapi import APIRouter
from loguru import logger
from starlette.responses import JSONResponse

from app.config import config
from app.core.milvus_client import milvus_manager

router = APIRouter()


async def _check_redis() -> dict[str, str]:
    """Redis 连接探测（仅 harness_backend=redis 时调用）。

    每次探活新建一条连接、用完即关：健康检查频率低（默认 30s 一次），
    代价可忽略，且避免了与业务锁/审批共用连接的生命周期耦合。
    """
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(config.redis_url, decode_responses=True)
        try:
            await client.ping()
            return {"status": "connected", "message": "Redis 连接正常"}
        finally:
            await client.aclose()
    except Exception as e:
        logger.warning(f"Redis 健康检查失败：{e}")
        return {"status": "error", "message": f"Redis 检查失败：{e}"}


@router.get('/health/live')
def health_live():
    """存活探针：进程活着就返回 200，不检查任何依赖。

    供 Docker / K8s 存活检测使用；与 /health（就绪探针）区分——
    Milvus 暂时不可用不代表进程该被重启，只说明「尚未完全就绪」。
    """
    return JSONResponse(status_code=200, content={'code': 200, 'message': 'alive'})


@router.get('/health')
async def health_check():
    """就绪探针：检查服务本身与关键依赖（Milvus；harness_backend=redis 时再加 Redis）。"""
    health_data: dict[str, Any] = {        # pyright: ignore[reportExplicitAny]
        'service': config.app_name,
        'version': config.app_version,
        'status': 'healthy',
    }

    # 检查 Milvus 连接状态
    try:
        milvus_healthy = milvus_manager.health_check()
        milvus_status: str = 'connected' if milvus_healthy else 'disconnected'
        milvus_message: str = 'Milvus连接正常' if milvus_healthy else "Milvus 连接异常"
        health_data['milvus'] = {
            'status': milvus_status,
            'message': milvus_message,
        }
    except Exception as e:
        logger.warning(f'Milvus健康检查失败:{e}')
        health_data['milvus'] = {
            'status': 'error',
            'message': f"Milvus检查失败：{str(e)}",
        }

    # 检查 Redis 连接状态（多 worker 部署才启用：锁/审批落在 Redis，坏掉即写操作不可用）
    unhealthy = health_data['milvus']['status'] != 'connected'
    if config.harness_backend == 'redis':
        health_data['redis'] = await _check_redis()
        if health_data['redis']['status'] != 'connected':
            unhealthy = True

    overall_status = 'unhealthy' if unhealthy else 'healthy'
    status_code = 503 if unhealthy else 200
    health_data['status'] = overall_status
    if unhealthy:
        health_data['error'] = '依赖不可用'

    return JSONResponse(
        status_code=status_code,
        content={
            'code': status_code,
            'message': '服务运行正常' if overall_status == 'healthy' else '服务不可用',
            'data': health_data,
        },
    )
