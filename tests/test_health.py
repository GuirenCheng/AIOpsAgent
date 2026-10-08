"""健康检查接口回归测试。

回归点：/health 此前计算了 health_data（含 Milvus 状态）却没有放进响应体，
本测试确保响应中包含 data 字段及其依赖状态。
"""
import asyncio
import json

from app.api.health import health_check


def test_health_check_returns_data_with_milvus_status():
    resp = asyncio.run(health_check())
    body = json.loads(resp.body)

    assert body["code"] in (200, 503)
    assert "data" in body, "健康检查响应必须包含 data 字段"

    data = body["data"]
    assert data["status"] in ("healthy", "unhealthy")
    assert "milvus" in data, "data 中必须包含 milvus 依赖状态"
    assert data["milvus"]["status"] in ("connected", "disconnected", "error")
