"""健康检查接口回归测试。

回归点：/health 此前计算了 health_data（含 Milvus 状态）却没有放进响应体，
本测试确保响应中包含 data 字段及其依赖状态。
"""
import asyncio
import json

import fakeredis.aioredis as fakeredis
import redis.asyncio as aioredis

from app.api.health import _check_redis, health_check, health_live


def test_health_check_returns_data_with_milvus_status():
    resp = asyncio.run(health_check())
    body = json.loads(resp.body)

    assert body["code"] in (200, 503)
    assert "data" in body, "健康检查响应必须包含 data 字段"

    data = body["data"]
    assert data["status"] in ("healthy", "unhealthy")
    assert "milvus" in data, "data 中必须包含 milvus 依赖状态"
    assert data["milvus"]["status"] in ("connected", "disconnected", "error")


def test_health_live_returns_200():
    resp = health_live()
    body = json.loads(resp.body)
    assert body["code"] == 200
    assert body["message"] == "alive"


async def test_check_redis_connected(monkeypatch):
    r = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(aioredis, "from_url", lambda url, **kw: r)
    result = await _check_redis()
    assert result["status"] == "connected"


def test_health_check_includes_redis_when_backend_is_redis(monkeypatch):
    from app.api import health as health_mod

    monkeypatch.setattr(health_mod.config, "harness_backend", "redis")

    async def fake_check_redis():
        return {"status": "connected", "message": "ok"}

    monkeypatch.setattr(health_mod, "_check_redis", fake_check_redis)

    resp = asyncio.run(health_mod.health_check())
    body = json.loads(resp.body)
    assert "redis" in body["data"]
    assert body["data"]["redis"]["status"] == "connected"
