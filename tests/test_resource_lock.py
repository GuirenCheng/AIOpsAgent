"""Milestone 1 资源锁：ResourceLockManager 与 retry_interceptor 的资源互斥行为。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.harness.lock import ResourceLockManager
from app.agent.mcp_client import retry_interceptor


def _request(name, args=None, server="test"):
    return SimpleNamespace(name=name, args=args or {}, server_name=server)


# ---------------------------------------------------------------------------
# ResourceLockManager 单元测试（独立实例，不碰全局单例）
# ---------------------------------------------------------------------------

async def test_acquire_and_release():
    mgr = ResourceLockManager()
    assert await mgr.acquire("svc") is True
    mgr.release("svc")
    assert await mgr.acquire("svc") is True  # 释放后可再次获取
    mgr.release("svc")


async def test_fail_fast_on_contention():
    mgr = ResourceLockManager()
    assert await mgr.acquire("svc") is True
    assert await mgr.acquire("svc") is False  # 已被持有，fail-fast 不排队
    mgr.release("svc")


async def test_wait_timeout():
    mgr = ResourceLockManager()
    assert await mgr.acquire("svc") is True
    assert await mgr.acquire("svc", wait_timeout=0.05) is False  # 等待仍拿不到
    mgr.release("svc")


async def test_different_resources_are_independent():
    mgr = ResourceLockManager()
    assert await mgr.acquire("svc-a") is True
    assert await mgr.acquire("svc-b") is True  # 不同资源不互斥
    mgr.release("svc-a")
    mgr.release("svc-b")


# ---------------------------------------------------------------------------
# retry_interceptor 的资源锁集成
# ---------------------------------------------------------------------------

async def test_interceptor_conflicts_on_same_resource(mocker):
    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    # restart_service 是 HIGH 风险写操作，会触发审批门；本测试聚焦资源锁，关闭审批
    mocker.patch("app.agent.mcp_client.config.harness_approval_enabled", False)
    mocker.patch("app.harness.guard.config.harness_write_allowlist", "svc-x")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_handler(request):
        entered.set()
        await release.wait()
        return {"restarted": request.args["service_name"]}

    req1 = _request("restart_service", {"service_name": "svc-x"})
    req2 = _request("restart_service", {"service_name": "svc-x"})

    t1 = asyncio.create_task(retry_interceptor(req1, slow_handler, max_retries=1, delay=0))
    await entered.wait()  # 第一个调用已拿到锁并进入 handler
    t2 = asyncio.create_task(retry_interceptor(req2, slow_handler, max_retries=1, delay=0))

    r2 = await t2
    assert r2.isError is True  # 第二个被拒绝，未执行

    release.set()
    r1 = await t1
    assert r1 == {"restarted": "svc-x"}


async def test_read_only_tool_runs_concurrently(mocker):
    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())

    async def handler(request):
        return {"logs": request.args["topic_id"]}

    req1 = _request("search_log", {"topic_id": "t1"})
    req2 = _request("search_log", {"topic_id": "t2"})

    # search_log 无 target_resource，不加锁，两个调用都能完成
    r1, r2 = await asyncio.gather(
        retry_interceptor(req1, handler, max_retries=1, delay=0),
        retry_interceptor(req2, handler, max_retries=1, delay=0),
    )
    assert r1 == {"logs": "t1"}
    assert r2 == {"logs": "t2"}
