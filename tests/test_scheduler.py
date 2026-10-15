"""Milestone 2 并发上限 + 优先级调度：PrioritySemaphore 与 retry_interceptor 的并发门槛。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.harness.scheduler import PrioritySemaphore
from app.agent.mcp_client import retry_interceptor


def _request(name, args=None, server="test"):
    return SimpleNamespace(name=name, args=args or {}, server_name=server)


# ---------------------------------------------------------------------------
# PrioritySemaphore 单元测试（独立实例，不碰全局单例 tool_gate）
# ---------------------------------------------------------------------------

async def test_gate_basic_acquire_release():
    gate = PrioritySemaphore(2)
    await gate.acquire()
    await gate.acquire()
    gate.release()
    gate.release()


async def test_gate_capacity_blocks_when_full():
    gate = PrioritySemaphore(1)
    await gate.acquire()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gate.acquire(), timeout=0.05)
    gate.release()


async def test_gate_release_wakes_waiter():
    gate = PrioritySemaphore(1)
    await gate.acquire()

    acquired = asyncio.Event()

    async def waiter():
        await gate.acquire()
        acquired.set()
        gate.release()

    t = asyncio.create_task(waiter())
    await asyncio.sleep(0)  # waiter 已入队但未获得槽位
    assert not acquired.is_set()

    gate.release()  # 放行持有者 → 槽位转移给 waiter
    await t
    assert acquired.is_set()


async def test_gate_priority_order():
    gate = PrioritySemaphore(1)
    await gate.acquire()  # 占满唯一槽位

    order = []

    async def worker(name, prio):
        await gate.acquire(prio)
        order.append(name)
        gate.release()

    t_low = asyncio.create_task(worker("low", 0))
    await asyncio.sleep(0)  # low 先入队
    t_high = asyncio.create_task(worker("high", 10))
    await asyncio.sleep(0)  # high 后入队，但优先级更高

    gate.release()  # 应优先唤醒 high
    await asyncio.gather(t_low, t_high)

    assert order == ["high", "low"]


async def test_gate_fifo_within_same_priority():
    gate = PrioritySemaphore(1)
    await gate.acquire()

    order = []

    async def worker(name):
        await gate.acquire(0)
        order.append(name)
        gate.release()

    t1 = asyncio.create_task(worker("a"))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(worker("b"))
    await asyncio.sleep(0)

    gate.release()
    await asyncio.gather(t1, t2)

    assert order == ["a", "b"]


async def test_gate_cancelled_waiter_is_removed():
    gate = PrioritySemaphore(1)
    await gate.acquire()

    t = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)  # t 已入队
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    # 取消后队列应清空，release 归还容量而非转给死任务
    gate.release()
    await gate.acquire()  # 容量已空闲，可直接获取
    gate.release()


# ---------------------------------------------------------------------------
# retry_interceptor 的并发门槛集成
# ---------------------------------------------------------------------------

async def test_interceptor_rejects_when_concurrency_full(mocker):
    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    # 用容量 1 的新信号量替换全局 gate，便于制造「并发已满」
    mocker.patch("app.harness.guard.tool_gate", new=PrioritySemaphore(1))
    mocker.patch("app.agent.mcp_client.config.harness_concurrency_enabled", True)
    mocker.patch("app.agent.mcp_client.config.harness_tool_queue_wait_seconds", 0.05)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_handler(request):
        entered.set()
        await release.wait()
        return "ok"

    req1 = _request("search_log", {"topic_id": "t1"})
    req2 = _request("search_log", {"topic_id": "t2"})

    t1 = asyncio.create_task(retry_interceptor(req1, slow_handler, max_retries=1, delay=0))
    await entered.wait()  # t1 已占满唯一并发槽位并进入 handler

    r2 = await retry_interceptor(req2, slow_handler, max_retries=1, delay=0)
    assert r2.isError is True
    assert "系统繁忙" in r2.content[0].text

    release.set()
    r1 = await t1
    assert r1 == "ok"


async def test_interceptor_skips_gate_when_concurrency_disabled(mocker):
    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    gate = PrioritySemaphore(1)
    mocker.patch("app.harness.guard.tool_gate", new=gate)
    mocker.patch("app.agent.mcp_client.config.harness_concurrency_enabled", False)

    await gate.acquire()  # 占用唯一槽位，但并发已禁用，interceptor 应忽略 gate

    async def handler(request):
        return "ok"

    r = await retry_interceptor(_request("search_log", {"topic_id": "t"}), handler, max_retries=1, delay=0)
    assert r == "ok"

    gate.release()
