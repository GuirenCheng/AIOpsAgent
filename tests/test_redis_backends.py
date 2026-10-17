"""Redis 后端：分布式资源锁 + 跨进程审批（fakeredis 注入，无真实 Redis）。"""
import asyncio
from unittest.mock import AsyncMock

import fakeredis.aioredis as fakeredis

from app.harness.approval import ApprovalStatus
from app.harness.redis_backends import RedisApprovalManager, RedisResourceLockManager


# ---------------------------------------------------------------------------
# 资源锁
# ---------------------------------------------------------------------------

async def test_lock_acquire_fail_fast_and_resource_independent():
    r = fakeredis.FakeRedis(decode_responses=True)
    mgr = RedisResourceLockManager(client=r)
    assert await mgr.acquire("svc") is True
    assert await mgr.acquire("svc") is False       # 已被持有，fail-fast
    assert await mgr.acquire("svc-b") is True      # 不同资源独立
    await r.aclose()


async def test_lock_release_frees_resource():
    r = fakeredis.FakeRedis(decode_responses=True)
    mgr = RedisResourceLockManager(client=r)
    await mgr.acquire("svc")
    mgr.release("svc")
    await asyncio.sleep(0.01)                      # 等后台 compare-and-delete 跑完
    assert await mgr.acquire("svc") is True
    await r.aclose()


async def test_lock_release_does_not_delete_other_holders_lock():
    r = fakeredis.FakeRedis(decode_responses=True)
    mgr_a = RedisResourceLockManager(client=r)
    mgr_b = RedisResourceLockManager(client=r)
    await mgr_a.acquire("svc")
    # 模拟他人/另一 worker 也来释放（token 不匹配 → 不应删除 A 的锁）
    mgr_b.release("svc")
    await asyncio.sleep(0.01)
    assert await mgr_a.acquire("svc") is False     # A 的锁仍在
    await r.aclose()


async def test_lock_wait_timeout_acquires_after_release():
    r = fakeredis.FakeRedis(decode_responses=True)
    mgr = RedisResourceLockManager(client=r)
    await mgr.acquire("svc")

    async def hold_and_release():
        await asyncio.sleep(0.1)
        mgr.release("svc")

    holder = asyncio.create_task(hold_and_release())
    acquired = await mgr.acquire("svc", wait_timeout=2.0)
    assert acquired is True
    await holder
    await r.aclose()


async def test_lock_cross_worker_mutual_exclusion():
    """多 worker：A 持有锁时，B（另一 manager 实例 = 另一进程）无法获取同一资源。"""
    r = fakeredis.FakeRedis(decode_responses=True)
    worker_a = RedisResourceLockManager(client=r)
    worker_b = RedisResourceLockManager(client=r)

    assert await worker_a.acquire("svc") is True
    assert await worker_b.acquire("svc") is False       # 跨 worker 互斥
    assert await worker_b.acquire("svc-other") is True  # 不同资源互不影响
    await r.aclose()


# ---------------------------------------------------------------------------
# 审批
# ---------------------------------------------------------------------------

async def test_approval_request_and_list_pending(mocker):
    mocker.patch("app.harness.redis_backends.audit_logger.record", new=AsyncMock())
    r = fakeredis.FakeRedis(decode_responses=True)
    mgr = RedisApprovalManager(client=r)

    aid = await mgr.request(tool="restart_service", args={"service_name": "svc"}, server="monitor")
    assert aid == "appr-1"

    pending = await mgr.list_pending()
    assert len(pending) == 1
    assert pending[0]["tool"] == "restart_service"
    assert pending[0]["args"] == {"service_name": "svc"}
    assert pending[0]["status"] == ApprovalStatus.PENDING.value
    await r.aclose()


async def test_approval_approve(mocker):
    mocker.patch("app.harness.redis_backends.audit_logger.record", new=AsyncMock())
    r = fakeredis.FakeRedis(decode_responses=True)
    mgr = RedisApprovalManager(client=r)

    aid = await mgr.request(tool="restart_service", args={}, server="monitor")
    assert await mgr.approve(aid, "ok") is True
    assert await mgr.approve(aid) is False                       # 已处理，二次失败
    assert await mgr.wait_for_decision(aid, timeout=0.1) == ApprovalStatus.APPROVED
    await r.aclose()


async def test_approval_deny(mocker):
    mocker.patch("app.harness.redis_backends.audit_logger.record", new=AsyncMock())
    r = fakeredis.FakeRedis(decode_responses=True)
    mgr = RedisApprovalManager(client=r)

    aid = await mgr.request(tool="restart_service", args={}, server="monitor")
    assert await mgr.deny(aid, "风险太高") is True
    assert await mgr.wait_for_decision(aid, timeout=0.1) == ApprovalStatus.DENIED
    await r.aclose()


async def test_approval_wait_timeout_expires(mocker):
    mocker.patch("app.harness.redis_backends.audit_logger.record", new=AsyncMock())
    r = fakeredis.FakeRedis(decode_responses=True)
    mgr = RedisApprovalManager(client=r)

    aid = await mgr.request(tool="restart_service", args={}, server="monitor")
    assert await mgr.wait_for_decision(aid, timeout=0.15) == ApprovalStatus.EXPIRED
    assert await mgr.list_pending() == []                        # 过期后不再 pending
    await r.aclose()


async def test_approval_missing_id_returns_expired(mocker):
    mocker.patch("app.harness.redis_backends.audit_logger.record", new=AsyncMock())
    r = fakeredis.FakeRedis(decode_responses=True)
    mgr = RedisApprovalManager(client=r)
    assert await mgr.wait_for_decision("appr-999", timeout=0.1) == ApprovalStatus.EXPIRED
    await r.aclose()


async def test_approval_cross_worker_decide(mocker):
    """多 worker：A 进程创建审批并阻塞等待，B 进程（另一 manager 实例）批准后 A 被唤醒。

    这是 C1 的核心场景——单 worker 时 approve/deny 都在同一个 manager 内；
    多 worker 时执行方与审批方是两个进程，靠 Redis 里的审批单状态沟通。
    """
    mocker.patch("app.harness.redis_backends.audit_logger.record", new=AsyncMock())
    r = fakeredis.FakeRedis(decode_responses=True)
    worker_a = RedisApprovalManager(client=r)
    worker_b = RedisApprovalManager(client=r)

    aid = await worker_a.request(
        tool="restart_service", args={"service_name": "svc"}, server="monitor"
    )

    async def approve_later():
        await asyncio.sleep(0.05)
        return await worker_b.approve(aid, "值班批准")

    approver = asyncio.create_task(approve_later())
    decision = await worker_a.wait_for_decision(aid, timeout=2.0)
    assert decision == ApprovalStatus.APPROVED
    assert await approver is True
    await r.aclose()


# ---------------------------------------------------------------------------
# 工厂选择
# ---------------------------------------------------------------------------

def test_factory_in_process_default(mocker):
    from app.harness import approval, lock

    mocker.patch("app.harness.lock.config.harness_backend", "in_process")
    mocker.patch("app.harness.approval.config.harness_backend", "in_process")
    assert type(lock._create_resource_locks()).__name__ == "ResourceLockManager"
    assert type(approval._create_approval_manager()).__name__ == "ApprovalManager"


def test_factory_redis(mocker):
    from app.harness import approval, lock

    mocker.patch("app.harness.lock.config.harness_backend", "redis")
    mocker.patch("app.harness.approval.config.harness_backend", "redis")
    assert type(lock._create_resource_locks()).__name__ == "RedisResourceLockManager"
    assert type(approval._create_approval_manager()).__name__ == "RedisApprovalManager"
