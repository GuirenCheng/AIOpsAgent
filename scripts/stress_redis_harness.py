#!/usr/bin/env python
"""C1 跨进程压测：真实 Redis 下的分布式锁互斥 + 跨进程审批。

需要真实 Redis 服务（如 `docker compose up -d redis`，或本地 redis-server）。
用法：
    python scripts/stress_redis_harness.py [REDIS_URL] [N_WORKERS] [ROUNDS]

两个用例：
1. 锁互斥：N 个 worker 进程并发对同一资源做「读-改-写」临界区。锁互斥正确时，
   最终计数器应 == 成功持锁次数（无丢失更新 / 无并发进临界区）。
2. 跨进程审批：进程 A 建审批单并阻塞等待，进程 B 批准后 A 被唤醒为 APPROVED。

退出码 0 = 全部通过；非 0 = 发现互斥被打破 / 丢失更新 / 审批未跨进程生效（fail-closed）。

注意：脚本用 multiprocessing 起真实多进程，与单进程内「两个 manager 实例」的单测不同，
专门验证锁/审批在「不同进程、各自独立事件循环、共用同一个 Redis」下的正确性。
"""
import asyncio
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

# 让脚本可直接运行（不需要 pip install）：把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_LOCK_RESOURCE = "stress-lock"
_COUNTER_KEY = "harness:stress:counter"
_ACQUIRED_KEY = "harness:stress:acquired"
_AID_KEY = "harness:stress:approval_id"
_RESULT_KEY = "harness:stress:approval_result"


def _patch_audit():
    """压测里把审计落库替换为 no-op，避免刷 data/audit.sqlite。"""
    from app.harness.redis_backends import audit_logger

    async def _noop(*_a, **_k):
        return None

    audit_logger.record = _noop


# ---------------------------------------------------------------------------
# 用例 1：分布式锁互斥（每个 worker 一个进程，连同一个 Redis）
# ---------------------------------------------------------------------------

async def _lock_worker(url: str, rounds: int) -> int:
    """返回本 worker 成功持锁并完成临界区操作的次数。"""
    from app.harness.redis_backends import RedisResourceLockManager

    import redis.asyncio as aioredis

    redis = aioredis.from_url(url, decode_responses=True)
    mgr = RedisResourceLockManager(url=url)
    acquired = 0
    for _ in range(rounds):
        if not await mgr.acquire(_LOCK_RESOURCE):
            await asyncio.sleep(0)  # 冲突：让出事件循环
            continue
        acquired += 1
        # 临界区：非原子「读-改-写」。锁正确时不会并发进入，故不丢更新；
        # 若互斥被打破，两个进程会读到同一个 v 再各写 v+1，最终计数偏小。
        v = await redis.get(_COUNTER_KEY) or 0
        await asyncio.sleep(0.001)
        await redis.set(_COUNTER_KEY, int(v) + 1)
        mgr.release(_LOCK_RESOURCE)
    await redis.incrby(_ACQUIRED_KEY, acquired)
    await redis.aclose()
    return acquired


def _lock_worker_entry(url: str, rounds: int):
    asyncio.run(_lock_worker(url, rounds))


def _run_lock_stress(url: str, n_workers: int, rounds: int) -> bool:
    import redis.asyncio as aioredis

    async def main():
        redis = aioredis.from_url(url, decode_responses=True)
        await redis.delete(_COUNTER_KEY, _ACQUIRED_KEY)
        await redis.aclose()

    asyncio.run(main())

    procs = [Process(target=_lock_worker_entry, args=(url, rounds)) for _ in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    async def summary():
        redis = aioredis.from_url(url, decode_responses=True)
        counter = int(await redis.get(_COUNTER_KEY) or 0)
        acquired = int(await redis.get(_ACQUIRED_KEY) or 0)
        await redis.delete(_COUNTER_KEY, _ACQUIRED_KEY)
        await redis.aclose()
        return counter, acquired

    counter, acquired = asyncio.run(summary())
    attempts = n_workers * rounds
    print(f"  [锁互斥] 进程={n_workers} 轮次={rounds} 尝试={attempts} "
          f"成功持锁={acquired} 计数器终值={counter}")
    if counter != acquired:
        print(f"  ❌ 丢失更新/互斥被打破：counter={counter} != acquired={acquired}")
        return False
    print(f"  ✅ 锁互斥正确：无丢失更新，{attempts - acquired} 次冲突被正确拒绝")
    return True


# ---------------------------------------------------------------------------
# 用例 2：跨进程审批（requester 建单等待，approver 在另一进程批准）
# ---------------------------------------------------------------------------

async def _requester(url: str):
    from app.harness.redis_backends import RedisApprovalManager

    import redis.asyncio as aioredis

    _patch_audit()
    redis = aioredis.from_url(url, decode_responses=True)
    mgr = RedisApprovalManager(url=url)
    aid = await mgr.request(tool="stress_tool", args={}, server="stress")
    await redis.set(_AID_KEY, aid)
    decision = await mgr.wait_for_decision(aid, timeout=20)
    await redis.set(_RESULT_KEY, decision.value)
    await redis.aclose()


async def _approver(url: str):
    from app.harness.redis_backends import RedisApprovalManager

    import redis.asyncio as aioredis

    _patch_audit()
    redis = aioredis.from_url(url, decode_responses=True)
    mgr = RedisApprovalManager(url=url)
    aid = None
    for _ in range(200):
        aid = await redis.get(_AID_KEY)
        if aid:
            break
        await asyncio.sleep(0.05)
    if not aid:
        await redis.aclose()
        raise RuntimeError("未等到 requester 写入审批单")
    await mgr.approve(aid, "压测批准")
    await redis.aclose()


def _requester_entry(url: str):
    asyncio.run(_requester(url))


def _approver_entry(url: str):
    asyncio.run(_approver(url))


def _run_approval_stress(url: str) -> bool:
    import redis.asyncio as aioredis

    async def clear():
        redis = aioredis.from_url(url, decode_responses=True)
        await redis.delete(_AID_KEY, _RESULT_KEY)
        await redis.aclose()

    asyncio.run(clear())

    requester = Process(target=_requester_entry, args=(url,))
    approver = Process(target=_approver_entry, args=(url,))
    requester.start()
    approver.start()
    requester.join()
    approver.join()

    async def result():
        redis = aioredis.from_url(url, decode_responses=True)
        value = await redis.get(_RESULT_KEY)
        await redis.delete(_AID_KEY, _RESULT_KEY)
        await redis.aclose()
        return value

    value = asyncio.run(result())
    print(f"  [跨进程审批] requester 等待到的决定 = {value}")
    if value != "approved":
        print("  ❌ 审批未跨进程生效（requester 未等到 approved）")
        return False
    print("  ✅ 审批跨进程生效：进程 B 批准，进程 A 被唤醒为 approved")
    return True


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("REDIS_URL", "redis://localhost:6379/0")
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    print(f"==> Redis 跨进程压测：{url}（进程={n_workers} 轮次={rounds}）")
    start = time.time()

    ok_lock = _run_lock_stress(url, n_workers, rounds)
    ok_approval = _run_approval_stress(url)

    elapsed = time.time() - start
    print(f"==> 用时 {elapsed:.1f}s")
    if ok_lock and ok_approval:
        print("✅ 全部通过：分布式锁互斥 + 跨进程审批均正确")
        return 0
    print("❌ 存在失败项，请检查上述输出")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
