"""进程内资源锁（Milestone 1）：防止并发写操作在同一资源上冲突。

当前部署为单进程（uvicorn 单 worker，见 Dockerfile CMD 无 --workers），进程内
asyncio.Lock 即可正确串行化对同一资源的并发写操作，无需引入 Redis。若未来改成
多 worker / 多进程，需把本模块替换为分布式锁（Redis/etcd），但 ResourceLockManager
的 acquire/release 接口保持不变，调用方（mcp_client.retry_interceptor）无需改动。

锁随协程生命周期自动释放（retry_interceptor 的 finally 兜底），进程崩溃时锁自然消失，
因此进程内锁无需显式 TTL 租约；租约仅对分布式锁有意义，留待多进程化时实现。
"""
import asyncio


class ResourceLockManager:
    """按资源名维护独立的 asyncio.Lock（进程内，全局单例 resource_locks）。"""

    def __init__(self) -> None:
        # asyncio 单线程 + 无 await 点，普通 dict 即可，无需额外互斥
        self._locks: dict[str, asyncio.Lock] = {}

    async def acquire(self, resource: str, wait_timeout: float = 0.0) -> bool:
        """尝试获取资源锁。

        wait_timeout <= 0 时 fail-fast（冲突即返回 False，不排队）；
        > 0 时最多等待该秒数，仍拿不到返回 False。
        """
        lock = self._locks.setdefault(resource, asyncio.Lock())
        if wait_timeout <= 0:
            if lock.locked():
                return False
            await lock.acquire()
            return True
        try:
            await asyncio.wait_for(lock.acquire(), timeout=wait_timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def release(self, resource: str) -> None:
        """释放资源锁（幂等：锁不存在或未被持有时不报错）。"""
        lock = self._locks.get(resource)
        if lock is not None and lock.locked():
            lock.release()


# 全局单例（与 tool_registry / audit_logger 同风格）
resource_locks = ResourceLockManager()
