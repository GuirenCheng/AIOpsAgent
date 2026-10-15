"""全局工具并发上限 + 优先级调度（Milestone 2）。

单进程内用 PrioritySemaphore 限制「同时执行中的工具调用」数量，保护 MCP server
与下游系统不被并发打爆；并发满时按工具优先级（ToolSpec.priority）先放行高优先级，
同优先级 FIFO 保证公平、避免饿死。

多进程化时需替换为分布式限流 / 优先级队列（Redis），接口 acquire/release 保持不变，
调用方（mcp_client.retry_interceptor）无需改动。
"""
import asyncio
import itertools

from app.config import config


class PrioritySemaphore:
    """带优先级的并发信号量（进程内，全局单例 tool_gate）。

    - capacity: 最大同时持有数
    - priority 越大越优先；同优先级按入队顺序（FIFO）
    - acquire() 被取消时自动从等待队列移除，不残留
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("PrioritySemaphore capacity 必须 >= 1")
        self._capacity = capacity
        self._in_use = 0
        # 等待队列元素 [priority, seq, future]，按 (-priority, seq) 排序
        self._waiters: list[list] = []
        self._seq = itertools.count()

    async def acquire(self, priority: int = 0) -> None:
        """获取一个槽位；容量满时按优先级排队等待（可能被取消）。"""
        if self._in_use < self._capacity and not self._waiters:
            self._in_use += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append([priority, next(self._seq), fut])
        self._waiters.sort(key=lambda w: (-w[0], w[1]))
        try:
            await fut
        except asyncio.CancelledError:
            # 等待中被取消（如外层 wait_for 超时）：从队列移除，不残留
            self._waiters = [w for w in self._waiters if w[2] is not fut]
            raise

    def release(self) -> None:
        """释放一个槽位；有等待者时按优先级转移给最高优先级者，否则归还容量。"""
        # 先清理已取消（future 已 done）的等待者，避免把槽位转给死任务
        while self._waiters and self._waiters[0][2].done():
            self._waiters.pop(0)
        if self._waiters:
            # 槽位直接转移给等待者，_in_use 不变
            self._waiters.pop(0)[2].set_result(None)
        else:
            if self._in_use <= 0:
                raise ValueError("PrioritySemaphore released too many times")
            self._in_use -= 1


# 全局单例：容量从配置读一次（与 tool_registry / resource_locks 同风格）
tool_gate = PrioritySemaphore(config.harness_max_concurrent_tools)
