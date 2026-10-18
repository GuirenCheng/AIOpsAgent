"""自动回滚 supervisor：写操作失败后按名触发回滚工具。

`guard._run_with_retry` 在写失败且 spec.rollback_tool 非空时，除落一条 rollback_required
审计（durable 留痕）外，还 enqueue 到本模块的 RollbackManager；后台 supervisor 循环消费
队列，按名调用回滚工具（走 get_mcp_client_with_retry → get_tools，即完整 harness 护栏）。

触发用进程内 asyncio.Queue（实时信号），审计仍是 durable 真相源。回滚工具本身若也失败，
因其契约 rollback_tool=None，不会再次入队，避免无限回滚。
"""
import asyncio
from contextlib import suppress
from typing import Any, Dict

from loguru import logger

from app.harness.audit import audit_logger
from app.harness.notify import notify_rollback_failure


class RollbackManager:
    """回滚任务队列 + 后台 supervisor（全局单例 rollback_manager）。"""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def enqueue(
        self,
        *,
        failed_tool: str,
        rollback_tool: str,
        args: Dict[str, Any] | None,
        server: str,
        actor: str = "",
    ) -> None:
        """入队一个回滚任务（供 guard._run_with_retry 在写失败时调用）。"""
        await self._queue.put({
            "failed_tool": failed_tool,
            "rollback_tool": rollback_tool,
            "args": args or {},
            "server": server,
            "actor": actor,
        })

    async def _run(self) -> None:
        """supervisor 主循环：顺序消费回滚任务（串行，避免回滚互相踩）。"""
        while True:
            item = await self._queue.get()
            try:
                await self._execute_rollback(item)
            except Exception as e:
                logger.error(f"回滚执行异常（忽略）：{e}")
            finally:
                self._queue.task_done()

    async def _execute_rollback(self, item: Dict[str, Any]) -> None:
        """按名调用回滚工具并审计结果。"""
        rollback_tool = item["rollback_tool"]
        # 懒导入：避免 harness → agent 的模块级循环依赖（mcp_client 顶层 import guard）
        from app.agent.mcp_client import get_mcp_client_with_retry

        try:
            client = await get_mcp_client_with_retry()
            tools = await client.get_tools()
        except Exception as e:
            await audit_logger.record(
                tool=item["failed_tool"], server=item["server"], args=item["args"],
                status="rollback_failed", error=f"获取回滚工具失败：{e}",
                actor=item.get("actor", ""),
            )
            await notify_rollback_failure(
                item["failed_tool"], item["rollback_tool"], f"获取回滚工具失败：{e}", item["server"]
            )
            return

        tool = next((t for t in tools if t.name == rollback_tool), None)
        if tool is None:
            await audit_logger.record(
                tool=item["failed_tool"], server=item["server"], args=item["args"],
                status="rollback_skipped", error=f"回滚工具 {rollback_tool} 未找到",
                actor=item.get("actor", ""),
            )
            await notify_rollback_failure(
                item["failed_tool"], item["rollback_tool"], f"回滚工具 {rollback_tool} 未找到", item["server"]
            )
            return

        try:
            # 经 retry_interceptor → execute_guarded，回滚本身被审计/锁/限流
            await tool.ainvoke(item["args"])
            await audit_logger.record(
                tool=rollback_tool, server=item["server"], args=item["args"],
                status="rollback_executed", error=str(item["failed_tool"]),
                actor=item.get("actor", ""),
            )
        except Exception as e:
            await audit_logger.record(
                tool=rollback_tool, server=item["server"], args=item["args"],
                status="rollback_failed", error=str(e),
                actor=item.get("actor", ""),
            )
            await notify_rollback_failure(item["failed_tool"], rollback_tool, str(e), item["server"])

    def start(self) -> None:
        """启动 supervisor 后台任务（幂等）。

        多 worker（uvicorn --workers N）下每个进程各起一份 supervisor 是**正确**的：
        回滚任务由「执行失败写操作的那个进程」enqueue、由同一进程消费，天然不跨进程，
        不会重复回滚或丢失（队列是进程内的，失败信号也产生于进程内）。
        """
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            logger.info("回滚 supervisor 已启动")

    async def stop(self) -> None:
        """停止 supervisor 后台任务。"""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
            logger.info("回滚 supervisor 已停止")


# 全局单例（与 tool_registry / audit_logger / resource_locks 同风格）
rollback_manager = RollbackManager()
