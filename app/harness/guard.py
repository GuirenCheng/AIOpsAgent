"""工具执行护栏核心（唯一收口点）：MCP 工具、本地工具、回滚 supervisor 共用。

把原 mcp_client.retry_interceptor 里硬耦合 MCPToolCallRequest 的护栏逻辑泛化为
execute_guarded(tool_name, args, server, handler)：handler 是零参 async callable，
返回原始结果；args 仅用于元数据/审计/取锁对象。retry_interceptor 退化为薄适配器，
本地工具经 app/tools/tool_wrapper.guard_local_tool 包装后也走这里。

护栏由外到内：审批门(M3) → 并发槽位(M2) → 资源锁(M1) → 执行(M0 幂等重试+超时+审计)。
审批门最外，等人工决定期间不占并发槽位 / 资源锁。
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from loguru import logger

from app.config import config
from app.harness.approval import ApprovalStatus, approval_manager, requires_approval
from app.harness.audit import audit_logger
from app.harness.lock import resource_locks
from app.harness.notify import notify_approval
from app.harness.registry import tool_registry
from app.harness.rollback import rollback_manager
from app.harness.scheduler import tool_gate


@dataclass
class GuardedResult:
    """execute_guarded 的结果：成功时 ok=True 且 value 为原始返回值，失败时携带 error 文案。"""
    ok: bool
    value: Any = None
    error: str = ""


def _extract_resource_from_args(args: dict, spec) -> str:
    """从工具参数中提取资源锁对象。

    spec.target_resource 指向一个参数名（如 restart_service 的 "service_name"），
    该参数的值即为锁对象。参数缺失/为空时回退到工具名（保守：同工具调用互斥）。
    """
    value = (args or {}).get(spec.target_resource)
    if value in (None, ""):
        return spec.name
    return str(value)


def _parse_write_allowlist() -> set[str]:
    """解析写操作服务白名单（逗号分隔，去空白；空 = 全部拒绝 fail-closed）。"""
    return {s.strip() for s in config.harness_write_allowlist.split(",") if s.strip()}


async def _run_with_retry(
    tool_name: str,
    args: dict,
    server: str,
    handler: Callable[[], Awaitable],
    spec,
    max_attempts: int,
    delay: float,
) -> GuardedResult:
    """M0 幂等感知重试 + 超时 + 审计：在已持有并发槽位/资源锁的前提下执行工具调用。"""
    last_error = None

    for attempt in range(max_attempts):
        started = time.perf_counter()
        try:
            logger.info(
                f"调用工具：{tool_name}(服务器：{server}, 第{attempt + 1}/{max_attempts} 次尝试)"
            )
            result = await asyncio.wait_for(handler(), timeout=spec.timeout_seconds)
            logger.info(f"工具 {tool_name} 调用成功")
            await audit_logger.record(
                tool=tool_name,
                server=server,
                args=args,
                status="success",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return GuardedResult(ok=True, value=result)

        except asyncio.TimeoutError:
            last_error = TimeoutError(f"工具 {tool_name} 执行超时（>{spec.timeout_seconds}s）")
            logger.warning(f"工具 {tool_name} 调用超时 (第{attempt + 1}/{max_attempts} 次)")
            await audit_logger.record(
                tool=tool_name,
                server=server,
                args=args,
                status="timeout",
                duration_ms=(time.perf_counter() - started) * 1000,
                error=str(last_error),
            )

        except Exception as e:
            last_error = e
            logger.warning(
                f"工具 {tool_name} 调用失败 (第{attempt + 1}/{max_attempts} 次):{str(e)}"
            )
            await audit_logger.record(
                tool=tool_name,
                server=server,
                args=args,
                status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
                error=str(e),
            )

        if attempt < max_attempts - 1:
            wait_time = delay * (2 ** attempt)  # 指数退避
            logger.info(f"等待 {wait_time:.1f} 秒后重试...")
            await asyncio.sleep(wait_time)

    # 所有尝试都失败
    error_msg = f"工具{tool_name} 在 {max_attempts}次尝试后仍然失败:{str(last_error)}"
    # 回滚挂点：写操作失败且声明了回滚工具时，提示执行回滚、审计留痕并触发自动回滚 supervisor
    if spec.rollback_tool:
        error_msg += f"；该操作支持回滚，请执行回滚工具 {spec.rollback_tool}"
        await audit_logger.record(
            tool=tool_name,
            server=server,
            args=args,
            status="rollback_required",
            error=f"建议执行回滚：{spec.rollback_tool}",
        )
        try:
            await rollback_manager.enqueue(
                failed_tool=tool_name,
                rollback_tool=spec.rollback_tool,
                args=args,
                server=server,
            )
        except Exception as e:
            logger.warning(f"回滚入队失败（忽略）：{e}")
    logger.error(error_msg)
    return GuardedResult(ok=False, error=error_msg)


async def execute_guarded(
    tool_name: str,
    args: dict,
    server: str,
    handler: Callable[[], Awaitable],
    *,
    max_retries: int = 3,
    delay: float = 1.0,
) -> GuardedResult:
    """唯一护栏收口点：审批门 → 并发槽位 → 资源锁 → 执行（幂等重试+超时+审计）。"""
    spec = tool_registry.get(tool_name)
    # 非幂等工具：单次执行，失败直接返回（不重试，防重复执行写操作）
    max_attempts = max_retries if spec.idempotent else 1

    # 写操作服务白名单（A2，最外层 fail-closed）：白名单外的服务直接拒绝，
    # 不进审批门 / 并发槽位 / 资源锁 / 执行，也不触发回滚（根本没动过生产）。
    if spec.requires_write_allowlist:
        resource = _extract_resource_from_args(args, spec)
        if resource not in _parse_write_allowlist():
            logger.warning(f"服务 {resource} 不在写操作白名单中，拒绝执行 {tool_name}")
            await audit_logger.record(
                tool=tool_name,
                server=server,
                args=args,
                status="denied",
                error=f"服务 {resource} 不在写操作白名单中",
            )
            return GuardedResult(
                ok=False,
                error=f"服务 {resource} 不在允许自动写操作的白名单中，已拒绝执行",
            )

    # 审批门（最外层）：高危操作先经人工审批，批准才放行，拒绝/超时 fail-closed。
    if config.harness_approval_enabled and requires_approval(
        spec.risk, config.harness_approval_risk_threshold
    ):
        approval_id = await approval_manager.request(tool=tool_name, args=args, server=server)
        await notify_approval(approval_id, tool_name, args, server)
        decision = await approval_manager.wait_for_decision(
            approval_id, timeout=config.harness_approval_timeout_seconds
        )
        if decision != ApprovalStatus.APPROVED:
            logger.warning(f"工具 {tool_name} 人工审批未通过（{decision.value}），拒绝执行")
            return GuardedResult(
                ok=False,
                error=f"工具 {tool_name} 需人工审批，审批未通过（{decision.value}），已拒绝执行",
            )

    # 并发槽位：满则按优先级排队等待，超时返回「系统繁忙」。
    gate_held = False
    if config.harness_concurrency_enabled:
        try:
            await asyncio.wait_for(
                tool_gate.acquire(spec.priority),
                timeout=config.harness_tool_queue_wait_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"系统繁忙（工具并发已满），拒绝执行工具 {tool_name}")
            await audit_logger.record(
                tool=tool_name,
                server=server,
                args=args,
                status="overloaded",
                error="系统繁忙（工具并发已满）",
            )
            return GuardedResult(ok=False, error="系统繁忙（工具并发已满），请稍后重试")
        gate_held = True

    try:
        # 资源锁（内层）：并发写同一资源时后者 fail-fast。
        resource = None
        locked = False
        if config.harness_lock_enabled and spec.target_resource:
            resource = _extract_resource_from_args(args, spec)
            if not await resource_locks.acquire(
                resource, wait_timeout=config.harness_lock_wait_seconds
            ):
                logger.warning(f"资源 {resource} 正被占用，拒绝执行工具 {tool_name}")
                await audit_logger.record(
                    tool=tool_name,
                    server=server,
                    args=args,
                    status="conflict",
                    error=f"资源 {resource} 正被其他操作占用",
                )
                return GuardedResult(
                    ok=False,
                    error=f"资源 {resource} 正被其他操作占用，已拒绝执行工具 {tool_name}",
                )
            locked = True

        try:
            return await _run_with_retry(tool_name, args, server, handler, spec, max_attempts, delay)
        finally:
            if locked:
                resource_locks.release(resource)
    finally:
        if gate_held:
            tool_gate.release()
