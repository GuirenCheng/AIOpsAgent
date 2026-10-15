"""
MCP 客户端管理
提供全局单例的 MCP 客户端，避免重复初始化
"""
import asyncio
import time
from typing import Optional, Dict, List, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from loguru import logger
from mcp.types import CallToolResult, TextContent

from app.harness.audit import audit_logger
from app.harness.registry import tool_registry
from app.harness.lock import resource_locks
from app.harness.scheduler import tool_gate
from app.harness.approval import ApprovalStatus, approval_manager, requires_approval

# 全局MCP客户端
_mcp_client: Optional[MultiServerMCPClient] = None


def _extract_resource(request: MCPToolCallRequest, spec) -> str:
    """从工具参数中提取资源锁对象。

    spec.target_resource 指向一个参数名（如 restart_service 的 "service_name"），
    该参数的值即为锁对象。参数缺失/为空时回退到工具名（保守：同工具调用互斥）。
    """
    args = request.args or {}
    value = args.get(spec.target_resource)
    if value in (None, ""):
        return spec.name
    return str(value)


async def _run_with_retry(
        request:MCPToolCallRequest,
        handler,
        spec,
        max_attempts:int,
        delay:float,
):
    """M0 幂等感知重试 + 超时 + 审计：在已持有并发槽位/资源锁的前提下执行工具调用。"""
    last_error = None

    for attempt in range(max_attempts):
        started = time.perf_counter()
        try:
            logger.info(
                f"调用MCP工具：{request.name}"
                f"(服务器：{request.server_name}, 第{attempt + 1}/{max_attempts} 次尝试)"
            )
            result = await asyncio.wait_for(handler(request), timeout=spec.timeout_seconds)
            logger.info(f"MCP 工具 {request.name} 调用成功")
            await audit_logger.record(
                tool=request.name,
                server=request.server_name,
                args=request.args,
                status="success",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return result

        except asyncio.TimeoutError:
            last_error = TimeoutError(f"工具 {request.name} 执行超时（>{spec.timeout_seconds}s）")
            logger.warning(
                f"MCP 工具 {request.name} 调用超时 (第{attempt + 1}/{max_attempts} 次)"
            )
            await audit_logger.record(
                tool=request.name,
                server=request.server_name,
                args=request.args,
                status="timeout",
                duration_ms=(time.perf_counter() - started) * 1000,
                error=str(last_error),
            )

        except Exception as e:
            last_error = e
            logger.warning(
                f"MCP 工具 {request.name} 调用失败 "
                f"(第{attempt + 1}/{max_attempts} 次):{str(e)}"
            )
            await audit_logger.record(
                tool=request.name,
                server=request.server_name,
                args=request.args,
                status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
                error=str(e),
            )

        # 如果不是最后一次尝试，等待后重试
        if attempt < max_attempts-1:
            wait_time = delay * (2**attempt) #指数退避
            logger.info(f"等待 {wait_time:.1f} 秒后重试...")
            await asyncio.sleep(wait_time)

    # 所有尝试都失败，返回错误结果而不是抛出异常
    error_msg = f"工具{request.name} 在 {max_attempts}次尝试后仍然失败:{str(last_error)}"
    # Milestone 3 回滚挂点：写操作失败且声明了回滚工具时，提示执行回滚并审计留痕
    if spec.rollback_tool:
        error_msg += f"；该操作支持回滚，请执行回滚工具 {spec.rollback_tool}"
        await audit_logger.record(
            tool=request.name,
            server=request.server_name,
            args=request.args,
            status="rollback_required",
            error=f"建议执行回滚：{spec.rollback_tool}",
        )
    logger.error(error_msg)
    return CallToolResult(
        content = [TextContent(type = 'text',text = error_msg)],
        isError=True
    )


async def retry_interceptor(
        request:MCPToolCallRequest,
        handler,
        max_retries:int = 3,
        delay:float = 1.0
):
    """MCP 工具调用安全收口：这是所有 MCP 工具调用的唯一收口点
    （RAG 的 create_agent 与 AIOps 的 executor/planner/replanner 都经 get_mcp_client_with_retry），
    在此统一施加六类护栏：

    1. 幂等感知重试——只有 tool_registry 中 spec.idempotent=True 的工具才自动重试；
       写操作（如 restart_service）单次执行、失败即返回，避免「超时但已成功 + 重试」导致重复执行。
    2. 超时——asyncio.wait_for 按 spec.timeout_seconds 兜底，防工具挂起拖垮 agent。
    3. 审计——每次调用的工具/参数/状态/耗时/request_id 写入 audit_logger（SQLite）。
    4. 资源锁（Milestone 1）——spec.target_resource 非空时，整个操作持有该资源锁；
       并发写同一资源时后者被拒绝（fail-fast），避免并发写冲突。
    5. 并发上限 + 优先级调度（Milestone 2）——申请全局并发槽位（tool_gate），
       并发满时按 spec.priority 排队、高优先级先入槽；等待超时返回「系统繁忙」。
    6. 人工审批（Milestone 3）——spec.risk 达阈值（默认 HIGH）的写操作先创建审批单
       并等待人工批准，批准才放行、拒绝/超时 fail-closed 不执行。

    护栏由外到内嵌套：审批门（M3）→ 并发槽位（M2）→ 资源锁（M1）→ 执行（M0）。
    审批门最外，等待人工决定期间不占并发槽位 / 资源锁，避免长时间阻塞其它调用。

    MCPToolCallRequest 结构：
    - name: str - 工具名称
    - args: dict[str, Any] - 工具参数
    - server_name: str - 服务器名称

    Args:
        request: MCP 工具调用请求
        handler: 实际的工具调用处理器
        max_retries: 最大重试次数（默认3次，仅对幂等工具生效）
        delay: 初始延迟时间（秒，默认1秒）

    Returns:
        CallToolResult: 工具调用结果或错误信息
    """
    spec = tool_registry.get(request.name)
    # 非幂等工具：单次执行，失败直接返回（不重试，防重复执行写操作）
    max_attempts = max_retries if spec.idempotent else 1

    # Milestone 3 审批门（最外层）：高危操作先经人工审批，批准才放行，拒绝/超时 fail-closed。
    # 审批生命周期已由 approval_manager 审计，这里只做「未通过则拒绝执行」。
    if config.harness_approval_enabled and requires_approval(spec.risk, config.harness_approval_risk_threshold):
        approval_id = await approval_manager.request(
            tool=request.name,
            args=request.args,
            server=request.server_name,
        )
        decision = await approval_manager.wait_for_decision(
            approval_id,
            timeout=config.harness_approval_timeout_seconds,
        )
        if decision != ApprovalStatus.APPROVED:
            logger.warning(f"工具 {request.name} 人工审批未通过（{decision.value}），拒绝执行")
            return CallToolResult(
                content=[TextContent(type='text', text=f"工具 {request.name} 需人工审批，审批未通过（{decision.value}），已拒绝执行")],
                isError=True,
            )

    # Milestone 2 并发槽位：先申请全局并发槽位，满则按优先级排队等待。
    gate_held = False
    if config.harness_concurrency_enabled:
        try:
            await asyncio.wait_for(
                tool_gate.acquire(spec.priority),
                timeout=config.harness_tool_queue_wait_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"系统繁忙（工具并发已满），拒绝执行工具 {request.name}")
            await audit_logger.record(
                tool=request.name,
                server=request.server_name,
                args=request.args,
                status="overloaded",
                error="系统繁忙（工具并发已满）",
            )
            return CallToolResult(
                content=[TextContent(type='text', text="系统繁忙（工具并发已满），请稍后重试")],
                isError=True,
            )
        gate_held = True

    try:
        # Milestone 1 资源锁（内层）：并发写同一资源时后者被拒绝（fail-fast）。
        resource = None
        locked = False
        if config.harness_lock_enabled and spec.target_resource:
            resource = _extract_resource(request, spec)
            if not await resource_locks.acquire(resource, wait_timeout=config.harness_lock_wait_seconds):
                logger.warning(f"资源 {resource} 正被占用，拒绝执行工具 {request.name}")
                await audit_logger.record(
                    tool=request.name,
                    server=request.server_name,
                    args=request.args,
                    status="conflict",
                    error=f"资源 {resource} 正被其他操作占用",
                )
                return CallToolResult(
                    content=[TextContent(type='text', text=f"资源 {resource} 正被其他操作占用，已拒绝执行工具 {request.name}")],
                    isError=True,
                )
            locked = True

        try:
            return await _run_with_retry(request, handler, spec, max_attempts, delay)
        finally:
            if locked:
                resource_locks.release(resource)
    finally:
        if gate_held:
            tool_gate.release()



# 从配置文件读取MCP服务器配置
from app.config import config

# 使用配置文件中定义的完整MCP服务区配置
DEFAULT_MCP_SERVERS = config.mcp_servers

async def get_mcp_client(
        servers:Optional[Dict[str,Dict[str,str]]] = None,
        tool_interceptors:Optional[List] = None,
        force_new:bool = False,
) -> MultiServerMCPClient:
    """
    获取或初始化 MCP 客户端（不带重试拦截器）

    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）

    从 langchain-mcp-adapters 0.1.0 开始，MultiServerMCPClient 不再支持作为上下文管理器使用。
    直接创建实例即可使用。

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）

    Returns:
        MultiServerMCPClient: MCP 客户端实例
    """
    global _mcp_client

    # 如果请求新实例，直接创建并返回（不缓存）
    if force_new:
        logger.info("创建新的MCP客户端实例（非单例）")
        client = _create_mcp_client(
            servers or DEFAULT_MCP_SERVERS,
            tool_interceptors
        )
        # 不再需要 __aenter__(),直接返回即可
        return client

    # 单例模式：如果已存在，直接返回
    if _mcp_client is None:
        logger.info("初始化全局MCP客户端...")
        _mcp_client = _create_mcp_client(
            servers or DEFAULT_MCP_SERVERS,
            tool_interceptors
        )
        # 不再需要 __aenter__()，直接使用即可
        logger.info("全局MCP客户端初始化完成")

    return _mcp_client


async def get_mcp_client_with_retry(
        servers:Optional[Dict[str,Dict[str,str]]] = None,
        tool_interceptors:Optional[List] = None,
        force_new:bool = False,
) -> MultiServerMCPClient:
    """
    获取或初始化带重试功能的 MCP 客户端

    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）
    重试拦截器会自动添加到拦截器列表的开头

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表（会在重试拦截器之后添加）
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）

    Returns:
        MultiServerMCPClient: 带重试功能的 MCP 客户端实例
    """
    # 构建拦截器列表：重试拦截器在最前面
    interceptors = [retry_interceptor]
    if tool_interceptors:
        interceptors.extend(tool_interceptors)

    return await get_mcp_client(
        servers = servers,
        tool_interceptors = interceptors,
        force_new = force_new,
    )

def _create_mcp_client(
        servers:Dict[str,Dict[str,str]],
        tool_interceptors:Optional[List] = None,
)->MultiServerMCPClient:
    """
    创建 MCP 客户端实例

    Args:
        servers: MCP 服务器配置
        tool_interceptors: 工具拦截器列表

    Returns:
        MultiServerMCPClient: 未初始化的客户端实例
    """
    # MultiServerMCPClient 的第一个参数直接接收 servers 配置字典
    # 格式: {server_name: {"transport": "...", "url": "..."}}
    kwargs:Dict[str,Any] = {}

    if tool_interceptors:
        kwargs['tool_interceptors'] = tool_interceptors

    # 第一个参数是servers配置，直接传递
    return MultiServerMCPClient(servers,**kwargs) # type:ignore[arg-type]

