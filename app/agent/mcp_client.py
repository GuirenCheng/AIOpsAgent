"""
MCP 客户端管理
提供全局单例的 MCP 客户端，避免重复初始化
"""
from typing import Optional, Dict, List, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from loguru import logger
from mcp.types import CallToolResult, TextContent

from app.harness.guard import execute_guarded

# 全局MCP客户端
_mcp_client: Optional[MultiServerMCPClient] = None


async def retry_interceptor(
        request:MCPToolCallRequest,
        handler,
        max_retries:int = 3,
        delay:float = 1.0
):
    """MCP 工具调用安全收口（薄适配器）：委托 app.harness.guard.execute_guarded。

    六类护栏（幂等重试/超时/审计/资源锁/并发优先级/人工审批）已下沉到 execute_guarded，
    供 MCP 工具、本地工具、回滚 supervisor 共用。此处只负责把 execute_guarded 的
    GuardedResult 映射回 MCP 层需要的 CallToolResult。

    MCPToolCallRequest 结构：
    - name: str - 工具名称
    - args: dict[str, Any] - 工具参数
    - server_name: str - 服务器名称

    Args:
        request: MCP 工具调用请求
        handler: 实际的工具调用处理器（接收 request，返回 CallToolResult）
        max_retries: 最大重试次数（默认3次，仅对幂等工具生效）
        delay: 初始延迟时间（秒，默认1秒）

    Returns:
        CallToolResult: 工具调用结果或错误信息
    """
    result = await execute_guarded(
        request.name,
        request.args,
        request.server_name,
        lambda: handler(request),
        max_retries=max_retries,
        delay=delay,
    )
    if result.ok:
        return result.value
    return CallToolResult(
        content = [TextContent(type = 'text',text = result.error)],
        isError=True
    )


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
