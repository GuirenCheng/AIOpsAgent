"""Langfuse 全链路 trace 接入：一个 HTTP 请求 = 一条 trace。

复用 `request_id_context` 的 ContextVar 模式：`RequestIDMiddleware` 在请求进来时调
`setup_trace_context(request_id)` 开一条 trace（trace_id = request_id），请求结束
`teardown_trace_context` 收尾。8 处 ChatQwen 的 callbacks 统一走 `get_llm_callbacks()`，
把当前请求的 CallbackHandler 挂上，LLM 调用 + 工具调用（executor 里 ToolNode 自动捕获）
都会归入这条 trace，可在 Langfuse 自托管 UI 里按请求回放整条链路。

fail-open：未启用 / 未装 langfuse / 缺 host 或 keys 时，`get_llm_callbacks()` 只返回
`[token_usage_callback]`，绝不影响主流程（与 token_usage 记账同一原则）。
"""
from contextvars import ContextVar, Token
from typing import Any, Optional

from loguru import logger

from app.config import config

# 当前请求的 Langfuse CallbackHandler（per-request，跨 await 传播）
_handler_context: ContextVar[Optional[Any]] = ContextVar("langfuse_handler", default=None)

# Langfuse client 进程级单例；初始化失败后置 _client_failed 避免每次都重试
_client: Optional[Any] = None
_client_failed: bool = False


def _enabled() -> bool:
    """是否启用 trace：开关 + host + 两把 key 齐备才算启用。"""
    return bool(
        config.langfuse_enabled
        and config.langfuse_host
        and config.langfuse_public_key
        and config.langfuse_secret_key
    )


def _get_client():
    """惰性初始化 Langfuse client；未启用 / 失败返回 None（fail-open）。"""
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed or not _enabled():
        return None
    try:
        from langfuse import Langfuse  # 惰性导入：未装依赖时只在真正启用时报错

        _client = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
        )
        logger.info(f"Langfuse trace 已启用（host={config.langfuse_host}）")
        return _client
    except Exception as e:  # noqa: BLE001 —— trace 是旁路，失败静默停用
        _client_failed = True
        logger.warning(f"Langfuse 初始化失败，trace 停用（不影响主流程）：{e}")
        return None


def setup_trace_context(request_id: str) -> Optional[Token]:
    """为当前请求开一条 trace（trace_id=request_id）。

    返回 ContextVar 的 reset token 供 teardown 复位；未启用 / 创建失败返回 None。
    request_id 默认是 uuid4().hex（32 位 hex，符合 Langfuse trace_id 格式）。
    """
    if _get_client() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler(trace_context={"trace_id": request_id})
        return _handler_context.set(handler)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Langfuse handler 创建失败，trace 停用：{e}")
        return None


def teardown_trace_context(token: Optional[Token]) -> None:
    """结束当前请求的 trace 上下文；token 为 None 时什么都不做。"""
    if token is None:
        return
    _handler_context.reset(token)


def get_trace_handler():
    """返回当前请求的 CallbackHandler（无请求上下文 / 未启用时为 None）。"""
    return _handler_context.get()


def get_llm_callbacks() -> list:
    """返回 LLM callbacks 列表：token_usage 始终在，langfuse handler 有则追加。

    token_usage_callback 惰性导入以断开 middleware → langfuse_trace → token_usage
    → middleware 的 import 环。
    """
    from app.services.token_usage import token_usage_callback

    callbacks = [token_usage_callback]
    handler = _handler_context.get()
    if handler is not None:
        callbacks.append(handler)
    return callbacks
