"""生产加固中间件：请求 ID、鉴权、限流。

全部实现为纯 ASGI 中间件（而非 BaseHTTPMiddleware），
避免对 SSE 流式响应造成缓冲 / 线程池问题（本应用大量使用 EventSourceResponse）。
"""
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextvars import ContextVar

from loguru import logger
from starlette.responses import JSONResponse

from app.config import config

# 请求 ID 上下文变量，供业务代码（如需）读取当前请求的追踪 ID
request_id_context: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """获取当前请求的追踪 ID（无请求上下文时返回 '-'）。"""
    return request_id_context.get()


def _client_ip(scope: dict) -> str:
    """提取客户端 IP，优先取反向代理透传的 X-Forwarded-For。"""
    headers = dict(scope.get("headers") or [])
    forwarded = headers.get(b"x-forwarded-for")
    if forwarded:
        return forwarded.decode("latin-1").split(",")[0].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


def _header(scope: dict, name: str) -> str | None:
    """按名读取请求头（大小写不敏感）。"""
    target = name.lower().encode("latin-1")
    for key, value in scope.get("headers") or []:
        if key.lower() == target:
            return value.decode("latin-1")
    return None


class RequestIDMiddleware:
    """为每个请求分配 / 透传 X-Request-ID，并记录访问日志与耗时。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _header(scope, "x-request-id") or uuid.uuid4().hex
        token = request_id_context.set(request_id)
        start = time.perf_counter()
        status_code = 0

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            request_id_context.reset(token)
            duration_ms = (time.perf_counter() - start) * 1000
            method = scope.get("method", "")
            path = scope.get("path", "")
            logger.info(
                f"access | {request_id} | {_client_ip(scope)} | "
                f"{method} {path} | {status_code} | {duration_ms:.1f}ms"
            )


class RateLimitMiddleware:
    """基于内存滑动窗口的单 IP 限流（进程内，无外部依赖）。"""

    def __init__(self, app):
        self.app = app
        self._hits: dict[str, deque] = defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not config.rate_limit_enabled:
            await self.app(scope, receive, send)
            return

        if not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return

        ip = _client_ip(scope)
        now = time.monotonic()
        window = 60.0
        limit = config.rate_limit_per_minute
        hits = self._hits[ip]

        # 清理窗口外的旧记录
        while hits and now - hits[0] > window:
            hits.popleft()

        if len(hits) >= limit:
            response = JSONResponse(
                status_code=429,
                content={"code": 429, "message": "请求过于频繁，请稍后再试", "data": None},
            )
            await response(scope, receive, send)
            return

        hits.append(now)
        await self.app(scope, receive, send)


class ApiKeyAuthMiddleware:
    """可选的 API Key 鉴权（作用于 /api/*，/health 等探活接口不拦截）。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not config.api_auth_enabled:
            await self.app(scope, receive, send)
            return

        if not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return

        provided = _header(scope, "x-api-key") or ""
        expected = config.api_auth_key or ""

        # 使用恒定时间比较，降低时序侧信道风险
        if not expected or not secrets.compare_digest(provided, expected):
            response = JSONResponse(
                status_code=401,
                content={"code": 401, "message": "未授权：缺少或错误的 API Key", "data": None},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
