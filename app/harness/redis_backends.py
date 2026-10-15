"""Redis 可切换后端：分布式资源锁 + 跨进程审批（多 worker / 分布式）。

默认单 worker 部署走进程内原语（app/harness/lock.py、app/harness/approval.py），
本模块提供 Redis 实现，供 config.harness_backend == 'redis' 时由工厂懒加载切换。

- 资源锁用 SET NX PX 实现带 TTL 租约的互斥；release 用 Lua compare-and-delete 只删
  自己持有的锁（防误删他人锁），进程崩溃后锁由 TTL 自动过期，避免死锁。
- 审批单落 Redis hash（harness:approval:{id}），id 由 INCR 计数生成；wait_for_decision
  用 0.1s 轮询 HGET status（简单够用，跨进程决定可被感知），超时置 EXPIRED。

依赖 redis>=5.0.0（异步客户端 redis.asyncio）；测试用 fakeredis 注入 client 覆盖，
无需真实 Redis 服务。
"""
import asyncio
import json
import uuid
from datetime import datetime, timezone

from loguru import logger

from app.config import config
from app.core.middleware import request_id_context
from app.harness.approval import ApprovalStatus
from app.harness.audit import audit_logger

_LOCK_PREFIX = "harness:lock:"
_APPROVAL_PREFIX = "harness:approval:"
_APPROVAL_COUNTER = "harness:approval_counter"  # 无冒号 → 不被 approval:* 的 SCAN 命中

# compare-and-delete：仅当 value 仍是自己的 token 时才删除（防误删他人锁）
_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


def _s(value) -> str:
    """客户端未开 decode_responses 时把 bytes 解码为 str。"""
    if isinstance(value, bytes):
        return value.decode()
    return value


def _parse_args(raw) -> dict:
    try:
        return json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {}


class RedisResourceLockManager:
    """分布式资源锁（Redis SET NX PX + Lua compare-and-delete）。

    与进程内 ResourceLockManager 的 acquire/release 签名一致，调用方（guard.py）零改动。
    """

    def __init__(self, url: str | None = None, client=None) -> None:
        if client is not None:
            self._redis = client
            self._owns_client = False
        else:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(url or config.redis_url, decode_responses=True)
            self._owns_client = True
        self._tokens: dict[str, str] = {}
        self._pending: set[asyncio.Task] = set()

    @staticmethod
    def _key(resource: str) -> str:
        return f"{_LOCK_PREFIX}{resource}"

    async def acquire(self, resource: str, wait_timeout: float = 0.0) -> bool:
        """尝试获取资源锁。wait_timeout <= 0 fail-fast；> 0 最多等待该秒数。"""
        token = uuid.uuid4().hex
        key = self._key(resource)
        ttl_ms = max(1, int(config.harness_lock_redis_ttl_seconds * 1000))
        if wait_timeout <= 0:
            got = await self._redis.set(key, token, nx=True, px=ttl_ms)
            if got:
                self._tokens[resource] = token
            return bool(got)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_timeout
        while True:
            got = await self._redis.set(key, token, nx=True, px=ttl_ms)
            if got:
                self._tokens[resource] = token
                return True
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(0.05)

    def release(self, resource: str) -> None:
        """释放锁（同步、fire-and-forget）：与进程内类签名一致，调用方无需改。

        真正删除在后台任务里做 Lua compare-and-delete；即便任务未及时执行，
        锁也会在 TTL 内自动过期，不会永久卡死。
        """
        token = self._tokens.pop(resource, None)
        if token is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # 无事件循环（不应发生），锁留给 TTL 过期
        task = loop.create_task(self._release_async(resource, token))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _release_async(self, resource: str, token: str) -> None:
        try:
            await self._redis.eval(_RELEASE_LUA, 1, self._key(resource), token)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"释放 Redis 资源锁失败（忽略，等 TTL 过期）：{e}")

    async def aclose(self) -> None:
        if self._owns_client and self._redis is not None:
            await self._redis.aclose()
        self._redis = None


class RedisApprovalManager:
    """跨进程人工审批（Redis hash + INCR 计数，接口与进程内 ApprovalManager 对齐）。"""

    def __init__(self, url: str | None = None, client=None) -> None:
        if client is not None:
            self._redis = client
            self._owns_client = False
        else:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(url or config.redis_url, decode_responses=True)
            self._owns_client = True

    @staticmethod
    def _key(approval_id: str) -> str:
        return f"{_APPROVAL_PREFIX}{approval_id}"

    async def _load(self, approval_id: str) -> dict | None:
        data = await self._redis.hgetall(self._key(approval_id))
        if not data:
            return None
        return {_s(k): _s(v) for k, v in data.items()}

    async def request(self, tool: str, args=None, server: str = "") -> str:
        """创建审批单并返回审批单号；同时写一条 pending_approval 审计。"""
        n = await self._redis.incr(_APPROVAL_COUNTER)
        approval_id = f"appr-{n}"
        await self._redis.hset(
            self._key(approval_id),
            mapping={
                "tool": tool,
                "server": server,
                "args": json.dumps(args or {}, ensure_ascii=False, default=str),
                "request_id": request_id_context.get() or "",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": ApprovalStatus.PENDING.value,
                "reason": "",
            },
        )
        await audit_logger.record(
            tool=tool, server=server, args=args,
            status="pending_approval", error=approval_id,
        )
        return approval_id

    async def _decide(self, approval_id: str, status: ApprovalStatus, reason: str = "") -> bool:
        data = await self._load(approval_id)
        if data is None or data.get("status") != ApprovalStatus.PENDING.value:
            return False
        await self._redis.hset(
            self._key(approval_id),
            mapping={"status": status.value, "reason": reason},
        )
        await audit_logger.record(
            tool=data.get("tool", ""),
            server=data.get("server", ""),
            args=_parse_args(data.get("args")),
            status=f"approval_{status.value}",
            error=reason or approval_id,
        )
        return True

    async def approve(self, approval_id: str, reason: str = "") -> bool:
        return await self._decide(approval_id, ApprovalStatus.APPROVED, reason)

    async def deny(self, approval_id: str, reason: str = "") -> bool:
        return await self._decide(approval_id, ApprovalStatus.DENIED, reason)

    async def wait_for_decision(self, approval_id: str, timeout: float = 300.0) -> ApprovalStatus:
        """等待审批决定；超时或单不存在均视为「过期/拒绝」（fail-closed）。"""
        key = self._key(approval_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            data = await self._load(approval_id)
            if data is None:
                return ApprovalStatus.EXPIRED
            status = data.get("status")
            if status != ApprovalStatus.PENDING.value:
                return ApprovalStatus(status)
            if loop.time() >= deadline:
                await self._redis.hset(key, "status", ApprovalStatus.EXPIRED.value)
                await audit_logger.record(
                    tool=data.get("tool", ""),
                    server=data.get("server", ""),
                    args=_parse_args(data.get("args")),
                    status="approval_expired",
                    error=approval_id,
                )
                return ApprovalStatus.EXPIRED
            await asyncio.sleep(0.1)

    async def list_pending(self) -> list[dict]:
        """列出所有待审批单（供 API / 控制台展示）。"""
        out: list[dict] = []
        async for key in self._redis.scan_iter(match=f"{_APPROVAL_PREFIX}*", count=100):
            data = {_s(k): _s(v) for k, v in (await self._redis.hgetall(key)).items()}
            if data.get("status") == ApprovalStatus.PENDING.value:
                out.append({
                    "id": key[len(_APPROVAL_PREFIX):],
                    "tool": data.get("tool", ""),
                    "args": _parse_args(data.get("args")),
                    "server": data.get("server", ""),
                    "request_id": data.get("request_id", ""),
                    "created_at": data.get("created_at", ""),
                    "status": ApprovalStatus.PENDING.value,
                    "reason": data.get("reason", ""),
                })
        return out

    async def aclose(self) -> None:
        if self._owns_client and self._redis is not None:
            await self._redis.aclose()
        self._redis = None
