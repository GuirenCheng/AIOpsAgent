"""RBAC 身份与授权（C2）：用户、角色、密码哈希、token 签发/校验、权限依赖。

设计要点：
- 身份：用户名 + 密码（PBKDF2-SHA256，stdlib 实现，零新增依赖）登录后签发无状态 HMAC token。
- 无状态 token：`base64url(payload).base64url(HMAC-SHA256(secret, payload))`，payload 含
  username/role/exp；服务端不存 session，天然支持将来多 worker。
- 授权：认证在纯 ASGI 中间件做（只写 principal_context，见 app/core/middleware.py），
  角色判断在端点用 `require_role(...)` 依赖声明。
- 账号挂钩：`audit.record()` 默认从 `get_actor()` 取当前用户名（仿照 request_id_context），
  使既有审计调用自动带账号，无需逐处传参。

本模块只依赖 app.config + loguru + fastapi，不 import app.harness，避免循环依赖。
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum

from fastapi import HTTPException
from loguru import logger

from app.config import config


class Role(str, Enum):
    """角色（权限从 admin → viewer 逐级收窄）。"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


@dataclass
class Principal:
    """一次请求里的「是谁、什么角色」。"""
    username: str
    role: Role


# 请求级身份上下文（纯 ASGI 中间件写入，端点依赖读取）；无请求上下文时为 None。
principal_context: ContextVar[Principal | None] = ContextVar("principal", default=None)


def get_current_principal() -> Principal | None:
    """返回当前请求的身份；无则 None。"""
    return principal_context.get()


def get_actor() -> str:
    """当前操作人用户名（审计用）；无身份时返回 '-'。"""
    p = principal_context.get()
    return p.username if p else "-"


# ---------------------------------------------------------------------------
# 密码哈希（PBKDF2-SHA256）
# ---------------------------------------------------------------------------

_PBKDF2_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    """生成密码哈希，格式 `pbkdf2_sha256$iterations$salt_hex$digest_hex`。"""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    """校验密码与存储哈希是否匹配（恒定时间比较）。"""
    try:
        scheme, iterations, salt, digest = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        calc = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        ).hex()
        return hmac.compare_digest(calc, digest)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# 无状态 token（HMAC-SHA256）
# ---------------------------------------------------------------------------

def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload: str) -> str:
    return _b64e(hmac.new(config.auth_token_secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())


def create_token(username: str, role: str, ttl_seconds: int | None = None) -> str:
    """为某用户签发一个带过期时间的 token。"""
    if not config.auth_token_secret:
        raise RuntimeError("AUTH_TOKEN_SECRET 未配置，无法签发 token")
    exp = int(time.time()) + (ttl_seconds if ttl_seconds is not None else config.auth_token_ttl_seconds)
    payload = _b64e(json.dumps({"sub": username, "role": role, "exp": exp}, separators=(",", ":")).encode("utf-8"))
    return f"{payload}.{_sign(payload)}"


def decode_token(token: str) -> dict | None:
    """校验 token 签名与过期，返回 payload dict；无效返回 None。"""
    if not config.auth_token_secret:
        return None
    try:
        payload, sig = token.split(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(payload)):
        return None
    try:
        data = json.loads(_b64d(payload))
    except (ValueError, TypeError):
        return None
    if int(data.get("exp", 0)) < time.time():
        return None
    return data


def principal_from_token(token: str) -> Principal | None:
    """把 token 解析成 Principal（签名/过期/角色名都合法才算通过）。"""
    data = decode_token(token)
    if not data:
        return None
    username = data.get("sub", "")
    try:
        role = Role(data.get("role", ""))
    except ValueError:
        return None
    if not username:
        return None
    return Principal(username=username, role=role)


# ---------------------------------------------------------------------------
# 用户存储（SQLite，独立于审计库）
# ---------------------------------------------------------------------------

_USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0
)
"""


def _parse_role(role) -> Role:
    """把 Role / str 归一化为 Role；非法值抛 ValueError。"""
    return role if isinstance(role, Role) else Role(str(role))


class UserStore:
    """用户账号存储（aiosqlite，全局单例 user_store）。"""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        self._conn = None

    async def _get_conn(self):
        if self._conn is None:
            import aiosqlite

            db_path = self._db_path or config.auth_db_path
            db_dir = os.path.dirname(os.path.abspath(db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self._conn = await aiosqlite.connect(db_path)
            # 多进程（uvicorn --workers N）安全：busy_timeout 防 SQLITE_BUSY，WAL 让多进程可并发写。
            await self._conn.execute("PRAGMA busy_timeout=5000")
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute(_USERS_SCHEMA)
            await self._conn.commit()
        return self._conn

    async def get_user(self, username: str) -> dict | None:
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT id, username, password_hash, role, created_at, disabled "
            "FROM users WHERE username = ?",
            (username,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        cols = ["id", "username", "password_hash", "role", "created_at", "disabled"]
        return dict(zip(cols, row))

    async def create_user(self, username: str, password: str, role) -> dict:
        """创建用户；用户名重复抛 ValueError。"""
        role = _parse_role(role)
        conn = await self._get_conn()
        try:
            await conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at, disabled) "
                "VALUES (?, ?, ?, ?, 0)",
                (username, hash_password(password), role.value, _now_iso()),
            )
            await conn.commit()
        except Exception as e:
            if "UNIQUE" in str(e).upper():
                raise ValueError(f"用户名 {username} 已存在") from e
            raise
        return await self.get_user(username)  # type: ignore[return-value]

    async def list_users(self) -> list[dict]:
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT id, username, role, created_at, disabled FROM users ORDER BY id"
        )
        rows = await cur.fetchall()
        cols = ["id", "username", "role", "created_at", "disabled"]
        return [dict(zip(cols, row)) for row in rows]

    async def delete_user(self, username: str) -> bool:
        conn = await self._get_conn()
        cur = await conn.execute("DELETE FROM users WHERE username = ?", (username,))
        await conn.commit()
        return cur.rowcount > 0

    async def set_role(self, username: str, role) -> bool:
        role = _parse_role(role)
        conn = await self._get_conn()
        cur = await conn.execute("UPDATE users SET role = ? WHERE username = ?", (role.value, username))
        await conn.commit()
        return cur.rowcount > 0

    async def set_password(self, username: str, password: str) -> bool:
        conn = await self._get_conn()
        cur = await conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (hash_password(password), username),
        )
        await conn.commit()
        return cur.rowcount > 0

    async def verify_credentials(self, username: str, password: str) -> Principal | None:
        """校验用户名密码，成功返回 Principal（禁用或错密码返回 None）。"""
        user = await self.get_user(username)
        if user is None or user["disabled"]:
            return None
        if not verify_password(password, user["password_hash"]):
            return None
        try:
            role = Role(user["role"])
        except ValueError:
            return None
        return Principal(username=user["username"], role=role)

    async def count(self) -> int:
        conn = await self._get_conn()
        cur = await conn.execute("SELECT COUNT(*) FROM users")
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# 全局单例（与 audit_logger / tool_registry 同风格）
user_store = UserStore()


# ---------------------------------------------------------------------------
# 初始管理员 + 权限依赖
# ---------------------------------------------------------------------------

async def ensure_bootstrap_admin() -> None:
    """首次启动（用户表为空）时创建初始管理员账号；密码为空则随机生成并打印一次。

    多 worker（uvicorn --workers N）下每个进程都会跑一次 lifespan：可能多个进程同时
    通过 count()==0 检查，再同时 create_user 撞 username UNIQUE。这里捕获该竞态，输掉
    的进程静默跳过（真正建号者才打印随机密码，避免误导）。
    """
    if not config.api_auth_enabled:
        return
    if await user_store.count() > 0:
        return
    username = config.auth_admin_username or "admin"
    password = config.auth_admin_password
    generated = not password
    if not password:
        password = secrets.token_urlsafe(12)
    try:
        await user_store.create_user(username, password, Role.ADMIN)
    except ValueError:
        # 多进程竞态：另一个 worker 已抢先建号（username UNIQUE 兜底），不算错误
        logger.info(f"初始管理员 {username} 已由其他进程创建，跳过")
        return
    if generated:
        logger.warning(
            f"首次启动：已创建初始管理员 {username}，临时密码：{password}（请立即记录并修改）"
        )
    logger.info(f"已创建初始管理员账号：{username}")


def require_role(*roles: Role):
    """FastAPI 依赖：要求当前请求具备所列角色之一；不传角色 = 任意已登录用户。

    - api_auth_enabled=False（开发）：放行并返回合成 admin，不破坏既有流程。
    - 否则读 principal_context：无身份 401、角色不足 403。
    """
    allowed = set(roles)

    def dependency() -> Principal:
        if not config.api_auth_enabled:
            return Principal(username="anonymous", role=Role.ADMIN)
        principal = principal_context.get()
        if principal is None:
            raise HTTPException(status_code=401, detail="未登录或凭证无效")
        if allowed and principal.role not in allowed:
            names = ", ".join(r.value for r in allowed)
            raise HTTPException(status_code=403, detail=f"权限不足：需要 {names} 角色")
        return principal

    return dependency
