"""RBAC 核心（C2）：密码哈希、无状态 token、用户存储、权限依赖、初始管理员。"""
import pytest
from fastapi import HTTPException

from app.core import auth
from app.core.auth import (
    Principal,
    Role,
    UserStore,
    create_approval_token,
    create_token,
    decode_approval_token,
    decode_token,
    ensure_bootstrap_admin,
    hash_password,
    principal_from_token,
    require_role,
    verify_password,
)
from app.config import config


# ---------------------------------------------------------------------------
# 密码哈希
# ---------------------------------------------------------------------------

def test_password_hash_roundtrip():
    stored = hash_password("s3cret")
    assert stored.startswith("pbkdf2_sha256$")
    assert verify_password("s3cret", stored) is True
    assert verify_password("wrong", stored) is False


def test_password_hash_is_salted():
    # 同一密码两次哈希应不同（随机盐），但都能通过校验
    a = hash_password("pw")
    b = hash_password("pw")
    assert a != b
    assert verify_password("pw", a) and verify_password("pw", b)


def test_verify_password_rejects_garbage():
    assert verify_password("pw", "not-a-valid-hash") is False
    assert verify_password("pw", "") is False


# ---------------------------------------------------------------------------
# 无状态 token
# ---------------------------------------------------------------------------

def test_token_sign_and_verify(monkeypatch):
    monkeypatch.setattr(config, "auth_token_secret", "test-secret")
    token = create_token("alice", "operator", ttl_seconds=3600)
    p = principal_from_token(token)
    assert p == Principal(username="alice", role=Role.OPERATOR)


def test_token_expiry(monkeypatch):
    monkeypatch.setattr(config, "auth_token_secret", "test-secret")
    token = create_token("alice", "viewer", ttl_seconds=-1)  # 已过期
    assert decode_token(token) is None
    assert principal_from_token(token) is None


def test_token_rejects_wrong_secret(monkeypatch):
    monkeypatch.setattr(config, "auth_token_secret", "secret-a")
    token = create_token("alice", "admin")
    monkeypatch.setattr(config, "auth_token_secret", "secret-b")
    assert decode_token(token) is None


def test_token_rejects_unknown_role(monkeypatch):
    monkeypatch.setattr(config, "auth_token_secret", "test-secret")
    token = create_token("alice", "superuser")  # 非法角色名
    assert principal_from_token(token) is None


def test_create_token_requires_secret(monkeypatch):
    monkeypatch.setattr(config, "auth_token_secret", "")
    with pytest.raises(RuntimeError):
        create_token("alice", "viewer")


# ---------------------------------------------------------------------------
# 一键审批回调 token（B2）
# ---------------------------------------------------------------------------

def test_approval_token_roundtrip(monkeypatch):
    monkeypatch.setattr(config, "auth_token_secret", "test-secret")
    token = create_approval_token("appr-7", ttl_seconds=600)
    assert decode_approval_token(token) == "appr-7"


def test_approval_token_expiry(monkeypatch):
    monkeypatch.setattr(config, "auth_token_secret", "test-secret")
    token = create_approval_token("appr-7", ttl_seconds=-1)
    assert decode_approval_token(token) is None


def test_approval_token_rejects_wrong_secret(monkeypatch):
    monkeypatch.setattr(config, "auth_token_secret", "secret-a")
    token = create_approval_token("appr-7")
    monkeypatch.setattr(config, "auth_token_secret", "secret-b")
    assert decode_approval_token(token) is None


def test_approval_token_not_confusable_with_login_token(monkeypatch):
    """登录 token 不能当审批 token 用（purpose 不同）；审批 token 也不能当登录身份用（无 sub/role）。"""
    monkeypatch.setattr(config, "auth_token_secret", "test-secret")
    login = create_token("alice", "admin", ttl_seconds=600)
    assert decode_approval_token(login) is None
    assert decode_token(login) is not None

    approval = create_approval_token("appr-7", ttl_seconds=600)
    assert principal_from_token(approval) is None  # 不能解析成登录身份
    assert decode_approval_token(approval) == "appr-7"


def test_approval_token_requires_secret(monkeypatch):
    monkeypatch.setattr(config, "auth_token_secret", "")
    with pytest.raises(RuntimeError):
        create_approval_token("appr-7")
    assert decode_approval_token("x.y") is None


# ---------------------------------------------------------------------------
# 用户存储（临时库）
# ---------------------------------------------------------------------------

async def test_userstore_crud_and_verify(tmp_path):
    store = UserStore(db_path=str(tmp_path / "users.sqlite"))
    try:
        assert await store.count() == 0

        u = await store.create_user("alice", "pw", Role.OPERATOR)
        assert u["username"] == "alice"
        assert u["role"] == "operator"
        assert await store.count() == 1

        got = await store.get_user("alice")
        assert got is not None
        assert "password_hash" in got and got["password_hash"] != "pw"

        # 验密：正确密码返回 Principal，错误/禁用返回 None
        p = await store.verify_credentials("alice", "pw")
        assert p == Principal(username="alice", role=Role.OPERATOR)
        assert await store.verify_credentials("alice", "bad") is None
        assert await store.verify_credentials("nobody", "pw") is None

        # 改角色 + 改密码
        assert await store.set_role("alice", "admin") is True
        assert (await store.get_user("alice"))["role"] == "admin"
        assert await store.set_password("alice", "newpw") is True
        assert await store.verify_credentials("alice", "newpw") is not None

        # 列表 + 删除
        assert [u["username"] for u in await store.list_users()] == ["alice"]
        assert await store.delete_user("alice") is True
        assert await store.get_user("alice") is None
        assert await store.delete_user("alice") is False
    finally:
        await store.close()


async def test_userstore_rejects_duplicate_username(tmp_path):
    store = UserStore(db_path=str(tmp_path / "users.sqlite"))
    try:
        await store.create_user("bob", "pw", Role.VIEWER)
        with pytest.raises(ValueError):
            await store.create_user("bob", "pw2", Role.ADMIN)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 权限依赖 require_role
# ---------------------------------------------------------------------------

def test_require_role_off_returns_synthetic_admin(monkeypatch):
    monkeypatch.setattr(config, "api_auth_enabled", False)
    dep = require_role(Role.ADMIN)()
    assert dep == Principal(username="anonymous", role=Role.ADMIN)


def test_require_role_401_without_principal(monkeypatch):
    monkeypatch.setattr(config, "api_auth_enabled", True)
    dep = require_role(Role.OPERATOR, Role.ADMIN)
    with pytest.raises(HTTPException) as ei:
        dep()
    assert ei.value.status_code == 401


def test_require_role_403_insufficient(monkeypatch):
    monkeypatch.setattr(config, "api_auth_enabled", True)
    token = auth.principal_context.set(Principal(username="bob", role=Role.VIEWER))
    try:
        dep = require_role(Role.OPERATOR, Role.ADMIN)
        with pytest.raises(HTTPException) as ei:
            dep()
        assert ei.value.status_code == 403
    finally:
        auth.principal_context.reset(token)


def test_require_role_allows_and_empty_roles_any_authenticated(monkeypatch):
    monkeypatch.setattr(config, "api_auth_enabled", True)
    token = auth.principal_context.set(Principal(username="bob", role=Role.OPERATOR))
    try:
        assert require_role(Role.OPERATOR, Role.ADMIN)() == Principal("bob", Role.OPERATOR)
        # 不传角色 = 任意已登录
        assert require_role()() == Principal("bob", Role.OPERATOR)
    finally:
        auth.principal_context.reset(token)


# ---------------------------------------------------------------------------
# 初始管理员
# ---------------------------------------------------------------------------

async def test_bootstrap_admin_creates_once(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "api_auth_enabled", True)
    monkeypatch.setattr(config, "auth_admin_username", "root")
    monkeypatch.setattr(config, "auth_admin_password", "secret123")

    store = UserStore(db_path=str(tmp_path / "users.sqlite"))
    monkeypatch.setattr(auth, "user_store", store)
    try:
        await ensure_bootstrap_admin()
        assert await store.count() == 1
        assert await store.verify_credentials("root", "secret123") is not None

        # 再次调用不重复建号
        await ensure_bootstrap_admin()
        assert await store.count() == 1
    finally:
        await store.close()


async def test_bootstrap_admin_skips_when_auth_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "api_auth_enabled", False)
    store = UserStore(db_path=str(tmp_path / "users.sqlite"))
    monkeypatch.setattr(auth, "user_store", store)
    try:
        await ensure_bootstrap_admin()
        assert await store.count() == 0
    finally:
        await store.close()


async def test_bootstrap_admin_tolerates_duplicate_race(tmp_path, monkeypatch):
    """多 worker 同时启动时，输掉建号竞态的进程不应崩溃（create_user 撞 UNIQUE）。"""
    monkeypatch.setattr(config, "api_auth_enabled", True)
    monkeypatch.setattr(config, "auth_admin_username", "root")
    monkeypatch.setattr(config, "auth_admin_password", "secret123")

    store = UserStore(db_path=str(tmp_path / "users.sqlite"))
    monkeypatch.setattr(auth, "user_store", store)

    # 模拟竞态：count() 仍为 0（两个 worker 都通过了检查），但真正 INSERT 时撞 UNIQUE
    async def fake_create_user(username, password, role):
        raise ValueError(f"用户名 {username} 已存在")

    monkeypatch.setattr(store, "create_user", fake_create_user)
    try:
        await ensure_bootstrap_admin()  # 不应抛异常
    finally:
        await store.close()


async def test_userstore_connection_uses_wal_and_busy_timeout(tmp_path):
    store = UserStore(db_path=str(tmp_path / "users.sqlite"))
    try:
        conn = await store._get_conn()
        mode = (await (await conn.execute("PRAGMA journal_mode")).fetchone())[0]
        assert str(mode).lower() == "wal"
        timeout = (await (await conn.execute("PRAGMA busy_timeout")).fetchone())[0]
        assert timeout == 5000
    finally:
        await store.close()
