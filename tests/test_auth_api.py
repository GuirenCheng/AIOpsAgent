"""认证 API 集成测试：中间件认证 + require_role 授权 + 登录流（TestClient）。"""
import asyncio

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api import auth as auth_api
from app.config import config
from app.core.auth import Principal, Role, UserStore, require_role
from app.core.middleware import AuthMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(auth_api.router, prefix="/api")

    @app.get("/api/admin_only")
    async def admin_only(_u: Principal = Depends(require_role(Role.ADMIN))):
        return {"code": 200, "data": {"username": _u.username}}

    return app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "api_auth_enabled", True)
    monkeypatch.setattr(config, "api_auth_key", "test-api-key")
    monkeypatch.setattr(config, "auth_token_secret", "test-secret")

    store = UserStore(db_path=str(tmp_path / "users.sqlite"))
    monkeypatch.setattr(auth_api, "user_store", store)

    async def seed():
        await store.create_user("admin", "adminpw", Role.ADMIN)
        await store.create_user("viewer", "viewerpw", Role.VIEWER)

    asyncio.run(seed())
    with TestClient(_build_app()) as c:
        yield c
    asyncio.run(store.close())


def _login(client, username, password):
    return client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )


def test_login_returns_token(client):
    resp = _login(client, "admin", "adminpw")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["username"] == "admin"
    assert body["data"]["role"] == "admin"
    assert body["data"]["token"]


def test_login_wrong_password_401(client):
    assert _login(client, "admin", "nope").status_code == 401


def test_me_with_bearer(client):
    token = _login(client, "viewer", "viewerpw").json()["data"]["token"]
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["data"] == {"username": "viewer", "role": "viewer"}


def test_no_credential_401(client):
    assert client.get("/api/admin_only").status_code == 401


def test_viewer_forbidden_on_admin_endpoint(client):
    token = _login(client, "viewer", "viewerpw").json()["data"]["token"]
    resp = client.get("/api/admin_only", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_x_api_key_backward_compat_admin(client):
    resp = client.get("/api/admin_only", headers={"X-API-Key": "test-api-key"})
    assert resp.status_code == 200
    assert resp.json()["data"]["username"] == "apikey"
