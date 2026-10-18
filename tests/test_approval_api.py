"""一键审批回调 API 集成测试（B2）：确认页 + 批准/拒绝执行（TestClient）。

验证：decision 端点免登录（靠签名 token），GET 只渲染不动作，POST 才真正
approve/deny；签名/过期/单号不符一律 400，不改变审批状态（fail-closed）。
"""
import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import approval as approval_api
from app.config import config
from app.core.auth import create_approval_token
from app.core.middleware import AuthMiddleware
from app.harness.approval import ApprovalManager, ApprovalStatus


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(approval_api.router, prefix="/api")
    return app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "api_auth_enabled", True)
    monkeypatch.setattr(config, "auth_token_secret", "test-secret")

    mgr = ApprovalManager()
    monkeypatch.setattr(approval_api, "approval_manager", mgr)
    monkeypatch.setattr("app.harness.approval.audit_logger.record", AsyncMock())

    with TestClient(_build_app()) as c:
        yield c


def _create_pending():
    return asyncio.run(
        approval_api.approval_manager.request(
            "restart_service", {"service_name": "web"}, "prod-01"
        )
    )


def test_decision_page_renders(client):
    aid = _create_pending()
    token = create_approval_token(aid)
    resp = client.get(f"/api/harness/approvals/{aid}/decision", params={"token": token})
    assert resp.status_code == 200
    assert "批准" in resp.text
    assert "拒绝" in resp.text
    assert "restart_service" in resp.text


def test_decision_page_rejects_invalid_token(client):
    aid = _create_pending()
    resp = client.get(f"/api/harness/approvals/{aid}/decision", params={"token": "bad.token"})
    assert resp.status_code == 400


def test_decision_page_rejects_token_for_other_id(client):
    aid = _create_pending()
    token = create_approval_token("appr-999")
    resp = client.get(f"/api/harness/approvals/{aid}/decision", params={"token": token})
    assert resp.status_code == 400


def test_decision_approve(client):
    aid = _create_pending()
    token = create_approval_token(aid)
    resp = client.post(
        f"/api/harness/approvals/{aid}/decision",
        data={"action": "approve", "token": token},
    )
    assert resp.status_code == 200
    assert "已批准" in resp.text
    decision = asyncio.run(approval_api.approval_manager.wait_for_decision(aid, timeout=0.1))
    assert decision == ApprovalStatus.APPROVED


def test_decision_deny(client):
    aid = _create_pending()
    token = create_approval_token(aid)
    resp = client.post(
        f"/api/harness/approvals/{aid}/decision",
        data={"action": "deny", "token": token},
    )
    assert resp.status_code == 200
    assert "已拒绝" in resp.text
    decision = asyncio.run(approval_api.approval_manager.wait_for_decision(aid, timeout=0.1))
    assert decision == ApprovalStatus.DENIED


def test_decision_post_rejects_invalid_token(client):
    aid = _create_pending()
    resp = client.post(
        f"/api/harness/approvals/{aid}/decision",
        data={"action": "approve", "token": "bad"},
    )
    assert resp.status_code == 400
    # 无效 token 不得改变审批状态（仍待审批）
    pending = asyncio.run(approval_api.approval_manager.list_pending())
    assert [p["id"] for p in pending] == [aid]


def test_decision_unknown_action(client):
    aid = _create_pending()
    token = create_approval_token(aid)
    resp = client.post(
        f"/api/harness/approvals/{aid}/decision",
        data={"action": "explode", "token": token},
    )
    assert resp.status_code == 400
