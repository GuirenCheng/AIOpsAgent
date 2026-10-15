"""Milestone 3 高危操作人工审批：ApprovalManager 与 retry_interceptor 的审批门。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.harness.approval import ApprovalManager, ApprovalStatus, requires_approval
from app.harness.tool_spec import RiskLevel
from app.agent.mcp_client import retry_interceptor


def _request(name, args=None, server="test"):
    return SimpleNamespace(name=name, args=args or {}, server_name=server)


# ---------------------------------------------------------------------------
# requires_approval 阈值判断（同步）
# ---------------------------------------------------------------------------

def test_requires_approval_threshold():
    assert requires_approval(RiskLevel.HIGH, "high") is True
    assert requires_approval(RiskLevel.MEDIUM, "high") is False
    assert requires_approval(RiskLevel.LOW, "high") is False
    assert requires_approval(RiskLevel.MEDIUM, "medium") is True
    assert requires_approval(RiskLevel.HIGH, "medium") is True
    # 非法阈值回退 HIGH（最保守）
    assert requires_approval(RiskLevel.HIGH, "bogus") is True
    assert requires_approval(RiskLevel.LOW, "bogus") is False


# ---------------------------------------------------------------------------
# ApprovalManager 单元测试（独立实例，不碰全局单例；审计 mock 掉避免落 SQLite）
# ---------------------------------------------------------------------------

async def test_request_and_approve(mocker):
    mocker.patch("app.harness.approval.audit_logger.record", new=AsyncMock())
    mgr = ApprovalManager()
    aid = await mgr.request("restart_service", {"service_name": "svc"})
    assert aid.startswith("appr-")

    async def approver():
        await asyncio.sleep(0)
        await mgr.approve(aid, "同意")

    t = asyncio.create_task(approver())
    decision = await mgr.wait_for_decision(aid, timeout=1.0)
    await t
    assert decision == ApprovalStatus.APPROVED


async def test_deny_then_cannot_reapprove(mocker):
    mocker.patch("app.harness.approval.audit_logger.record", new=AsyncMock())
    mgr = ApprovalManager()
    aid = await mgr.request("restart_service", {"service_name": "svc"})

    assert await mgr.deny(aid, "太危险") is True
    assert await mgr.wait_for_decision(aid, timeout=1.0) == ApprovalStatus.DENIED
    assert await mgr.approve(aid) is False  # 已处理，不能再批


async def test_wait_timeout_expires(mocker):
    mocker.patch("app.harness.approval.audit_logger.record", new=AsyncMock())
    mgr = ApprovalManager()
    aid = await mgr.request("restart_service", {"service_name": "svc"})
    assert await mgr.wait_for_decision(aid, timeout=0.05) == ApprovalStatus.EXPIRED


async def test_unknown_approval_fail_closed(mocker):
    mocker.patch("app.harness.approval.audit_logger.record", new=AsyncMock())
    mgr = ApprovalManager()
    assert await mgr.approve("appr-999") is False
    assert await mgr.deny("appr-999") is False
    assert await mgr.wait_for_decision("appr-999") == ApprovalStatus.EXPIRED


async def test_list_pending(mocker):
    mocker.patch("app.harness.approval.audit_logger.record", new=AsyncMock())
    mgr = ApprovalManager()
    a1 = await mgr.request("restart_service", {"service_name": "a"})
    a2 = await mgr.request("restart_service", {"service_name": "b"})
    await mgr.approve(a1)

    pending = await mgr.list_pending()
    assert [p["id"] for p in pending] == [a2]


# ---------------------------------------------------------------------------
# retry_interceptor 的审批门集成
# ---------------------------------------------------------------------------

async def test_interceptor_approves_high_risk_tool(mocker):
    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    mocker.patch("app.harness.guard.approval_manager.request", new=AsyncMock(return_value="appr-1"))
    mocker.patch("app.harness.guard.approval_manager.wait_for_decision", new=AsyncMock(return_value=ApprovalStatus.APPROVED))

    async def handler(request):
        return {"restarted": request.args["service_name"]}

    r = await retry_interceptor(_request("restart_service", {"service_name": "svc"}), handler, max_retries=1, delay=0)
    assert r == {"restarted": "svc"}


async def test_interceptor_denies_high_risk_tool(mocker):
    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    mocker.patch("app.harness.guard.approval_manager.request", new=AsyncMock(return_value="appr-1"))
    mocker.patch("app.harness.guard.approval_manager.wait_for_decision", new=AsyncMock(return_value=ApprovalStatus.DENIED))

    handler = AsyncMock()
    r = await retry_interceptor(_request("restart_service", {"service_name": "svc"}), handler, max_retries=1, delay=0)
    assert r.isError is True
    assert "审批" in r.content[0].text
    handler.assert_not_called()


async def test_interceptor_skips_approval_for_low_risk(mocker):
    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    req_mock = mocker.patch("app.harness.guard.approval_manager.request", new=AsyncMock())

    async def handler(request):
        return {"logs": request.args["topic_id"]}

    r = await retry_interceptor(_request("search_log", {"topic_id": "t1"}), handler, max_retries=1, delay=0)
    assert r == {"logs": "t1"}
    req_mock.assert_not_called()


async def test_interceptor_skips_approval_when_disabled(mocker):
    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    mocker.patch("app.agent.mcp_client.config.harness_approval_enabled", False)
    req_mock = mocker.patch("app.harness.guard.approval_manager.request", new=AsyncMock())

    async def handler(request):
        return {"restarted": request.args["service_name"]}

    r = await retry_interceptor(_request("restart_service", {"service_name": "svc"}), handler, max_retries=1, delay=0)
    assert r == {"restarted": "svc"}
    req_mock.assert_not_called()


async def test_interceptor_rollback_hint_on_failed_write(mocker):
    audit_mock = mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    mocker.patch("app.agent.mcp_client.config.harness_approval_enabled", False)

    async def failing_handler(request):
        raise RuntimeError("重启失败")

    r = await retry_interceptor(_request("restart_service", {"service_name": "svc"}), failing_handler, max_retries=1, delay=0)
    assert r.isError is True
    assert "回滚" in r.content[0].text

    rollback_audits = [c for c in audit_mock.call_args_list if c.kwargs.get("status") == "rollback_required"]
    assert len(rollback_audits) == 1
