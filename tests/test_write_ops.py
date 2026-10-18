"""A2 写操作后端：可插拔 restart/rollback 后端 + 未配置时 fail-closed。"""
from unittest.mock import AsyncMock

import pytest

from mcp_servers.write_ops import (
    MockWriteOpBackend,
    WriteOpBackendNotConfigured,
    _UnconfiguredBackend,
    _create_backend,
    write_op_backend,
)


def test_module_singleton_is_a_backend():
    assert hasattr(write_op_backend, "restart")
    assert hasattr(write_op_backend, "rollback")


def test_create_backend_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("WRITE_OP_BACKEND", raising=False)
    assert isinstance(_create_backend(), MockWriteOpBackend)


def test_create_backend_switches_by_env(monkeypatch):
    monkeypatch.setenv("WRITE_OP_BACKEND", "k8s")
    assert isinstance(_create_backend(), _UnconfiguredBackend)
    monkeypatch.setenv("WRITE_OP_BACKEND", "systemd")
    assert isinstance(_create_backend(), _UnconfiguredBackend)
    monkeypatch.setenv("WRITE_OP_BACKEND", "tencent")
    assert isinstance(_create_backend(), _UnconfiguredBackend)


def test_create_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv("WRITE_OP_BACKEND", "bogus")
    with pytest.raises(WriteOpBackendNotConfigured):
        _create_backend()


def test_mock_restart(mocker):
    mocker.patch("mcp_servers.write_ops.time.sleep")
    r = MockWriteOpBackend().restart("svc")
    assert r["service_name"] == "svc"
    assert r["status"] == "restarted"
    assert r["reason"] == "manual"
    assert r["mock"] is True


def test_mock_rollback(mocker):
    mocker.patch("mcp_servers.write_ops.time.sleep")
    r = MockWriteOpBackend().rollback("svc", "回滚原因")
    assert r["status"] == "rolled_back"
    assert r["reason"] == "回滚原因"
    assert r["mock"] is True


def test_unconfigured_backend_fails_closed():
    backend = _UnconfiguredBackend("k8s")
    with pytest.raises(WriteOpBackendNotConfigured):
        backend.restart("svc")
    with pytest.raises(WriteOpBackendNotConfigured):
        backend.rollback("svc")


async def test_guard_denies_service_not_in_allowlist(mocker):
    """护栏白名单（fail-closed）：白名单外的服务直接拒绝，不执行、不进审批/回滚。"""
    from app.harness.guard import execute_guarded

    record = mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    mocker.patch("app.harness.guard.config.harness_write_allowlist", "allowed-svc")
    handler = AsyncMock()

    result = await execute_guarded(
        "restart_service", {"service_name": "evil-svc"}, "monitor", handler,
        max_retries=1, delay=0,
    )

    assert result.ok is False
    assert "白名单" in result.error
    handler.assert_not_called()
    denied = [c for c in record.call_args_list if c.kwargs.get("status") == "denied"]
    assert len(denied) == 1
