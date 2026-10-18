"""A2 写操作后端：可插拔 restart/rollback 后端 + 未配置时 fail-closed。"""
import os
import types
from unittest.mock import AsyncMock

import pytest

from mcp_servers.write_ops import (
    MockWriteOpBackend,
    SystemdWriteOpBackend,
    WriteOpBackendNotConfigured,
    _UnconfiguredBackend,
    _create_backend,
    write_op_backend,
)


def _ok_run():
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def test_module_singleton_is_a_backend():
    assert hasattr(write_op_backend, "restart")
    assert hasattr(write_op_backend, "rollback")


def test_create_backend_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("WRITE_OP_BACKEND", raising=False)
    assert isinstance(_create_backend(), MockWriteOpBackend)


def test_create_backend_switches_by_env(monkeypatch):
    monkeypatch.setenv("WRITE_OP_BACKEND", "systemd")
    assert isinstance(_create_backend(), SystemdWriteOpBackend)
    monkeypatch.setenv("WRITE_OP_BACKEND", "k8s")
    assert isinstance(_create_backend(), _UnconfiguredBackend)
    monkeypatch.setenv("WRITE_OP_BACKEND", "tencent")
    assert isinstance(_create_backend(), _UnconfiguredBackend)


def test_create_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv("WRITE_OP_BACKEND", "bogus")
    with pytest.raises(WriteOpBackendNotConfigured):
        _create_backend()


def _make_releases(root: str, service: str, versions: list[str]) -> str:
    """在 root 下创建真实 releases/<ver> 目录（跨平台），返回 service_dir。"""
    service_dir = os.path.join(root, service)
    releases_dir = os.path.join(service_dir, "releases")
    for v in versions:
        os.makedirs(os.path.join(releases_dir, v))
    return service_dir


def _fake_current_link(mocker, current_link: str, target: str) -> None:
    """模拟 current 软链：Windows 无管理员/开发者模式无法建软链（WinError 1314），
    用 monkeypatch 只替 islink/readlink，让后端「软链检测 + 读目标」逻辑可跨平台测。
    真实的 os.symlink/os.replace 原子切换已在 _switch_current 里，生产 Linux 验证。"""
    mocker.patch("os.path.islink", side_effect=lambda p: p == current_link)
    mocker.patch("os.readlink", side_effect=lambda p: target)


def test_systemd_restart_runs_systemctl(mocker):
    run = mocker.patch("mcp_servers.write_ops.subprocess.run", return_value=_ok_run())
    backend = SystemdWriteOpBackend(deploy_root="/srv")
    r = backend.restart("data-sync-service", "CPU 过高")
    assert r["status"] == "restarted"
    assert r["unit"] == "data-sync-service.service"
    assert r["reason"] == "CPU 过高"
    assert r["mock"] is False
    run.assert_called_once()
    assert run.call_args.args[0] == ["systemctl", "restart", "data-sync-service.service"]


def test_systemd_restart_preserves_explicit_service_suffix(mocker):
    run = mocker.patch("mcp_servers.write_ops.subprocess.run", return_value=_ok_run())
    backend = SystemdWriteOpBackend(deploy_root="/srv")
    backend.restart("nginx.service")
    assert run.call_args.args[0] == ["systemctl", "restart", "nginx.service"]


def test_systemd_restart_fails_closed_on_error(mocker):
    mocker.patch(
        "mcp_servers.write_ops.subprocess.run",
        return_value=types.SimpleNamespace(returncode=1, stdout="", stderr="unit not found"),
    )
    backend = SystemdWriteOpBackend(deploy_root="/srv")
    with pytest.raises(RuntimeError):
        backend.restart("missing-svc")


def test_systemd_rejects_unsafe_service_name(mocker):
    run = mocker.patch("mcp_servers.write_ops.subprocess.run", return_value=_ok_run())
    backend = SystemdWriteOpBackend(deploy_root="/srv")
    with pytest.raises(ValueError):
        backend.restart("../etc/passwd")
    with pytest.raises(ValueError):
        backend.restart("a/b")
    run.assert_not_called()


def test_systemd_rollback_switches_to_previous_and_restarts(tmp_path, mocker):
    root = str(tmp_path)
    service_dir = _make_releases(root, "app", ["20261017", "20261018"])
    current_link = os.path.join(service_dir, "current")
    _fake_current_link(mocker, current_link, os.path.join(service_dir, "releases", "20261018"))

    backend = SystemdWriteOpBackend(deploy_root=root)
    switch = mocker.patch.object(backend, "_switch_current")
    run = mocker.patch.object(backend, "_run_systemctl")

    r = backend.rollback("app", "诊断后回滚")

    switch.assert_called_once_with(current_link, os.path.join(service_dir, "releases", "20261017"))
    run.assert_called_once_with("restart", "app.service")
    assert r["from_version"] == "20261018"
    assert r["to_version"] == "20261017"
    assert r["status"] == "rolled_back"
    assert r["mock"] is False


def test_systemd_rollback_goes_to_chronologically_previous(tmp_path, mocker):
    """已回滚过时（current=中间的版本，更新版本仍在），再回滚必须退到前一个、而非跳到最新。"""
    root = str(tmp_path)
    service_dir = _make_releases(root, "app", ["20261016", "20261017", "20261018"])
    current_link = os.path.join(service_dir, "current")
    _fake_current_link(mocker, current_link, os.path.join(service_dir, "releases", "20261017"))

    backend = SystemdWriteOpBackend(deploy_root=root)
    switch = mocker.patch.object(backend, "_switch_current")
    mocker.patch.object(backend, "_run_systemctl")

    r = backend.rollback("app")

    switch.assert_called_once_with(current_link, os.path.join(service_dir, "releases", "20261016"))
    assert r["from_version"] == "20261017"
    assert r["to_version"] == "20261016"


def test_systemd_rollback_fails_closed_without_symlink(tmp_path, mocker):
    mocker.patch("mcp_servers.write_ops.subprocess.run", return_value=_ok_run())
    # 只有版本目录、没有 current 软链
    os.makedirs(os.path.join(str(tmp_path), "app", "releases", "20261018"))
    backend = SystemdWriteOpBackend(deploy_root=str(tmp_path))
    with pytest.raises(WriteOpBackendNotConfigured):
        backend.rollback("app")


def test_systemd_rollback_fails_closed_single_release(tmp_path, mocker):
    mocker.patch("mcp_servers.write_ops.subprocess.run", return_value=_ok_run())
    service_dir = _make_releases(str(tmp_path), "app", ["20261018"])
    current_link = os.path.join(service_dir, "current")
    _fake_current_link(mocker, current_link, os.path.join(service_dir, "releases", "20261018"))
    backend = SystemdWriteOpBackend(deploy_root=str(tmp_path))
    with pytest.raises(WriteOpBackendNotConfigured):
        backend.rollback("app")


def test_systemd_rollback_fails_closed_when_already_oldest(tmp_path, mocker):
    """current 已是最早版本时回滚应 fail-closed，而不是循环到其它版本。"""
    mocker.patch("mcp_servers.write_ops.subprocess.run", return_value=_ok_run())
    service_dir = _make_releases(str(tmp_path), "app", ["20261017", "20261018"])
    current_link = os.path.join(service_dir, "current")
    _fake_current_link(mocker, current_link, os.path.join(service_dir, "releases", "20261017"))
    backend = SystemdWriteOpBackend(deploy_root=str(tmp_path))
    with pytest.raises(WriteOpBackendNotConfigured):
        backend.rollback("app")


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
