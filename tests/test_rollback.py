"""自动回滚 supervisor：写失败信号入队 + 按名触发回滚工具。"""
from unittest.mock import AsyncMock

from app.harness.registry import tool_registry
from app.harness.rollback import RollbackManager
from app.harness.tool_spec import RiskLevel


def _item(**overrides):
    base = {
        "failed_tool": "restart_service",
        "rollback_tool": "rollback_service",
        "args": {"service_name": "svc"},
        "server": "monitor",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# rollback_service 契约
# ---------------------------------------------------------------------------

def test_rollback_service_spec_is_medium_and_non_recursive():
    spec = tool_registry.get("rollback_service")
    assert spec.read_only is False
    assert spec.risk == RiskLevel.MEDIUM          # 低于默认 HIGH 阈值 → 自动回滚无人审批
    assert spec.rollback_tool is None             # 回滚自身失败不再触发回滚（防无限回滚）
    assert spec.target_resource == "service_name"


# ---------------------------------------------------------------------------
# RollbackManager._execute_rollback
# ---------------------------------------------------------------------------

async def test_execute_rollback_invokes_tool_by_name(mocker):
    record = mocker.patch("app.harness.rollback.audit_logger.record", new=AsyncMock())

    fake_tool = mocker.MagicMock()
    fake_tool.name = "rollback_service"
    fake_tool.ainvoke = AsyncMock(return_value={"status": "rolled_back"})
    fake_client = mocker.MagicMock()
    fake_client.get_tools = AsyncMock(return_value=[fake_tool])
    mocker.patch("app.agent.mcp_client.get_mcp_client_with_retry", new=AsyncMock(return_value=fake_client))

    mgr = RollbackManager()
    await mgr._execute_rollback(_item())

    fake_tool.ainvoke.assert_awaited_once_with({"service_name": "svc"})
    executed = [c for c in record.call_args_list if c.kwargs.get("status") == "rollback_executed"]
    assert len(executed) == 1
    assert executed[0].kwargs["tool"] == "rollback_service"


async def test_execute_rollback_skips_when_tool_missing(mocker):
    record = mocker.patch("app.harness.rollback.audit_logger.record", new=AsyncMock())

    fake_client = mocker.MagicMock()
    fake_client.get_tools = AsyncMock(return_value=[])  # 无回滚工具
    mocker.patch("app.agent.mcp_client.get_mcp_client_with_retry", new=AsyncMock(return_value=fake_client))

    mgr = RollbackManager()
    await mgr._execute_rollback(_item())

    skipped = [c for c in record.call_args_list if c.kwargs.get("status") == "rollback_skipped"]
    assert len(skipped) == 1


# ---------------------------------------------------------------------------
# 写失败 → enqueue 信号
# ---------------------------------------------------------------------------

async def test_write_failure_enqueues_rollback(mocker):
    from app.harness.guard import execute_guarded

    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    mocker.patch("app.harness.guard.config.harness_approval_enabled", False)
    mocker.patch("app.harness.guard.config.harness_write_allowlist", "svc")
    enqueue = mocker.patch("app.harness.guard.rollback_manager.enqueue", new=AsyncMock())

    async def failing():
        raise RuntimeError("重启失败")

    result = await execute_guarded(
        "restart_service", {"service_name": "svc"}, "monitor", failing,
        max_retries=1, delay=0,
    )

    assert result.ok is False
    enqueue.assert_awaited_once()
    assert enqueue.await_args.kwargs["failed_tool"] == "restart_service"
    assert enqueue.await_args.kwargs["rollback_tool"] == "rollback_service"
    assert enqueue.await_args.kwargs["args"] == {"service_name": "svc"}


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------

async def test_start_and_stop():
    mgr = RollbackManager()
    mgr.start()
    assert mgr._task is not None
    await mgr.stop()
    assert mgr._task is None
