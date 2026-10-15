"""retry_interceptor 幂等感知重试行为（mock handler，无网络）。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.agent.mcp_client import retry_interceptor


def _request(name, args=None, server="test"):
    return SimpleNamespace(name=name, args=args or {}, server_name=server)


async def test_idempotent_tool_is_retried(mocker):
    mocker.patch("app.agent.mcp_client.audit_logger.record", new=AsyncMock())
    handler = AsyncMock(side_effect=[Exception("boom"), {"ok": True}])

    result = await retry_interceptor(_request("search_log"), handler, max_retries=2, delay=0)

    assert handler.await_count == 2
    assert result == {"ok": True}


async def test_non_idempotent_tool_is_not_retried(mocker):
    mocker.patch("app.agent.mcp_client.audit_logger.record", new=AsyncMock())
    # restart_service 是 HIGH 风险写操作，会触发审批门；本测试聚焦幂等语义，关闭审批
    mocker.patch("app.agent.mcp_client.config.harness_approval_enabled", False)
    handler = AsyncMock(side_effect=Exception("boom"))

    result = await retry_interceptor(_request("restart_service"), handler, max_retries=3, delay=0)

    assert handler.await_count == 1  # 写操作只执行一次
    assert result.isError is True


async def test_timeout_yields_error_and_audit_timeout(mocker):
    from app.harness.registry import tool_registry
    from app.harness.tool_spec import ToolSpec

    # 注册一个超时极短的临时工具，验证超时兜底；名字唯一、不会影响其它测试
    tool_registry.register(ToolSpec(name="_test_slow_tool", idempotent=False, timeout_seconds=0.05))
    record = mocker.patch("app.agent.mcp_client.audit_logger.record", new=AsyncMock())

    async def slow(_request):
        await asyncio.sleep(0.3)
        return {"ok": True}

    result = await retry_interceptor(_request("_test_slow_tool"), slow, max_retries=1, delay=0)

    assert result.isError is True
    statuses = [c.kwargs.get("status") for c in record.await_args_list]
    assert "timeout" in statuses
