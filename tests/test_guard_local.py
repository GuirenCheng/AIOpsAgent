"""本地工具收口：guard_local_tool 包装与 execute_guarded 的本地工具护栏。"""
from unittest.mock import AsyncMock

from langchain_core.tools import tool

from app.tools.tool_wrapper import (
    guard_local_tool,
    guarded_get_current_time,
    guarded_retrieve_knowledge,
)


@tool
def _add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b


@tool
def _boom(x: int) -> int:
    """总是失败（用于验证失败返回错误文案）。"""
    raise RuntimeError("boom")


async def test_guard_local_tool_preserves_name_and_description():
    wrapped = guard_local_tool(_add)
    assert wrapped.name == "_add"
    assert "相加" in wrapped.description


async def test_guard_local_tool_passes_through_value(mocker):
    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    wrapped = guard_local_tool(_add)
    result = await wrapped.ainvoke({"a": 1, "b": 2})
    assert result == 3


async def test_guard_local_tool_returns_error_text_on_failure(mocker):
    mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    wrapped = guard_local_tool(_boom)
    # 失败不抛出，而是返回错误文案（与工具内部 catch 返回错误字符串的风格一致）
    result = await wrapped.ainvoke({"x": 1})
    assert isinstance(result, str)
    assert "失败" in result


async def test_local_tool_audited_with_server_local(mocker):
    record = mocker.patch("app.harness.guard.audit_logger.record", new=AsyncMock())
    result = await guarded_get_current_time.ainvoke({})
    assert isinstance(result, str)
    # 成功调用写一条 server="local" 的审计
    success_calls = [c for c in record.call_args_list if c.kwargs.get("status") == "success"]
    assert len(success_calls) == 1
    assert success_calls[0].kwargs["server"] == "local"
    assert success_calls[0].kwargs["tool"] == "get_current_time"


async def test_guarded_singletons_have_expected_names():
    assert guarded_retrieve_knowledge.name == "retrieve_knowledge"
    assert guarded_get_current_time.name == "get_current_time"
