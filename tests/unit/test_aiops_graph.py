"""
Plan-Execute-Replan 重构的确定性单测（mock LLM / 工具，无网络）。

覆盖：
- 结构化计划（PlanStep）与观察（Observation）的模型行为
- planner / executor / replanner 三节点的结构化 I/O 与证据驱动决策
- SSE 事件格式化（plan 扁平化为 List[str]、current_step 为 str）
- 图级端到端（验证 PlanStep/Observation 过 checkpoint 序列化）

所有 LLM/MCP/知识检索均被 mock，运行不依赖网络与 API key。
"""
import importlib
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

# 注意：app.agent.aiops.__init__ 会把节点【函数】以模块名重导出（from .planner import planner），
# 导致 `from app.agent.aiops import planner` 拿到的是函数而非模块，故用 importlib 取真实模块。
planner_mod = importlib.import_module("app.agent.aiops.planner")
executor_mod = importlib.import_module("app.agent.aiops.executor")
replanner_mod = importlib.import_module("app.agent.aiops.replanner")

from app.agent.aiops.state import (
    PlanStep,
    Observation,
    coerce_step,
    coerce_observation,
)
from app.services.aiops_service import AIOpsService


# ---------------------------------------------------------------------------
# mock 助手
# ---------------------------------------------------------------------------

def _state(plan, past_steps=None, response="", input_text="诊断某服务的告警"):
    return {
        "input": input_text,
        "plan": plan,
        "past_steps": past_steps or [],
        "response": response,
    }


def _stub_mcp_client(mocker, tools=None):
    '''返回一个 get_tools() 为空/给定列表的假 MCP 客户端'''
    client = mocker.MagicMock()
    client.get_tools = AsyncMock(return_value=tools or [])
    return client


def _structured_output_side_effect(plan=None, act=None, response=None):
    '''
    ChatQwen.with_structured_output(schema) 的 side_effect。

    返回真实 Runnable（RunnableLambda），因为 planner/replanner 用
    `prompt | llm.with_structured_output(...)` 组合，裸 MagicMock 会破坏 __or__。
    '''

    def _factory(schema):
        async def _fn(_input):
            if schema is planner_mod.Plan and plan is not None:
                return plan
            if schema is replanner_mod.Act and act is not None:
                return act
            if schema is replanner_mod.Response and response is not None:
                return response
            raise AssertionError(f"unexpected schema {schema}")

        return RunnableLambda(_fn)

    return _factory


def _patch_executor_llm(mocker, first_msg, final_msg):
    '''mock executor 的 LLM 路径：bind_tools().ainvoke 两次调用 + ToolNode'''
    mock_cls = mocker.patch("app.agent.aiops.executor.ChatQwen")
    llm_with_tools = mocker.MagicMock()
    llm_with_tools.ainvoke = AsyncMock(side_effect=[first_msg, final_msg])
    mock_cls.return_value.bind_tools.return_value = llm_with_tools

    tool_node_stub = mocker.MagicMock()
    tool_node_stub.ainvoke = AsyncMock(return_value={
        "messages": [ToolMessage(content="cpu=95", tool_call_id="call_1", name="query_cpu_metrics")]
    })
    mocker.patch("app.agent.aiops.executor.ToolNode", return_value=tool_node_stub)
    mocker.patch(
        "app.agent.aiops.executor.get_mcp_client_with_retry",
        new=AsyncMock(return_value=_stub_mcp_client(mocker)),
    )
    return mock_cls, llm_with_tools, tool_node_stub


def _patch_replanner(mocker, act=None, response=None):
    '''mock replanner 的 LLM + MCP'''
    mock_cls = mocker.patch("app.agent.aiops.replanner.ChatQwen")
    mock_cls.return_value.with_structured_output.side_effect = _structured_output_side_effect(
        act=act, response=response
    )
    mocker.patch(
        "app.agent.aiops.replanner.get_mcp_client_with_retry",
        new=AsyncMock(return_value=_stub_mcp_client(mocker)),
    )
    return mock_cls


# ---------------------------------------------------------------------------
# 1~2. 模型行为
# ---------------------------------------------------------------------------

def test_planstep_evidence_alias_normalization():
    assert PlanStep(description="x", evidence="log").evidence == "logs"
    assert PlanStep(description="x", evidence="report").evidence == "synthesis"
    assert PlanStep(description="x", evidence="CPU").evidence == "metrics"
    with pytest.raises(ValidationError):
        PlanStep(description="x", evidence="invalid_type")


def test_planstep_and_observation_json_roundtrip():
    step = PlanStep(
        description="查 CPU", evidence="metrics",
        tool_name="query_cpu_metrics", tool_args={"service_name": "s"},
    )
    assert PlanStep.model_validate_json(step.model_dump_json()) == step

    obs = Observation(
        step_description="d", evidence="logs",
        tool_names=["search_log"], success=True, result_summary="r",
    )
    assert Observation.model_validate_json(obs.model_dump_json()) == obs


def test_coerce_step_handles_str_and_dict():
    assert coerce_step("旧步骤") == PlanStep(description="旧步骤", evidence="synthesis")
    assert coerce_step({"description": "d", "evidence": "logs"}).evidence == "logs"
    assert coerce_step({"description": "d"}).evidence == "synthesis"
    assert coerce_step(PlanStep(description="d", evidence="metrics")).evidence == "metrics"


def test_coerce_observation_handles_legacy_tuple():
    obs = coerce_observation(("查日志", "找到 3 条错误"))
    assert obs.step_description == "查日志"
    assert obs.evidence == "synthesis"
    assert obs.success is True

    failed = coerce_observation(("查日志", "执行失败：超时"))
    assert failed.success is False


# ---------------------------------------------------------------------------
# 3~4. planner
# ---------------------------------------------------------------------------

async def test_planner_returns_structured_plan(mocker):
    mock_cls = mocker.patch("app.agent.aiops.planner.ChatQwen")
    mock_cls.return_value.with_structured_output.side_effect = _structured_output_side_effect(
        plan=planner_mod.Plan(steps=[
            PlanStep(description="查询 CPU 指标", evidence="metrics", tool_name="query_cpu_metrics")
        ])
    )
    mocker.patch(
        "app.agent.aiops.planner.get_mcp_client_with_retry",
        new=AsyncMock(return_value=_stub_mcp_client(mocker)),
    )
    rk = mocker.patch("app.agent.aiops.planner.guarded_retrieve_knowledge")
    rk.ainvoke = AsyncMock(return_value="经验文档内容")

    result = await planner_mod.planner(_state([]))

    assert isinstance(result["plan"], list)
    assert all(isinstance(s, PlanStep) for s in result["plan"])
    assert result["plan"][0].description == "查询 CPU 指标"
    assert result["plan"][0].evidence == "metrics"


async def test_planner_fallback_plan_is_structured(mocker):
    mocker.patch(
        "app.agent.aiops.planner.get_mcp_client_with_retry",
        new=AsyncMock(side_effect=Exception("mcp down")),
    )
    rk = mocker.patch("app.agent.aiops.planner.guarded_retrieve_knowledge")
    rk.ainvoke = AsyncMock(return_value="")

    result = await planner_mod.planner(_state([]))

    plan = result["plan"]
    assert len(plan) == 3
    assert all(isinstance(s, PlanStep) for s in plan)


# ---------------------------------------------------------------------------
# 5~9. executor
# ---------------------------------------------------------------------------

async def test_executor_direct_local_tool_success(mocker):
    stub = mocker.MagicMock()
    stub.ainvoke = AsyncMock(return_value="2026-10-12 10:00:00")
    mocker.patch.dict(executor_mod.DIRECT_TOOLS, {"get_current_time": stub})

    result = await executor_mod.executor(_state([
        PlanStep(description="获取当前时间", evidence="time", tool_name="get_current_time")
    ]))

    obs = result["past_steps"][0]
    assert obs.success is True
    assert obs.tool_names == ["get_current_time"]
    assert obs.tool_name == "get_current_time"
    assert result["plan"] == []
    stub.ainvoke.assert_awaited_once_with({})


async def test_executor_direct_local_tool_failure(mocker):
    stub = mocker.MagicMock()
    stub.ainvoke = AsyncMock(side_effect=Exception("boom"))
    mocker.patch.dict(executor_mod.DIRECT_TOOLS, {"retrieve_knowledge": stub})

    result = await executor_mod.executor(_state([
        PlanStep(description="检索知识", evidence="knowledge", tool_name="retrieve_knowledge")
    ]))

    obs = result["past_steps"][0]
    assert obs.success is False
    assert obs.result_summary.startswith("执行失败")
    assert result["plan"] == []


async def test_executor_llm_tool_path_records_observation(mocker):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_cpu_metrics", "args": {"service_name": "s"}, "id": "call_1", "type": "tool_call"}
    ])
    final = AIMessage(content="CPU 95%, 疑似死循环")
    _patch_executor_llm(mocker, first, final)

    result = await executor_mod.executor(_state([
        PlanStep(description="查询 CPU", evidence="metrics")
    ]))

    obs = result["past_steps"][0]
    assert isinstance(obs, Observation)
    assert obs.success is True
    assert obs.tool_names == ["query_cpu_metrics"]
    assert obs.result_summary == "CPU 95%, 疑似死循环"
    assert result["plan"] == []


async def test_executor_no_tool_call_uses_llm_content(mocker):
    first = AIMessage(content="无需工具，直接分析")
    _, _, tool_node_stub = _patch_executor_llm(mocker, first, AIMessage(content="unused"))

    result = await executor_mod.executor(_state([
        PlanStep(description="综合分析", evidence="synthesis")
    ]))

    obs = result["past_steps"][0]
    assert obs.success is True
    assert obs.tool_names == []
    tool_node_stub.ainvoke.assert_not_called()


async def test_executor_legacy_string_step_coerced(mocker):
    first = AIMessage(content="结果")
    _patch_executor_llm(mocker, first, AIMessage(content="unused"))

    result = await executor_mod.executor(_state(["旧步骤"]))

    obs = result["past_steps"][0]
    assert obs.evidence == "synthesis"
    assert obs.step_description == "旧步骤"


# ---------------------------------------------------------------------------
# 10~14. replanner
# ---------------------------------------------------------------------------

async def test_replanner_respond_when_evidence_covered(mocker):
    act = replanner_mod.Act(action="respond", reason="信息充足")
    response = replanner_mod.Response(response="# 诊断报告")
    _patch_replanner(mocker, act=act, response=response)

    state = _state(
        [PlanStep(description="综合生成报告", evidence="synthesis")],
        past_steps=[Observation(step_description="查 CPU", evidence="metrics", success=True, result_summary="CPU 95%")],
    )

    result = await replanner_mod.replanner(state)
    assert result.get("response") == "# 诊断报告"


async def test_replanner_replan_not_banned_after_5_steps(mocker):
    new_step = PlanStep(description="查日志", evidence="logs", tool_name="search_log")
    act = replanner_mod.Act(action="replan", new_steps=[new_step], missing_evidence="logs", reason="缺日志证据")
    _patch_replanner(mocker, act=act)

    obs = [Observation(step_description=f"s{i}", evidence="metrics", success=True, result_summary="ok") for i in range(6)]
    state = _state([PlanStep(description="剩余步骤", evidence="metrics")], past_steps=obs)

    result = await replanner_mod.replanner(state)
    assert result == {"plan": [new_step]}


async def test_replanner_replan_new_steps_longer_than_remaining_accepted(mocker):
    new_steps = [PlanStep(description=f"补证据{i}", evidence="logs") for i in range(4)]
    act = replanner_mod.Act(action="replan", new_steps=new_steps, missing_evidence="logs", reason="缺证据")
    _patch_replanner(mocker, act=act)

    state = _state(
        [PlanStep(description="r1", evidence="metrics"), PlanStep(description="r2", evidence="metrics")],
        past_steps=[Observation(step_description="s", evidence="time", success=True, result_summary="ok")],
    )

    result = await replanner_mod.replanner(state)
    assert len(result["plan"]) == 4


async def test_replanner_continue_returns_empty(mocker):
    act = replanner_mod.Act(action="continue", reason="继续")
    _patch_replanner(mocker, act=act)

    state = _state(
        [PlanStep(description="下一步", evidence="metrics")],
        past_steps=[Observation(step_description="s", evidence="time", success=True, result_summary="ok")],
    )

    result = await replanner_mod.replanner(state)
    assert result == {}


async def test_replanner_max_steps_safety_net_forces_response(mocker):
    response = replanner_mod.Response(response="# 兜底报告")
    mock_cls = mocker.patch("app.agent.aiops.replanner.ChatQwen")
    mock_cls.return_value.with_structured_output.side_effect = _structured_output_side_effect(response=response)
    get_mcp = mocker.patch(
        "app.agent.aiops.replanner.get_mcp_client_with_retry",
        new=AsyncMock(return_value=_stub_mcp_client(mocker)),
    )

    obs = [Observation(step_description=f"s{i}", evidence="metrics", success=True, result_summary="ok") for i in range(12)]
    state = _state([PlanStep(description="剩余", evidence="metrics")], past_steps=obs)

    result = await replanner_mod.replanner(state)
    assert result.get("response") == "# 兜底报告"
    get_mcp.assert_not_called()


# ---------------------------------------------------------------------------
# 15~16. SSE 格式化
# ---------------------------------------------------------------------------

def test_format_planner_event_flattens_to_list_of_str():
    event = AIOpsService._format_planner_event({
        "plan": [
            PlanStep(description="d1", evidence="metrics"),
            PlanStep(description="d2", evidence="logs"),
        ]
    })

    assert event["plan"] == ["d1", "d2"]
    assert all(isinstance(x, str) for x in event["plan"])
    assert len(event["plan_details"]) == 2
    assert event["plan_details"][0]["evidence"] == "metrics"


def test_format_executor_event_reads_observation():
    event = AIOpsService._format_executor_event({
        "plan": [PlanStep(description="d2", evidence="logs")],
        "past_steps": [Observation(step_description="d1", evidence="metrics", success=True, result_summary="ok")],
    })

    assert event["type"] == "step_complete"
    assert event["current_step"] == "d1"
    assert isinstance(event["current_step"], str)


# ---------------------------------------------------------------------------
# 17. 图级端到端（mock LLM，验证 checkpoint 序列化）
# ---------------------------------------------------------------------------

async def test_graph_end_to_end_with_mocked_llm(mocker):
    # planner
    mock_planner_cls = mocker.patch("app.agent.aiops.planner.ChatQwen")
    mock_planner_cls.return_value.with_structured_output.side_effect = _structured_output_side_effect(
        plan=planner_mod.Plan(steps=[
            PlanStep(description="查 CPU", evidence="metrics", tool_name="query_cpu_metrics")
        ])
    )
    mocker.patch(
        "app.agent.aiops.planner.get_mcp_client_with_retry",
        new=AsyncMock(return_value=_stub_mcp_client(mocker)),
    )
    rk = mocker.patch("app.agent.aiops.planner.guarded_retrieve_knowledge")
    rk.ainvoke = AsyncMock(return_value="")

    # executor（query_cpu_metrics 不在 DIRECT_TOOLS，走 LLM 路径）
    first = AIMessage(content="", tool_calls=[
        {"name": "query_cpu_metrics", "args": {"service_name": "s"}, "id": "call_1", "type": "tool_call"}
    ])
    final = AIMessage(content="CPU 95%")
    mock_exec_cls = mocker.patch("app.agent.aiops.executor.ChatQwen")
    llm_with_tools = mocker.MagicMock()
    llm_with_tools.ainvoke = AsyncMock(side_effect=[first, final])
    mock_exec_cls.return_value.bind_tools.return_value = llm_with_tools
    tool_node_stub = mocker.MagicMock()
    tool_node_stub.ainvoke = AsyncMock(return_value={
        "messages": [ToolMessage(content="cpu=95", tool_call_id="call_1", name="query_cpu_metrics")]
    })
    mocker.patch("app.agent.aiops.executor.ToolNode", return_value=tool_node_stub)
    mocker.patch(
        "app.agent.aiops.executor.get_mcp_client_with_retry",
        new=AsyncMock(return_value=_stub_mcp_client(mocker)),
    )

    # replanner（executor 消费完唯一步骤后 plan 为空，直接走 _generate_response）
    mock_repl_cls = mocker.patch("app.agent.aiops.replanner.ChatQwen")
    mock_repl_cls.return_value.with_structured_output.side_effect = _structured_output_side_effect(
        response=replanner_mod.Response(response="# 诊断报告")
    )
    mocker.patch(
        "app.agent.aiops.replanner.get_mcp_client_with_retry",
        new=AsyncMock(return_value=_stub_mcp_client(mocker)),
    )

    service = AIOpsService()
    initial = {"input": "诊断 CPU 告警", "plan": [], "past_steps": [], "response": ""}
    final_state = await service.graph.ainvoke(initial, {"configurable": {"thread_id": "e2e-test"}})

    assert final_state["response"] == "# 诊断报告"
    assert len(final_state["past_steps"]) >= 1
    obs = coerce_observation(final_state["past_steps"][0])
    assert obs.success is True
