"""诊断报告「根因结论」硬校验测试。"""
import importlib
from unittest.mock import AsyncMock

from langchain_core.runnables import RunnableLambda

from app.agent.aiops.state import PlanStep, Observation
from app.agent.aiops.verifier import (
    ClaimVerdict,
    VerificationResult,
    build_verification_section,
    verify_report,
)

replanner_mod = importlib.import_module("app.agent.aiops.replanner")


# ---------------------------------------------------------------------------
# build_verification_section
# ---------------------------------------------------------------------------

def test_build_section_empty_returns_empty():
    assert build_verification_section(VerificationResult(verdicts=[])) == ""


def test_build_section_marks_ungrounded():
    result = VerificationResult(verdicts=[
        ClaimVerdict(claim="磁盘即将写满", grounded=False),
        ClaimVerdict(claim="CPU 飙高", grounded=True, evidence_ref="cpu=95"),
    ])
    section = build_verification_section(result)
    assert "结论可信度校验" in section
    assert "⚠️ 无据可查：磁盘即将写满" in section
    assert "✅ 有据可查：CPU 飙高（依据：cpu=95）" in section
    assert "1 条无据可查" in section


def test_build_section_all_grounded_has_no_warning():
    result = VerificationResult(verdicts=[
        ClaimVerdict(claim="CPU 飙高", grounded=True, evidence_ref="cpu=95"),
    ])
    section = build_verification_section(result)
    assert "⚠️ 无据可查：" not in section  # 无「无据」条目标记（汇总行的「0 条无据可查」不算）
    assert "0 条无据可查" in section


# ---------------------------------------------------------------------------
# verify_report：成功与 fail-open
# ---------------------------------------------------------------------------

async def test_verify_report_success():
    async def _chain(_input):
        return VerificationResult(verdicts=[
            ClaimVerdict(claim="CPU 飙高", grounded=True, evidence_ref="cpu=95"),
            ClaimVerdict(claim="磁盘写满", grounded=False),
        ])

    class FakeLLM:
        def with_structured_output(self, schema):
            assert schema is VerificationResult
            return RunnableLambda(_chain)

    result = await verify_report("# 报告", "cpu=95", FakeLLM())
    assert len(result.verdicts) == 2
    assert result.verdicts[1].grounded is False


async def test_verify_report_fail_open():
    async def _boom(_input):
        raise RuntimeError("llm boom")

    class FakeLLM:
        def with_structured_output(self, schema):
            return RunnableLambda(_boom)

    result = await verify_report("# 报告", "证据", FakeLLM())
    assert result.verdicts == []


# ---------------------------------------------------------------------------
# replanner 集成：报告末尾追加校验小节
# ---------------------------------------------------------------------------

def _stub_mcp_client(mocker, tools=None):
    client = mocker.MagicMock()
    client.get_tools = AsyncMock(return_value=tools or [])
    return client


async def test_replanner_appends_verification_section(mocker):
    act = replanner_mod.Act(action="respond", reason="信息充足")
    response = replanner_mod.Response(response="# 诊断报告\n\n根因结论：磁盘即将写满")
    verification = VerificationResult(verdicts=[
        ClaimVerdict(claim="磁盘即将写满", grounded=False),
    ])

    def _factory(schema):
        async def _fn(_input):
            if schema is replanner_mod.Act:
                return act
            if schema is replanner_mod.Response:
                return response
            if schema is VerificationResult:
                return verification
            raise AssertionError(f"unexpected schema {schema}")

        return RunnableLambda(_fn)

    mock_cls = mocker.patch("app.agent.aiops.replanner.ChatQwen")
    mock_cls.return_value.with_structured_output.side_effect = _factory
    mocker.patch(
        "app.agent.aiops.replanner.get_mcp_client_with_retry",
        new=AsyncMock(return_value=_stub_mcp_client(mocker)),
    )

    state = {
        "input": "诊断告警",
        "plan": [PlanStep(description="综合分析", evidence="synthesis")],
        "past_steps": [Observation(step_description="查 CPU", evidence="metrics", success=True, result_summary="cpu=95")],
        "response": "",
    }

    result = await replanner_mod.replanner(state)
    assert "结论可信度校验" in result["response"]
    assert "⚠️ 无据可查：磁盘即将写满" in result["response"]
