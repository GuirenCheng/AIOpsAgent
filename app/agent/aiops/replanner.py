"""
Replanner 节点：重新规划或生成最终响应
基于 LangGraph 官方教程实现
"""
from textwrap import dedent
from typing import Dict, Any, List, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.agent.aiops.state import (
    PlanExecuteState,
    PlanStep,
    Observation,
    coerce_step,
    coerce_observation,
)
from app.agent.aiops.utils import format_tools_description
from app.agent.aiops.verifier import verify_report, build_verification_section
from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.services.token_usage import token_usage_callback
from app.services.langfuse_trace import get_llm_callbacks
from app.tools.knowledge_tool import retrieve_knowledge
from app.tools.time_tool import get_current_time


# 绝对步数上限：仅作为防死循环的兜底，不再驱动 replan 决策
MAX_STEPS = 12


class Response(BaseModel):
    '''最终响应的格式'''
    response: str = Field(description='对用户的最终响应')


class Act(BaseModel):
    '''重新规划的输出格式'''
    action: Literal['continue', 'replan', 'respond'] = Field(
        description="下一步行动：continue=继续执行剩余计划；replan=调整计划（提供新步骤）；respond=信息充足，生成最终响应"
    )
    # action 为 'replan' 时，新的步骤列表(会替换当前剩余计划)
    new_steps: List[PlanStep] = Field(
        default_factory=list,
        description="新的步骤列表（仅当 action='replan' 时提供，会替换剩余计划）"
    )
    missing_evidence: Optional[str] = Field(
        None, description="当 action='replan' 时，缺失/失败的证据类型（逗号分隔）"
    )
    reason: str = Field('', description='决策理由，一句话')


# Replan提示词
replanner_prompts = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            dedent("""
                作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定或调整计划，实际的工具调用由 Executor 负责执行。

                你将收到以下输入：
                1. 原始任务
                2. 剩余计划（每个步骤带有 evidence 标签）
                3. 已收集的观察（每条带有 [成功/失败] 和 evidence 标签）
                4. 一份证据覆盖摘要（计划证据类型 / 已成功 / 已失败 / 仍缺失）

                证据类型含义：logs=日志、metrics=监控指标、knowledge=知识库、time=时间、synthesis=综合分析（不产生新证据）。

                你有三个选择，请根据【证据覆盖度】而非步数来决策：

                **1. respond** —— 所需证据已收集齐全，或缺失的仅剩 synthesis（综合分析）
                   - 所有任务必需的关键证据（logs/metrics/knowledge/time）都已成功获得；或虽尝试失败但已无替代途径

                **2. replan** —— 某个计划内的证据类型缺失或失败
                   - 在 new_steps 中给出针对缺失证据的新步骤，并填写 missing_evidence（缺失的证据类型）与 reason
                   - 没有"新步骤数量必须 <= 剩余步骤数"的限制，也没有"执行 N 步后禁止 replan"的限制

                **3. continue** —— 剩余步骤仍能收集到需要的证据类型，且没有已计划的证据失败
                   - 直接继续执行，不需要调整

                决策原则：
                - 以"是否已收集到足够的证据得出结论"为唯一标准，不要为了填满步骤而继续
                - 证据失败时优先判断能否换一种方式补齐；确实补不齐就 respond，并诚实说明
                - 不要陷入无限重试：同样的证据已经尝试过且失败，就不要再生成相同的新步骤
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

# 最终响应生成提示词
response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            dedent("""
                根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

                响应要求：
                - 清晰、结构化
                - 基于实际数据，不要编造
                - 如果某些步骤失败，要诚实说明
                - 使用 Markdown 格式
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


def _format_step(step) -> str:
    '''格式化剩余计划中的单个步骤（已归一化为 PlanStep）'''
    s = coerce_step(step)
    tool = f" (tool={s.tool_name})" if s.tool_name else ""
    return f"- [evidence:{s.evidence}] {s.description}{tool}"


def _format_observation(obs) -> str:
    '''格式化单条观察（已归一化为 Observation）'''
    o = coerce_observation(obs)
    status = "成功" if o.success else "失败"
    return f"- [{status}][evidence:{o.evidence}] {o.step_description}: {o.result_summary[:300]}"


def _evidence_digest(plan, past_steps):
    '''计算证据覆盖摘要：计划证据 / 已成功 / 已失败 / 仍缺失（输入须为已归一化的列表）'''
    planned = {s.evidence for s in plan}
    ok = {o.evidence for o in past_steps if o.success}
    failed = {o.evidence for o in past_steps if not o.success}
    missing = planned - ok
    return planned, ok, failed, missing


async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    重新规划节点：决定是继续、调整计划还是生成最终响应

    三种决策（基于证据覆盖度，而非步数）：
    1. continue - 继续执行当前计划
    2. replan - 调整计划（替换剩余步骤）
    3. respond - 生成最终响应
    """
    logger.info(" === Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = [coerce_step(s) for s in state.get("plan", [])]
    past_steps = [coerce_observation(o) for o in state.get("past_steps", [])]

    logger.info(f"剩余计划步骤：{len(plan)}")
    logger.info(f"已执行步骤：{len(past_steps)}")

    # 兜底：步数达到硬上限，直接生成响应（防死循环）
    if len(past_steps) >= MAX_STEPS:
        logger.warning(f"已执行{len(past_steps)}个步骤，达到硬上限{MAX_STEPS}，强制生成最终响应")
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0,
            callbacks=get_llm_callbacks(),
        )
        return await _generate_response(state, llm)

    # 获取可用工具列表
    try:
        # 获取本地工具
        local_tools = [
            get_current_time,
            retrieve_knowledge,
        ]

        # 获取MCP工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量：本地{len(local_tools)}+MCP{len(mcp_tools)}")

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)
    except Exception as e:
        logger.warning(f"获取工具列表失败：{e}")
        tools_description = '无法获取工具列表'

    # 创建LLM
    llm = ChatQwen(
        model=config.rag_model,
        api_key=config.dashscope_api_key,
        base_url=config.dashscope_api_base,
        temperature=0,
        callbacks=get_llm_callbacks(),
    )

    # 格式化已执行观察 + 证据覆盖摘要
    obs_text = "\n".join(_format_observation(o) for o in past_steps) or "（无）"
    planned, ok, failed, missing = _evidence_digest(plan, past_steps)
    digest = (
        f"计划证据类型: {sorted(planned) or '无'}\n"
        f"已成功收集: {sorted(ok) or '无'}\n"
        f"已失败: {sorted(failed) or '无'}\n"
        f"仍缺失: {sorted(missing) or '无'}"
    )

    # 如果还有剩余计划，进行决策
    if plan:
        logger.info("还有剩余计划，评估下一步行动")

        replanner_chain = replanner_prompts | llm.with_structured_output(Act)

        try:
            remaining_text = "\n".join(_format_step(s) for s in plan)
            messages = [
                ("user", f"原始任务:{input_text}"),
                ("user", f"剩余计划:\n{remaining_text}"),
                ("user", f"已收集的观察:\n{obs_text}"),
                ("user", f"证据覆盖摘要:\n{digest}"),
            ]

            act = await replanner_chain.ainvoke({
                'messages': messages,
                'tools_description': tools_description,
            })

            # 处理返回结果
            if isinstance(act, Act):
                action = act.action
                new_steps = act.new_steps
            else:
                # 如果返回的是字典
                action = act.get("action", "continue")  # type: ignore
                new_steps = act.get("new_steps", [])  # type: ignore

            logger.info(f"Replanner 决策：{action}")

            if action == 'respond':
                logger.info("决定生成最终响应")
                return await _generate_response(state, llm)

            elif action == 'replan':
                new_steps = [coerce_step(s) for s in new_steps]
                if new_steps:
                    logger.info(f"决定调整计划，新步骤数量：{len(new_steps)}")
                    return {"plan": new_steps}
                else:
                    logger.warning("replan但未提供新步骤，继续执行原计划")
                    return {}

            else:  # action == "continue"
                logger.info('决定继续执行当前计划')
                return {}  # 不修改状态，继续执行

        except Exception as e:
            logger.info(f"重新规划失败:{e},继续执行剩余计划")
            return {}

    else:
        # 没有剩余计划，生成最终响应
        logger.info("计划已执行完毕，生成最终响应")
        return await _generate_response(state, llm)


async def _generate_response(state: PlanExecuteState, llm: ChatQwen) -> Dict[str, Any]:
    '''生成最终响应'''
    logger.info("生成最终响应...")

    input_text = state.get("input", "")
    past_steps = [coerce_observation(o) for o in state.get("past_steps", [])]

    # 格式化执行历史
    execution_history = "\n\n".join(
        f"### 步骤：{o.step_description}\n**证据类型：** {o.evidence}\n**结果：** {'成功' if o.success else '失败'}\n{o.result_summary}"
        for o in past_steps
    ) or "（无）"

    response_gen = response_prompt | llm.with_structured_output(Response)

    try:
        messages = [
            ("user", f"原始任务：{input_text}"),
            ("user", f"执行历史：\n{execution_history}"),
            ("user", f"请基于以上消息生成全面的最终响应"),
        ]

        response_obj = await response_gen.ainvoke({"messages": messages})

        # 处理返回结果
        if isinstance(response_obj, Response):
            final_response = response_obj.response
        else:
            # 如果返回的是字典
            final_response = response_obj.get("response", "")  # type: ignore

        logger.info(f"最终响应生成完成，长度:{len(final_response)}")

        # 根因结论硬校验：把报告结论逐条跟执行证据核对，无据的自动标「无据可查」。
        # verify_report 自身 fail-open，异常时返回空 verdicts，不会破坏报告。
        if config.aiops_report_verification_enabled:
            verification = await verify_report(final_response, execution_history, llm)
            section = build_verification_section(verification)
            if section:
                final_response = final_response.rstrip() + "\n\n" + section

        return {"response": final_response}

    except Exception as e:
        logger.error(f"生成响应失败：{e}")
        # 生成简单的后备响应
        fallback_response = f"""# 任务执行结果

## 原始任务
{input_text}

## 执行的步骤
{_format_simple_steps(state.get("past_steps", []))}

## 说明
由于系统异常，无法生成完整响应。以上是已收集的信息。
"""
        return {"response": fallback_response}


def _format_simple_steps(past_steps: list) -> str:
    """格式化步骤列表(简单版)"""
    if not past_steps:
        return "无"

    formatted = []
    for i, raw in enumerate(past_steps, 1):
        o = coerce_observation(raw)
        preview = o.result_summary[:200] + ("..." if len(o.result_summary) > 200 else "")
        status = "成功" if o.success else "失败"
        formatted.append(f"{i}. **{o.step_description}**（{status}）\n  {preview}\n")

    return "\n".join(formatted)
