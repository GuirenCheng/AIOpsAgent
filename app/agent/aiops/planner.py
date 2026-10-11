"""
Planner 节点：制定执行计划
基于 LangGraph 官方教程实现
"""
from textwrap import dedent
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field
from langchain_qwq import ChatQwen
from app.agent.aiops.utils import format_tools_description
from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.tools.knowledge_tool import retrieve_knowledge
from app.agent.aiops.state import PlanExecuteState, PlanStep, coerce_step
from app.tools.time_tool import get_current_time


class Plan(BaseModel):
    '''计划的输出格式'''
    steps: List[PlanStep] = Field(
        description='完成任务所需的不同步骤，这些步骤应该按顺序执行，每一步都建立在前一步的基础上。'
    )


# Planner 提示词
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            dedent('''
                作为一个专家级别的规划者，你需要将复杂的任务分解为可执行的步骤。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定计划，实际的工具调用由 Executor 负责执行。

                {experience_context}

                对于给定的任务，请创建一个简单的、逐步的计划来完成它。计划的每一步都是如下结构的对象：
                - description（字符串）：该步骤的具体、可操作描述，一个逻辑动作
                - evidence（字符串）：该步骤要收集的证据类型，必须是以下之一：
                  * "logs"：CLS/search_log 等日志证据
                  * "metrics"：CPU、内存等监控指标证据
                  * "knowledge"：内部知识库经验
                  * "time"：当前时间/时间戳
                  * "synthesis"：综合分析/生成报告，不产生新证据
                - tool_name（字符串，可选）：如果该步骤明确需要某个工具，填写工具名；否则留空，由 Executor 根据描述选择
                - tool_args（对象，可选）：tool_name 对应的调用参数；不确定时留空

                制定计划的要求：
                - 将任务分解为逻辑上独立的步骤，步骤之间应该有清晰的依赖关系
                - 优先收集证据（logs/metrics/knowledge/time），最后以 synthesis 步骤收尾
                - **如果有相关经验文档，请参考其中的方法和步骤制定计划**

                示例（假设有对应工具）：
                [
                  {{"description": "查询 data-sync-service 最近15分钟的 CPU 指标", "evidence": "metrics", "tool_name": "query_cpu_metrics", "tool_args": {{"service_name": "data-sync-service"}}}},
                  {{"description": "检索服务日志中的异常与报错", "evidence": "logs", "tool_name": null, "tool_args": null}},
                  {{"description": "综合分析上述证据并生成诊断报告", "evidence": "synthesis", "tool_name": null, "tool_args": null}}
                ]
            ''').strip(),
        ),
        ('placeholder', '{messages}'),
    ]
)


async def planner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    规划节点：根据用户输入生成执行计划

    流程：
    1. 先查询内部文档，获取相关经验和最佳实践
    2. 基于经验文档和可用工具制定执行计划
    """
    logger.info("=== Planner：指定执行计划 ===")

    input_text = state.get('input', '')
    logger.info(f'用户输入: {input_text}')

    try:
        # 步骤1，查询内部文档获取相关经验
        logger.info("查询内部文档，寻找相关经验...")
        experience_docs = ""
        try:
            # retrieve_knowledge 使用 response_format="content_and_artifact"
            # ainvoke() 只返回 content（字符串），不是元组
            context_str = await retrieve_knowledge.ainvoke({'query': input_text})
            if context_str and context_str.strip():
                experience_docs = context_str
                logger.info(f"找到相关经验文档，长度：{len(experience_docs)}")
            else:
                logger.info(f"未找到相关经验文档")
        except Exception as e:
            logger.warning(f'查询内部文档失败：{e}')

        # 步骤2: 获取可用工具列表
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
        logger.info(f"可用工具数量：本地{len(local_tools)}+MCP {len(mcp_tools)}")

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)

        # 步骤3：格式化经验文档上下文
        experience_context = ""
        if experience_docs:
            experience_context = dedent(f'''
                ### 相关经验文档

                以下是从知识库中检索到的相关经验和最佳实践，请参考这些经验指定执行计划：

                {experience_docs}

                ---
            ''').strip()

        # 步骤4：创建LLM并生成计划
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )

        planner_chain = planner_prompt | llm.with_structured_output(Plan)

        # 调用LLM生成计划
        plan_result = await planner_chain.ainvoke({
            'messages': [("user", input_text)],
            'tools_description': tools_description,
            'experience_context': experience_context,
        })

        # 提取步骤列表
        if isinstance(plan_result, Plan):
            plan_steps = plan_result.steps
        else:
            # 如果返回的是字典，提取steps字段
            plan_steps = plan_result.get("steps", [])

        # 归一化为 PlanStep（兼容 structured output 返回 dict 的情况）
        plan_steps = [coerce_step(s) for s in plan_steps]

        logger.info(f"计划已生成，共{len(plan_steps)}个步骤")
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step.description} [evidence={step.evidence}]")

        return {'plan': plan_steps}

    except Exception as e:
        logger.info(f"生成计划失败：{e}", exc_info=True)
        # 返回一个默认计划
        return {
            'plan': [
                PlanStep(description='收集相关信息（知识库与时间范围）', evidence='knowledge'),
                PlanStep(description='分析数据', evidence='synthesis'),
                PlanStep(description='生成报告', evidence='synthesis'),
            ]
        }
