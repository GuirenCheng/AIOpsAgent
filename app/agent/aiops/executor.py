"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""
import json
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.agent.aiops.state import PlanExecuteState, PlanStep, Observation, coerce_step
from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.services.token_usage import token_usage_callback
from app.tools.tool_wrapper import guarded_get_current_time, guarded_retrieve_knowledge


# 可确定性直接调用的本地工具（无需 LLM 选择参数）
DIRECT_TOOLS = {
    "get_current_time": guarded_get_current_time,
    "retrieve_knowledge": guarded_retrieve_knowledge,
}


def _is_error_result(result) -> bool:
    '''判断工具执行结果是否为错误'''
    if isinstance(result, dict):
        return bool(result.get("error")) or bool(result.get("isError"))
    if isinstance(result, str):
        return result.startswith(("执行失败", "获取时间失败", "检索知识时发生错误"))
    return False


def _build_observation(step: PlanStep, tool_name, tool_names, success, result) -> Observation:
    '''统一构造观察记录（结果截断到 500 字符）'''
    if isinstance(result, str):
        summary = result
    else:
        summary = json.dumps(result, ensure_ascii=False, default=str)
    return Observation(
        step_description=step.description,
        evidence=step.evidence,
        tool_name=tool_name,
        tool_names=tool_names,
        success=success,
        result_summary=summary[:500],
    )


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    '''
    执行节点：执行计划中的下一个步骤

    若步骤已绑定确定性本地工具（get_current_time / retrieve_knowledge），直接调用；
    否则使用 LLM + ToolNode 自适应选择工具。
    '''
    logger.info("=== Executor:执行步骤 ===")

    plan = state.get('plan', [])

    # 如果计划为空，不执行
    if not plan:
        logger.info('计划为空，跳过执行')
        return {}

    # 取出第一个步骤（兼容旧 checkpoint 的 str / serde 降级的 dict）
    step = coerce_step(plan[0])
    logger.info(f"当前任务:{step.description}")

    try:
        # 路径1：确定性本地工具，直接调用
        if step.tool_name in DIRECT_TOOLS:
            tool = DIRECT_TOOLS[step.tool_name]
            logger.info(f"直接调用确定性工具：{step.tool_name}")
            try:
                # retrieve_knowledge 使用 content_and_artifact，ainvoke 只返回 str
                result = await tool.ainvoke(step.tool_args or {})
                success = not _is_error_result(result)
                obs = _build_observation(
                    step, step.tool_name, [step.tool_name], success, result
                )
            except Exception as e:
                logger.error(f"直接调用工具失败：{e}")
                obs = _build_observation(
                    step, step.tool_name, [step.tool_name], False, f"执行失败：{str(e)}"
                )
            return {'plan': plan[1:], 'past_steps': [obs]}

        # 路径2：LLM 自适应选择工具
        # 获取本地工具
        local_tools = [
            guarded_get_current_time,
            guarded_retrieve_knowledge,
        ]

        # 获取MCP工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"可用工具数量：本地{len(local_tools)}+MCP {len(mcp_tools)}")

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 创建LLM（绑定工具）
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0,
            callbacks=[token_usage_callback],
        )

        llm_with_tools = llm.bind_tools(all_tools)

        # 创建工具节点（自动执行工具调用）
        tool_node = ToolNode(all_tools)

        # 任务文本：优先用步骤描述，附带可选的工具建议
        task_text = step.description
        if step.tool_name:
            task_text += f"\n（建议工具：{step.tool_name}，参数：{json.dumps(step.tool_args or {}, ensure_ascii=False)}）"

        # 构建消息（只包含当前步骤，避免原始任务干扰）
        messages = [
            SystemMessage(content='''
            你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务'''),
            HumanMessage(content=f"请执行以下任务:{task_text}")
        ]

        # 第一步：LLM决定是否调用
        llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM响应类型：{type(llm_response)}")

        tool_names: list = []
        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
            for tc in llm_response.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                if name:
                    tool_names.append(name)
            logger.info(f"检测到{len(tool_names)}个工具调用")

            # 使用ToolNode自动执行工具
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})

            # 检查工具调用是否报错
            tool_error = False
            for tm in tool_messages.get('messages', []):
                status = getattr(tm, 'status', None)
                content = getattr(tm, 'content', '')
                if status == 'error' or (isinstance(content, str) and _is_error_result(content)):
                    tool_error = True
                    break

            # 第三步：将工具返回给LLM生成最终答案
            messages.extend(tool_messages['messages'])
            final_response = await llm_with_tools.ainvoke(messages)
            result = final_response.content if hasattr(final_response, 'content') else str(final_response)
            success = not tool_error
        else:
            # 没有工具调用，直接使用LLM的输出
            logger.info("LLM未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            success = bool(result and result.strip())

        logger.info(f"步骤执行完成，结果长度：{len(result) if isinstance(result, str) else 0}")

        obs = _build_observation(
            step,
            tool_names[0] if tool_names else step.tool_name,
            tool_names,
            success,
            result,
        )

        # 返回更新，移除已执行的步骤，添加执行历史
        return {
            'plan': plan[1:],  # 移除第一个步骤
            'past_steps': [obs],  # 使用operator.add追加
        }
    except Exception as e:
        logger.error(f"执行步骤失败：{e}", exc_info=True)
        obs = _build_observation(step, step.tool_name, [], False, f"执行失败：{str(e)}")
        return {
            'plan': plan[1:],
            'past_steps': [obs],
        }
