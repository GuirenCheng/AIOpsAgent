"""Agent 兜底护栏（Guardrails）中间件

针对"模型说错话 / 编造事实"的硬约束与软约束：

1. no_context_guard   —— 检索未命中或检索出错时，直接短路到固定话术（硬兜底），
                         绝不让模型在"无依据"状态下自由发挥。
2. tool_failure_guard —— 工具调用失败时，把错误改写成"禁止编造"指令再交给模型。
3. citation_check     —— 生成后校验：检索到了资料但回答未引用【参考资料】，打日志告警
                         （脱稿检测，供人工复核 / 后续升级为 LLM-as-judge）。

实现要点：
- before/after_model 钩子写成同步函数即可（RunnableCallable 在异步执行下也会正确调用）。
- wrap_tool_call 必须写成 async：异步 MCP 工具若走同步包装器会破坏其执行路径。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import after_model, before_model, wrap_tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

# —— 固定兜底话术 ——
NO_CONTEXT_ANSWER = (
    "抱歉，知识库中未检索到与该问题相关的可靠资料，我不能给出有依据的回答，"
    "以免编造信息。建议您换个问法，或提供更多上下文。"
)
RETRIEVAL_ERROR_ANSWER = (
    "知识检索服务暂时不可用（向量库查询异常），我无法获取参考资料，请稍后重试。"
)

CITATION_MARKER = "【参考资料"
_RETRIEVE_TOOL = "retrieve_knowledge"


def _msg_text(content: Any) -> str:
    """把消息 content（可能是 str 或内容块列表）归一化为纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _retrieval_failure_reason(msg: Any) -> str | None:
    """判断最后一条消息是否代表"知识检索未提供可靠资料"。

    Returns:
        "error" 表示检索服务出错；"empty" 表示未命中；None 表示无需兜底。
    """
    if not isinstance(msg, ToolMessage):
        return None

    name = getattr(msg, "name", None)
    text = _msg_text(getattr(msg, "content", ""))

    # 检索出错：以唯一错误串兜底识别（即使 name 缺失也能命中）
    if "检索知识时发生错误" in text:
        return "error"

    if name != _RETRIEVE_TOOL:
        return None

    # 检索成功时 content 必然包含格式化参考资料标记；否则视为未命中
    if CITATION_MARKER not in text:
        return "empty"
    return None


@before_model(can_jump_to=["end"])
def no_context_guard(state: dict, runtime: Any) -> dict[str, Any] | None:
    """检索未命中/出错时，硬短路到固定话术，跳过模型生成（杜绝编造）。"""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None

    reason = _retrieval_failure_reason(last)
    if reason is None:
        return None

    content = RETRIEVAL_ERROR_ANSWER if reason == "error" else NO_CONTEXT_ANSWER
    logger.warning(f"[护栏] 知识检索{'出错' if reason == 'error' else '未命中'}，短路返回兜底话术")
    return {
        "jump_to": "end",
        "messages": [AIMessage(content=content)],
    }


def _turn_had_grounding(messages: list) -> bool:
    """当前轮（回到最近一条 HumanMessage 为止）是否检索到了有效参考资料。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None)
            text = _msg_text(getattr(msg, "content", ""))
            if name == _RETRIEVE_TOOL and CITATION_MARKER in text:
                return True
    return False


@after_model
def citation_check(state: dict, runtime: Any) -> None:
    """生成后脱稿检测：检索到了资料但最终回答未引用，打日志告警（不改写输出）。

    说明：流式场景下 token 已逐条发出，无法安全改写已输出的内容，故此处只做
    检测与告警，供人工复核；如需硬改写，可在此返回带免责声明的消息更新。
    """
    messages = state.get("messages") or []
    if not messages:
        return None

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return None
    if getattr(last, "tool_calls", None):
        return None  # 还在调用工具，非最终回答

    answer = _msg_text(getattr(last, "content", ""))
    if _turn_had_grounding(messages) and CITATION_MARKER not in answer:
        logger.warning("[护栏] 模型回答未引用任何【参考资料】（脱稿），建议人工复核该条回复")
    return None


def _tool_name(request: Any) -> str:
    """从 ToolCallRequest 中取出工具名。"""
    tool = getattr(request, "tool", None)
    if tool is not None and getattr(tool, "name", None):
        return tool.name
    call = getattr(request, "tool_call", {}) or {}
    return call.get("name", "unknown")


def _format_tool_failure(name: str, original: str) -> str:
    return (
        f"⚠️ 工具「{name}」调用失败。严禁编造或猜测该工具的返回结果；"
        f"如无法完成用户请求，请如实告知失败原因并建议稍后重试。\n\n"
        f"原始错误：\n{original}"
    )


@wrap_tool_call
async def tool_failure_guard(request: Any, handler: Any) -> Any:
    """工具失败兜底：把异常/错误改写成"禁止编造"指令，再交给模型。

    MCP 工具 isError 会抛 ToolException（langchain_mcp_adapters 行为），
    需要在这里捕获并转成带护栏提示的 ToolMessage(status="error")。
    """
    try:
        result = await handler(request)
    except Exception as exc:  # noqa: BLE001 —— 工具层的任何失败都转为兜底消息
        name = _tool_name(request)
        logger.warning(f"[护栏] 工具「{name}」调用异常，改写为兜底消息：{exc}")
        return ToolMessage(
            content=_format_tool_failure(name, str(exc)),
            tool_call_id=request.tool_call.get("id"),
            name=name,
            status="error",
        )

    if isinstance(result, ToolMessage) and getattr(result, "status", None) == "error":
        name = _tool_name(request)
        logger.warning(f"[护栏] 工具「{name}」返回错误，追加防编造提示")
        return ToolMessage(
            content=_format_tool_failure(name, _msg_text(result.content)),
            tool_call_id=getattr(result, "tool_call_id", None),
            name=getattr(result, "name", None),
            status="error",
            artifact=getattr(result, "artifact", None),
        )

    return result
