"""意图识别：判断一句话该走「诊断」还是「问答」，供统一入口路由。

用 LLM 结构化输出分类（复用 replanner 的 Act 模式），分类失败 / LLM 不可用时
兜底 `chat`（问答）——问答更保守，不触发诊断里的 MCP 监控/日志工具调用。
"""
from textwrap import dedent
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from app.config import config
from app.services.token_usage import token_usage_callback

# 顶层意图：diagnose=故障诊断（走 AIOps Plan-Execute-Replan）；chat=知识问答（走 RAG）
IntentType = Literal["diagnose", "chat"]

# LLM 可能输出的近义词 -> 规范意图，避免 structured output 校验失败导致整段 fallback
_INTENT_ALIASES = {
    "诊断": "diagnose",
    "故障": "diagnose",
    "故障排查": "diagnose",
    "排查": "diagnose",
    "根因": "diagnose",
    "根因分析": "diagnose",
    "诊断分析": "diagnose",
    "诊断报告": "diagnose",
    "troubleshoot": "diagnose",
    "troubleshooting": "diagnose",
    "diagnosis": "diagnose",
    "问答": "chat",
    "闲聊": "chat",
    "知识问答": "chat",
    "其他": "chat",
    "other": "chat",
    "qa": "chat",
}


class IntentDecision(BaseModel):
    """意图分类结果。intent 为规范化后的顶层意图，reason 为一句话理由（供前端/日志）。"""

    intent: IntentType = Field(..., description="顶层意图：diagnose=诊断 / chat=问答")
    reason: str = Field("", description="分类理由，一句话")

    @field_validator("intent", mode="before")
    @classmethod
    def _normalize_intent(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            return _INTENT_ALIASES.get(v, v)
        return v


# 分类提示词：只判断「诊断 / 问答」两类
INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是一个意图分类器，负责判断用户这句话属于哪一类意图。

                只有两类意图：
                - diagnose（诊断）：用户描述了一个故障/异常，想要排查原因、定位根因或恢复服务。
                  特征词/场景：报错、告警、异常、故障、性能下降、CPU/内存/磁盘飙高、服务挂了/
                  不可用/超时/无响应/变慢、排查、根因、为什么出问题、怎么恢复、怎么解决线上问题。
                - chat（问答）：用户想了解某个概念、配置方法、最佳实践或通用知识，不是排查具体故障。
                  特征：是什么、为什么（知识层面）、怎么用、怎么配置、怎么调优、有什么最佳实践。

                判断原则：
                - 优先判断是不是「诊断」：只要提到了具体故障现象、报错、异常、性能问题，一律 diagnose。
                - 其余（概念/配置/知识/闲聊）归 chat。
                - 只输出 intent 与 reason 两个字段，不要输出 JSON 以外的内容。
            """).strip(),
        ),
        ("human", "{question}"),
    ]
)


async def classify_intent(text: str) -> IntentDecision:
    """对用户输入做意图分类；任何异常都兜底返回 chat（问答），绝不抛异常。"""
    try:
        llm = ChatQwen(
            model=config.intent_model or config.rag_model,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0,
            callbacks=[token_usage_callback],
        )
        chain = INTENT_PROMPT | llm.with_structured_output(IntentDecision)
        decision = await chain.ainvoke({"question": text})

        if isinstance(decision, IntentDecision):
            return decision
        if isinstance(decision, dict):
            return IntentDecision(**decision)
        logger.warning(f"意图分类返回异常类型 {type(decision)}，兜底 chat")
        return IntentDecision(intent="chat", reason="分类结果异常，默认问答")
    except Exception as e:  # noqa: BLE001 —— 分类失败一律兜底，不影响主流程
        logger.warning(f"意图分类失败，兜底 chat：{e}")
        return IntentDecision(intent="chat", reason="分类失败，默认问答")


def intent_name(intent: str) -> str:
    """把意图转成给前端/日志的可读名。"""
    return "诊断" if intent == "diagnose" else "问答"
