"""知识冲突检测 + 权威/最新自动仲裁。

检索命中多条不同来源资料时，先让 LLM 判断它们之间是否存在针对同一事实的
互相矛盾（语义判断），若冲突则按「权威度降序 → 更新时间降序 → 原检索序升序」
确定性仲裁，只保留赢家，避免把互相矛盾的内容同时喂给模型。

权威度/更新时间来自文档元数据 `_authority` / `_updated_at`，由
`vector_index_service.index_single_file` 在索引时写入（见 config.rag_authority_rules）。

注意：本模块的 detect_conflict 是**同步**函数——调用点 retrieve_knowledge 是 sync
@tool，经 tool_wrapper.guard_local_tool → execute_guarded → tool.ainvoke 在 executor
线程里执行，同步阻塞 LLM 调用安全，不会阻塞事件循环。
"""
from datetime import datetime
from textwrap import dedent
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.services.token_usage import token_usage_callback


class ConflictDecision(BaseModel):
    """冲突检测结果。has_conflict 为是否存在矛盾；conflicting_indices 为参与冲突的文档下标（0-based）。"""

    has_conflict: bool = Field(..., description="检索资料之间是否存在针对同一事实的互相矛盾")
    conflicting_indices: List[int] = Field(
        default_factory=list,
        description="参与冲突的文档下标（0-based），无冲突时为空列表",
    )
    reason: str = Field("", description="判断理由，一句话")


CONFLICT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是一个知识冲突检测器。下面给出若干条检索到的知识库资料，每条带编号、来源、
                权威度、更新时间与内容摘要。

                请判断这些资料之间是否存在「针对同一事实、互相矛盾」的说法（例如同一参数推荐了
                相反的值、同一操作给了相反的顺序、同一结论一个肯定一个否定）。
                注意：
                - 只是从不同角度补充、粒度不同、或谈论不同主题的，不算冲突。
                - 只有确实在同一事实点上互相矛盾，才判定为冲突。
                - 若冲突，把互相矛盾的资料编号全部放进 conflicting_indices；不冲突则为空。
                - 只输出 has_conflict / conflicting_indices / reason 三个字段，不要输出 JSON 以外的内容。
            """).strip(),
        ),
        ("human", "{candidates}"),
    ]
)


def _render_candidates(docs: List[Document]) -> str:
    """把候选文档渲染成给 LLM 看的紧凑文本（内容截断，控制 prompt 长度）。"""
    lines = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        source = meta.get("_file_name") or meta.get("_source") or "未知来源"
        authority = meta.get("_authority", 1)
        updated_at = meta.get("_updated_at", "")
        content = (doc.page_content or "").strip()
        snippet = content[:200] + ("…" if len(content) > 200 else "")
        lines.append(
            f"【{i}】来源:{source} | 权威度:{authority} | 更新时间:{updated_at}\n内容:{snippet}"
        )
    return "\n\n".join(lines)


def detect_conflict(docs: List[Document]) -> ConflictDecision:
    """LLM 语义判断候选资料是否存在矛盾；任何异常兜底返回「无冲突」（不仲裁，维持现状）。"""
    try:
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0,
            callbacks=[token_usage_callback],
        )
        chain = CONFLICT_PROMPT | llm.with_structured_output(ConflictDecision)
        decision = chain.invoke({"candidates": _render_candidates(docs)})

        if isinstance(decision, ConflictDecision):
            return decision
        if isinstance(decision, dict):
            return ConflictDecision(**decision)
        logger.warning(f"冲突检测返回异常类型 {type(decision)}，兜底不仲裁")
        return ConflictDecision(has_conflict=False, reason="检测结果异常，默认不仲裁")
    except Exception as e:  # noqa: BLE001 —— 检测失败一律兜底，不影响主流程
        logger.warning(f"知识冲突检测失败，兜底不仲裁：{e}")
        return ConflictDecision(has_conflict=False, reason="检测失败，默认不仲裁")


def _parse_authority_rules() -> List[tuple]:
    """解析 config.rag_authority_rules（`前缀:权重` 逗号分隔），按前缀长度降序（最具体优先）。"""
    rules = []
    for item in config.rag_authority_rules.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        prefix, _, weight_str = item.rpartition(":")
        prefix = prefix.strip()
        try:
            weight = int(weight_str.strip())
        except ValueError:
            logger.warning(f"忽略非法权威度规则：{item!r}")
            continue
        if prefix:
            rules.append((prefix, weight))
    rules.sort(key=lambda r: -len(r[0]))
    return rules


def authority_from_path(path: str) -> int:
    """根据路径前缀规则计算权威度；无命中规则时默认 1（同权）。"""
    rules = _parse_authority_rules()
    if not rules:
        return 1
    normalized = path.replace("\\", "/")
    for prefix, weight in rules:
        if prefix in normalized:
            return weight
    return 1


def _parse_updated_at(value) -> float:
    """把 ISO 时间字符串解析为 epoch 秒（用于排序）；缺失/非法返回 0（视为最旧）。"""
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _doc_priority(doc: Document, index: int) -> tuple:
    """仲裁排序键：权威度降序、更新时间降序、原检索序升序。"""
    meta = doc.metadata
    authority = meta.get("_authority", 1)
    if not isinstance(authority, (int, float)):
        authority = 1
    updated_at = _parse_updated_at(meta.get("_updated_at", ""))
    return (-authority, -updated_at, index)


def resolve_conflict(docs: List[Document], decision: ConflictDecision) -> List[Document]:
    """按权威/最新仲裁：冲突集合内保留唯一赢家，丢弃其余；非冲突文档全部保留。"""
    if not decision.has_conflict or not decision.conflicting_indices:
        return list(docs)

    conflict_set = {i for i in decision.conflicting_indices if 0 <= i < len(docs)}
    if len(conflict_set) < 2:
        return list(docs)

    winner = min(conflict_set, key=lambda i: _doc_priority(docs[i], i))
    losers = sorted(conflict_set - {winner})

    for i in losers:
        src = docs[i].metadata.get("_file_name") or docs[i].metadata.get("_source") or "未知来源"
        logger.info(f"知识冲突仲裁：丢弃 [{i}] {src}（权威度/更新时间低于赢家 [{winner}]）")
    logger.info(f"知识冲突仲裁：{decision.reason or '检出矛盾'}，保留赢家 [{winner}]")

    return [doc for idx, doc in enumerate(docs) if idx not in losers]
