"""DashScope 重排客户端（精排）。

对 RRF 融合出的候选文档做语义重排，得到 [0,1] 的相关性分数，供阈值过滤 + top_k。
模型默认 qwen3-rerank（gte-rerank 已于 2026-05 下线）。失败抛异常，交上层降级。
"""
from http import HTTPStatus
from typing import List, Tuple

import dashscope
from langchain_core.documents import Document
from loguru import logger

from app.config import config

# 显式注入统一 API key（SDK 默认读 DASHSCOPE_API_KEY 环境变量，这里保持一致）
dashscope.api_key = config.dashscope_api_key

# 单个文档送入重排的最大字符数（控制成本与延迟；内部 runbook chunk 通常远小于此）
_MAX_DOC_CHARS = 2000


def rerank_documents(
    query: str,
    docs: List[Document],
    top_n: int,
) -> List[Tuple[Document, float]]:
    """对候选文档重排，返回 [(doc, relevance_score)] 按分数降序。

    Raises:
        RuntimeError: DashScope 返回非 200 或结果解析失败时抛出。
    """
    if not docs:
        return []

    documents = [(d.page_content or "")[:_MAX_DOC_CHARS] for d in docs]
    resp = dashscope.TextReRank.call(
        model=config.rag_rerank_model,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False,
    )

    if getattr(resp, "status_code", None) != HTTPStatus.OK:
        code = getattr(resp, "code", "")
        message = getattr(resp, "message", "")
        raise RuntimeError(f"DashScope 重排失败：code={code} message={message}")

    results = getattr(getattr(resp, "output", None), "results", None) or []
    out: List[Tuple[Document, float]] = []
    for r in results:
        idx = getattr(r, "index", None)
        score = float(getattr(r, "relevance_score", 0.0))
        if idx is None or idx < 0 or idx >= len(docs):
            continue
        out.append((docs[idx], score))

    out.sort(key=lambda t: t[1], reverse=True)
    logger.debug(f"重排完成：候选 {len(docs)} → 返回 {len(out)}（模型 {config.rag_rerank_model}）")
    return out
