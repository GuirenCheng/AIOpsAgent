"""生产级混合检索编排：向量 + 关键词双路召回 → RRF 融合 → DashScope 重排。

管线（全部收敛到 retrieve_knowledge 这一个检索出口）：

    query
     ├─ 稠密召回：similarity_search_with_score(query, k=rag_recall_top_k)
     ├─ 稀疏召回：keyword_index.search(query, k=rag_recall_top_k)
     ├─ RRF 融合：对两路结果按「排名」做倒数融合（RRF_K=60），去重取并集
     ├─ 重排：DashScope qwen3-rerank 打相关度分（0~1）
     └─ 过滤：relevance_score >= rag_relevance_threshold 取前 rag_top_k

回退（fail-open，绝不因某一路失败而整体不可用）：
    - 重排失败 / 关闭 → 直接用 RRF 融合顺序取前 top_k（不按阈值过滤）。
    - 关键词索引为空 / Milvus 未连 → 退化为仅稠密召回。
"""
from typing import List, Tuple

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.keyword_index import keyword_index
from app.services.reranker import rerank_documents
from app.services.vector_store_manager import vector_store_manager

# RRF 倒数融合的平滑常数（越大对排名越不敏感，60 是业界常用值）
RRF_K = 60


def l2_distance_to_relevance(distance: float) -> float:
    """将 Milvus L2 距离映射为 [0,1] 的相关性分数（越高越相关）。

    Milvus 的 L2 度量返回的是未开方的欧氏距离平方，归一化向量取值约 [0,4]；
    这里采用与 langchain_milvus 一致的映射：1 - distance / 4.0。
    （本项目的 biz collection 索引 metric 固定为 L2，见 milvus_client._create_index）
    """
    return max(0.0, min(1.0, 1.0 - distance / 4.0))


def _doc_key(doc: Document) -> Tuple[str, str]:
    """去重键：来源文件 + 文本内容，稠密路与关键词路可稳定对齐。"""
    source = (doc.metadata or {}).get("_source", "")
    return (source, doc.page_content)


def dense_recall(query: str, k: int) -> List[Tuple[Document, float]]:
    """稠密（向量）召回：返回 [(doc, relevance_score)]，按分数降序。"""
    vector_store = vector_store_manager.get_vector_store()
    scored = vector_store.similarity_search_with_score(query, k=k)
    return [(doc, l2_distance_to_relevance(distance)) for doc, distance in scored]


def rrf_fuse(*ranked_lists: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """把多路「按相关度排好序」的召回结果做倒数融合，返回 [(doc, rrf_score)] 降序。

    rrf_score = Σ 1/(RRF_K + rank)，rank 从 1 开始；同一 doc 多路命中则累加。
    只依赖「排名」，不依赖各路分数的量纲，天然适配稠密/稀疏分数不可比的问题。
    """
    acc: dict = {}
    for lst in ranked_lists:
        for rank, (doc, _score) in enumerate(lst, start=1):
            key = _doc_key(doc)
            rrf = 1.0 / (RRF_K + rank)
            if key in acc:
                acc[key][1] += rrf
            else:
                acc[key] = [doc, rrf]
    fused = [(doc, score) for doc, score in acc.values()]
    fused.sort(key=lambda t: t[1], reverse=True)
    return fused


def hybrid_retrieve(query: str) -> List[Tuple[Document, float]]:
    """混合检索：返回最终 [(doc, score)]（已阈值过滤 + top_k）。"""
    dense = dense_recall(query, config.rag_recall_top_k)

    try:
        sparse = keyword_index.search(query, config.rag_recall_top_k)
    except Exception as e:  # noqa: BLE001 —— 关键词路失败不影响稠密主路
        logger.warning(f"关键词召回失败，退化为仅向量召回：{e}")
        sparse = []

    fused = rrf_fuse(dense, sparse) if sparse else dense
    if not fused:
        return []

    if not config.rag_rerank_enabled:
        return fused[: config.rag_top_k]

    try:
        docs = [doc for doc, _ in fused]
        reranked = rerank_documents(query, docs, top_n=len(docs))
        # 重排成功后按阈值诚实过滤：分数不达标就不喂（与单路向量行为一致）
        out = [(doc, score) for doc, score in reranked if score >= config.rag_relevance_threshold]
        return out[: config.rag_top_k]
    except Exception as e:  # noqa: BLE001 —— 重排失败降级为 RRF 顺序
        logger.warning(f"重排失败，退化为 RRF 融合顺序：{e}")
        return fused[: config.rag_top_k]
