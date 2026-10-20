"""知识检索工具 - 从向量数据库中检索相关信息"""
from typing import Tuple, List

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.hybrid_retrieval import hybrid_retrieve, l2_distance_to_relevance
from app.services.knowledge_conflict import detect_conflict, resolve_conflict
from app.services.vector_store_manager import vector_store_manager


@tool(response_format='content_and_artifact')
def retrieve_knowledge(query:str) -> Tuple[str,List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。
    仅返回相关性达标的文档；检索结果不可靠时会明确告知「未找到可靠资料」。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用:query='{query}'")

        # 检索：默认走混合管线（向量+关键词→RRF→重排），关闭则退回单路向量。
        if config.rag_hybrid_retrieval_enabled:
            relevant = hybrid_retrieve(query)
        else:
            relevant = _vector_retrieve(query)

        if not relevant:
            logger.warning(
                f"检索结果相关性不足（阈值 {config.rag_relevance_threshold}），"
                f"不提供参考资料，避免喂入无关内容"
            )
            return "未找到可靠的相关资料。", []

        docs = [doc for doc, _ in relevant]

        # 知识冲突检测 + 权威/最新自动仲裁：多条不同来源资料间若存在事实矛盾，
        # 按「权威度降序 → 更新时间降序」只保留赢家，避免把矛盾内容同时喂给模型。
        if config.rag_conflict_resolution_enabled:
            relevant = _apply_conflict_resolution(relevant)
            docs = [doc for doc, _ in relevant]

        context = (
            "请严格基于下方参考资料作答，并在每个事实结论后标注引用编号"
            "（形如【参考资料 N】）。\n\n" + format_docs(docs)
        )

        scores = [round(score, 3) for _, score in relevant]
        logger.info(f"检索到 {len(docs)} 个相关文档（相关性分数 {scores}）")
        return context, docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败:{e}")
        return f"检索知识时发生错误:{str(e)}",[]


def _vector_retrieve(query: str) -> List[Tuple[Document, float]]:
    """单路向量检索（混合检索关闭时的回退路径）：带分数检索 + 阈值过滤。

    Returns:
        List[Tuple[Document, float]]: [(doc, relevance_score)]，仅保留分数达标的文档。
    """
    vector_store = vector_store_manager.get_vector_store()
    scored = vector_store.similarity_search_with_score(query, k=config.rag_top_k)

    threshold = config.rag_relevance_threshold
    relevant = []
    for doc, distance in scored:
        score = l2_distance_to_relevance(distance)
        if score >= threshold:
            relevant.append((doc, score))
    return relevant


def _apply_conflict_resolution(relevant: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """对检索结果做冲突检测 + 仲裁，返回幸存 (doc, score) 列表（保原检索序）。

    只在「≥2 条且来自 ≥2 个不同来源」时触发，同文件多分片不算冲突，省 LLM 调用。
    """
    if len(relevant) < 2:
        return relevant

    docs = [doc for doc, _ in relevant]
    sources = {doc.metadata.get("_source", "") for doc in docs}
    if len(sources) < 2:
        return relevant

    decision = detect_conflict(docs)
    survivors = resolve_conflict(docs, decision)
    if len(survivors) == len(docs):
        return relevant

    survivor_ids = {id(doc) for doc in survivors}
    return [(doc, score) for doc, score in relevant if id(doc) in survivor_ids]


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []
    for i,doc in enumerate(docs,1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name","未知来源")

        # 提取标题信息(如果有）
        headers = []
        for key in ['h1','h2','h3']:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = ' > '.join(headers) if headers else ""

        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题:{header_str}"
        formatted += f"\n来源:{source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return  "\n".join(formatted_parts)
