"""知识检索工具 - 从向量数据库中检索相关信息"""
from typing import Tuple, List

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager


def l2_distance_to_relevance(distance:float) -> float:
    """
    将 Milvus L2 距离映射为 [0,1] 的相关性分数（越高越相关）。

    Milvus 的 L2 度量返回的是未开方的欧氏距离平方，归一化向量取值约 [0,4]；
    这里采用与 langchain_milvus 一致的映射：1 - distance / 4.0。
    （本项目的 biz collection 索引 metric 固定为 L2，见 milvus_client._create_index）
    """
    return max(0.0, min(1.0, 1.0 - distance / 4.0))


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

        # 带分数检索：similarity_search_with_score 返回 (doc, L2 距离)
        vector_store = vector_store_manager.get_vector_store()
        scored = vector_store.similarity_search_with_score(
            query, k=config.rag_top_k
        )

        # 相关性过滤：L2 距离 -> 相关性分数，低于阈值视为不可靠，不喂给模型
        threshold = config.rag_relevance_threshold
        relevant = []
        best_score = 0.0
        for doc, distance in scored:
            score = l2_distance_to_relevance(distance)
            best_score = max(best_score, score)
            if score >= threshold:
                relevant.append((doc, score))

        if not relevant:
            logger.warning(
                f"检索结果相关性不足（最高分 {best_score:.3f} < 阈值 {threshold}），"
                f"不提供参考资料，避免喂入无关内容"
            )
            return "未找到可靠的相关资料。", []

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
