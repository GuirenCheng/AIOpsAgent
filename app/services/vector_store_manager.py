"""向量存储管理器 - 封装 Milvus VectorStore 操作"""
import threading
from typing import List

from langchain_core.documents import Document
from loguru import logger

from langchain_milvus import Milvus

from app.core.milvus_client import milvus_manager
from app.services.keyword_index import keyword_index
from app.services.vector_embedding_service import vector_embedding_service


class VectorStoreManager:
    """向量存储管理器

    注意：本类不再拥有自己的 Milvus 连接，而是复用 milvus_manager 建立的唯一连接
    （同一条 alias / 同一个 biz collection）。LangChain 的 Milvus 仅在首次使用时
    惰性初始化，避免在 import 阶段、且早于 milvus_manager.connect() 时就建立连接。
    """

    def __init__(self):
        """初始化向量存储管理器（不建立连接）"""
        self.vector_store = None
        # collection 名称以 milvus_manager 为唯一来源，避免两处硬编码漂移
        self.collection_name = milvus_manager.COLLECTION_NAME
        self._init_lock = threading.Lock()

    def _ensure_initialized(self):
        """惰性初始化 LangChain VectorStore（线程安全，幂等）"""
        if self.vector_store is not None:
            return

        with self._init_lock:
            if self.vector_store is not None:
                return

            # 确保 milvus_manager 已建立唯一连接并创建好 collection（schema/生命周期由它负责）。
            # 未连接时 get_collection() 会抛出 RuntimeError，避免 LangChain 自行创建
            # 一份 schema 可能不一致的 collection。
            milvus_manager.get_collection()

            self._initialize_vector_store()

    def _initialize_vector_store(self):
        """初始化 Milvus VectorStore（复用 milvus_manager 的唯一连接）"""
        try:
            # 使用与 milvus_manager 完全一致的 URI，从而复用同一条连接 alias，
            # 而不是再开一条并行连接。
            connection_args = {"uri": milvus_manager.connection_uri}

            # 创建 LangChain Milvus VectorStore
            # 使用 biz collection，字段映射：text_field -> content, vector_field -> vector
            self.vector_store = Milvus(
                embedding_function = vector_embedding_service,
                collection_name = self.collection_name,
                connection_args = connection_args,
                auto_id = False,# 使用自定义ID
                drop_old=False,
                text_field="content",# 文本内容存储到content字段
                vector_field = 'vector',# 向量存储到vector字段
                primary_field="id", # 主键字段
                metadata_field="metadata", # 元数据字段
            )

            logger.info(
                f"VectorStore 初始化成功（复用连接 alias={milvus_manager.alias}）,"
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore初始化失败：{e}")
            raise

    def add_document(self,documents:List[Document]) ->List[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            self._ensure_initialized()
            import time
            import uuid
            start_time = time.time()

            # 为每个文档生成唯一id（因为auto_id = False）
            ids = [str(uuid.uuid4()) for _ in documents]

            # LangChain Milvus 的 add_documents 会自动调用 embedding_function
            # 并进行批量处理，性能更好
            result_ids = self.vector_store.add_documents(documents,ids = ids)

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)}个文档到 VectorStore 完成,"
                f"耗时:{elapsed:.2f}秒，平均：{elapsed/len(documents):.2f}秒/个"
            )
            keyword_index.mark_dirty()
            return result_ids
        except Exception as e:
            logger.error(f"添加文档失败：{e}")
            raise

    def delete_by_source(self,file_path:str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            # 复用 milvus_manager 持有的唯一已连接 collection
            collection = milvus_manager.get_collection()

            # metadata 是 JSON 字段，使用JSON路径查询语法
            # _source 是文档的来源文件路径
            # 转义反斜杠与单引号，防止路径中的特殊字符破坏 Milvus 布尔表达式（注入）
            escaped_path = file_path.replace("\\", "\\\\").replace("'", "\\'")
            expr = f"metadata['_source'] == '{escaped_path}'"

            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, 'delete_count') else 0

            logger.info(f"删除文件旧数据：{file_path},删除数量: {deleted_count}")
            keyword_index.mark_dirty()
            return deleted_count

        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引):{e}")
            return 0

    def get_vector_store(self)->Milvus:
        """
        获取 VectorStore 实例

        Returns:
            Milvus: VectorStore 实例
        """
        self._ensure_initialized()
        return self.vector_store

    def similarity_search(self,query:str,k:int = 3) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表

        Raises:
            RuntimeError: 搜索失败时抛出（不再静默返回空列表，避免调用方误判为「无结果」）
        """
        try:
            self._ensure_initialized()
            docs = self.vector_store.similarity_search(query,k = k)
            logger.debug(f"相似度搜索完成: query='{query}',结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f'相似度搜索失败: {e}')
            raise RuntimeError(f"相似度搜索失败：{e}") from e

# 全局单例
vector_store_manager = VectorStoreManager()