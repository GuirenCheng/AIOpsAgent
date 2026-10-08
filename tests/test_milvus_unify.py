"""Milvus 实例统一化回归测试。

回归点：此前存在两条并行连接路径 ——
1) milvus_manager 用 connections.connect(alias='default') + MilvusClient(uri) 各开一条连接；
2) vector_store_manager 在 import 阶段就构造 LangChain Milvus，再开一条自己的连接。

两者同时管理 biz collection，且 vector_store_manager 在 main 生命周期调用
milvus_manager.connect() 之前就尝试连接，导致时序错误与 schema 双写风险。

本测试验证：
- vector_store_manager 改为惰性初始化，import 时不再建立连接；
- biz collection 名称以 milvus_manager 为唯一来源；
- 未连接时惰性初始化会被守卫拦截（RuntimeError），而非静默创建一份 schema 可能不一致的 collection。
"""
import pytest

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_store_manager import vector_store_manager


def test_vector_store_manager_is_lazy_on_import():
    """import 后不应建立任何连接，vector_store 保持未初始化状态。"""
    assert vector_store_manager.vector_store is None


def test_collection_name_has_single_source_of_truth():
    """biz collection 名称只应有一处定义，两个管理器引用同一常量。"""
    assert milvus_manager.COLLECTION_NAME == "biz"
    assert vector_store_manager.collection_name == milvus_manager.COLLECTION_NAME


def test_connection_uri_is_derived_from_config():
    """连接 URI 应由配置统一生成，供 langchain 复用同一条连接。"""
    expected = f"http://{config.milvus_host}:{config.milvus_port}"
    assert milvus_manager.connection_uri == expected


def test_lazy_init_guards_against_unconnected_use():
    """未调用 milvus_manager.connect() 时，惰性初始化应抛出 RuntimeError，
    而非让 LangChain 自行建立连接 / 创建 collection。"""
    # 仅在确实未连接时断言；若测试环境已连接则跳过守卫断言。
    if milvus_manager.alias is None:
        with pytest.raises(RuntimeError):
            vector_store_manager._ensure_initialized()
