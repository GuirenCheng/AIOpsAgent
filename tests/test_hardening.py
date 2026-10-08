"""生产加固回归测试（第 3 项数据安全 + P2 加固配置）。

回归点：
- 向量维度散落多处硬编码，改动 embedding 模型维度时会触发 Milvus 静默 drop collection 丢数据；
- CORS 此前为 allow_origins=["*"] + allow_credentials=True（浏览器会拒绝的无效组合）；
- 文件上传存在目录穿越风险；问题长度无上限。
"""
import pytest
from pydantic import ValidationError

from app.api.file import _sanitize_filename
from app.config import config
from app.core.milvus_client import milvus_manager
from app.models.request import ChatRequest


def test_vector_dim_has_single_source():
    """向量维度以 config 为唯一来源。"""
    assert milvus_manager.VECTOR_DIM == config.milvus_vector_dim == 1024


def test_dimension_migration_disabled_by_default():
    """维度不匹配时默认禁止自动删除重建（防止静默丢数据）。"""
    assert config.milvus_allow_dimension_migration is False


def test_cors_defaults_to_same_origin():
    """默认不发送 CORS 头（同源访问），且不开启凭据。"""
    assert config.cors_origins == []
    assert config.cors_allow_credentials is False


def test_auth_off_and_rate_limit_on_by_default():
    """鉴权默认关闭（向后兼容），限流默认开启。"""
    assert config.api_auth_enabled is False
    assert config.rate_limit_enabled is True
    assert config.rate_limit_per_minute > 0


def test_question_length_is_validated():
    """超长问题应被请求模型拒绝。"""
    with pytest.raises(ValidationError):
        ChatRequest(Id="s1", Question="x" * (config.max_question_length + 1))

    ok = ChatRequest(Id="s1", Question="hello")
    assert ok.question == "hello"


def test_sanitize_filename_neutralizes_traversal():
    """文件名规范化应消除 '..' 目录穿越与路径分隔符。"""
    assert _sanitize_filename("..") == ""
    assert _sanitize_filename("../etc/passwd") == "_etc_passwd"
    assert _sanitize_filename("..\\windows\\system32") == "_windows_system32"
    assert "/" not in _sanitize_filename("a/b.txt")
    assert "\\" not in _sanitize_filename("a\\b.txt")


def test_checkpointer_factory_returns_a_usable_checkpointer():
    """会话检查点工厂即使无 SQLite 依赖也能返回可用 checkpointer（回退内存）。"""
    from app.core.checkpointer import create_checkpointer

    cp = create_checkpointer()
    assert cp is not None
    # 具备 LangGraph checkpointer 的基本接口
    assert hasattr(cp, "get")
    assert hasattr(cp, "delete_thread")


def test_l2_relevance_mapping():
    """L2 距离到相关性分数的映射应单调递减且落在 [0,1]。"""
    from app.tools.knowledge_tool import l2_distance_to_relevance

    assert l2_distance_to_relevance(0.0) == 1.0
    assert l2_distance_to_relevance(4.0) == 0.0
    assert abs(l2_distance_to_relevance(2.0) - 0.5) < 1e-9
    assert l2_distance_to_relevance(10.0) == 0.0  # 越界裁剪
