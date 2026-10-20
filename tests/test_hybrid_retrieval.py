"""混合检索（BM25 关键词索引 / RRF 融合 / DashScope 重排 / 编排）单元测试。"""
import pytest
from langchain_core.documents import Document

from app.services import hybrid_retrieval as hr
from app.services import keyword_index as ki
from app.services import reranker as rk


# ---------------------------------------------------------------------------
# tokenize：中文 bigram + ASCII 段
# ---------------------------------------------------------------------------

def test_tokenize_chinese_bigram_and_ascii():
    toks = ki.tokenize("CPU飙高到99%")
    assert "cpu" in toks
    assert "99" in toks
    assert "飙高" in toks  # 相邻二字 bigram
    assert "飙" in toks and "高" in toks  # 单字


def test_tokenize_keeps_ascii_run_with_underscore():
    toks = ki.tokenize("data_sync")
    assert "data_sync" in toks


# ---------------------------------------------------------------------------
# KeywordIndex：构建 / 检索 / 空语料 / 脏标记重建
# ---------------------------------------------------------------------------

class _FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def query(self, expr="", output_fields=None, offset=0, limit=1000):
        return self.rows[offset:offset + limit]


def test_keyword_index_search_ranks_matched_doc_first(monkeypatch):
    rows = [
        {"content": "CPU飙高到99%，需要扩容", "metadata": {"_source": "a.md"}},
        {"content": "磁盘写满需要清理日志", "metadata": {"_source": "b.md"}},
    ]
    monkeypatch.setattr(ki.milvus_manager, "get_collection", lambda: _FakeCollection(rows))
    idx = ki.KeywordIndex()
    results = idx.search("CPU 飙高")
    assert results
    assert "CPU飙高" in results[0][0].page_content


def test_keyword_index_empty_when_milvus_unavailable(monkeypatch):
    def _boom():
        raise RuntimeError("not connected")

    monkeypatch.setattr(ki.milvus_manager, "get_collection", _boom)
    idx = ki.KeywordIndex()
    assert idx.search("任意查询") == []


def test_keyword_index_rebuild_after_dirty(monkeypatch):
    fake = _FakeCollection([{"content": "AAA 重启", "metadata": {"_source": "a.md"}}])
    monkeypatch.setattr(ki.milvus_manager, "get_collection", lambda: fake)
    idx = ki.KeywordIndex()
    assert idx.search("AAA")

    fake.rows = [{"content": "BBB 扩容", "metadata": {"_source": "b.md"}}]
    idx.mark_dirty()
    results = idx.search("BBB")
    assert results and results[0][0].page_content == "BBB 扩容"


# ---------------------------------------------------------------------------
# rrf_fuse：去重 + 排名融合
# ---------------------------------------------------------------------------

def test_rrf_fuse_dedup_and_rank():
    d0 = Document(page_content="x", metadata={"_source": "a"})
    d1 = Document(page_content="y", metadata={"_source": "b"})
    d2 = Document(page_content="z", metadata={"_source": "c"})
    fused = hr.rrf_fuse([(d0, 1.0), (d1, 0.9)], [(d1, 3.0), (d2, 2.0)])
    docs = [d for d, _ in fused]
    assert docs[0] is d1  # d1 两路都命中，融合分最高
    assert {id(d) for d in docs} == {id(d0), id(d1), id(d2)}


# ---------------------------------------------------------------------------
# reranker：DashScope 重排
# ---------------------------------------------------------------------------

class _ResultItem:
    def __init__(self, index, score):
        self.index = index
        self.relevance_score = score


class _Resp:
    def __init__(self, status_code, results=None, code="", message=""):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.output = type("Out", (), {"results": results or []})()


def test_rerank_documents_success(monkeypatch):
    docs = [Document(page_content="a"), Document(page_content="b")]
    resp = _Resp(200, [_ResultItem(1, 0.9), _ResultItem(0, 0.6)])
    monkeypatch.setattr(rk.dashscope.TextReRank, "call", lambda **kw: resp)
    out = rk.rerank_documents("q", docs, top_n=2)
    assert out == [(docs[1], 0.9), (docs[0], 0.6)]


def test_rerank_documents_error_raises(monkeypatch):
    resp = _Resp(400, code="InvalidParameter", message="bad model")
    monkeypatch.setattr(rk.dashscope.TextReRank, "call", lambda **kw: resp)
    with pytest.raises(RuntimeError):
        rk.rerank_documents("q", [Document(page_content="a")], top_n=1)


# ---------------------------------------------------------------------------
# hybrid_retrieve：编排 + 回退
# ---------------------------------------------------------------------------

def test_hybrid_retrieve_end_to_end(monkeypatch):
    d0 = Document(page_content="disk full", metadata={"_source": "a"})
    d1 = Document(page_content="cpu high", metadata={"_source": "b"})
    monkeypatch.setattr(hr, "dense_recall", lambda q, k: [(d0, 0.8), (d1, 0.7)])
    monkeypatch.setattr(hr.keyword_index, "search", lambda q, k: [(d1, 3.0)])
    monkeypatch.setattr(hr, "rerank_documents", lambda q, docs, top_n: [(d1, 0.95), (d0, 0.6)])
    monkeypatch.setattr(hr.config, "rag_rerank_enabled", True)
    monkeypatch.setattr(hr.config, "rag_relevance_threshold", 0.5)
    monkeypatch.setattr(hr.config, "rag_top_k", 3)
    result = hr.hybrid_retrieve("cpu")
    assert [d for d, _ in result] == [d1, d0]


def test_hybrid_retrieve_rerank_failure_falls_back_to_rrf(monkeypatch):
    d0 = Document(page_content="x", metadata={"_source": "a"})
    d1 = Document(page_content="y", metadata={"_source": "b"})
    monkeypatch.setattr(hr, "dense_recall", lambda q, k: [(d0, 0.9)])
    monkeypatch.setattr(hr.keyword_index, "search", lambda q, k: [(d1, 3.0)])

    def _boom(q, docs, top_n):
        raise RuntimeError("rerank down")

    monkeypatch.setattr(hr, "rerank_documents", _boom)
    monkeypatch.setattr(hr.config, "rag_rerank_enabled", True)
    monkeypatch.setattr(hr.config, "rag_top_k", 5)
    result = hr.hybrid_retrieve("q")
    assert {id(d) for d, _ in result} == {id(d0), id(d1)}


def test_hybrid_retrieve_sparse_empty_uses_dense_only(monkeypatch):
    d0 = Document(page_content="x", metadata={"_source": "a"})
    monkeypatch.setattr(hr, "dense_recall", lambda q, k: [(d0, 0.9)])
    monkeypatch.setattr(hr.keyword_index, "search", lambda q, k: [])
    monkeypatch.setattr(hr, "rerank_documents", lambda q, docs, top_n: [(d0, 0.9)])
    monkeypatch.setattr(hr.config, "rag_rerank_enabled", True)
    monkeypatch.setattr(hr.config, "rag_relevance_threshold", 0.5)
    monkeypatch.setattr(hr.config, "rag_top_k", 3)
    result = hr.hybrid_retrieve("q")
    assert [d for d, _ in result] == [d0]


def test_hybrid_retrieve_rerank_below_threshold_returns_empty(monkeypatch):
    d0 = Document(page_content="x", metadata={"_source": "a"})
    monkeypatch.setattr(hr, "dense_recall", lambda q, k: [(d0, 0.9)])
    monkeypatch.setattr(hr.keyword_index, "search", lambda q, k: [])
    monkeypatch.setattr(hr, "rerank_documents", lambda q, docs, top_n: [(d0, 0.3)])
    monkeypatch.setattr(hr.config, "rag_rerank_enabled", True)
    monkeypatch.setattr(hr.config, "rag_relevance_threshold", 0.5)
    monkeypatch.setattr(hr.config, "rag_top_k", 3)
    assert hr.hybrid_retrieve("q") == []


def test_hybrid_retrieve_rerank_disabled_returns_fused(monkeypatch):
    d0 = Document(page_content="x", metadata={"_source": "a"})
    d1 = Document(page_content="y", metadata={"_source": "b"})
    monkeypatch.setattr(hr, "dense_recall", lambda q, k: [(d0, 0.9)])
    monkeypatch.setattr(hr.keyword_index, "search", lambda q, k: [(d1, 3.0)])
    monkeypatch.setattr(hr.config, "rag_rerank_enabled", False)
    monkeypatch.setattr(hr.config, "rag_top_k", 5)
    result = hr.hybrid_retrieve("q")
    assert {id(d) for d, _ in result} == {id(d0), id(d1)}
