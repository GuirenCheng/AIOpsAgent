"""知识冲突检测 + 权威/最新自动仲裁单元测试。"""
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.services import knowledge_conflict as kc
from app.services.knowledge_conflict import ConflictDecision
from app.tools import knowledge_tool


def _doc(content, source, authority=1, updated="2026-01-01T00:00:00+00:00"):
    return Document(
        page_content=content,
        metadata={
            "_source": source,
            "_file_name": source.rsplit("/", 1)[-1],
            "_authority": authority,
            "_updated_at": updated,
        },
    )


# ---------------------------------------------------------------------------
# authority_from_path：路径前缀 → 权威度
# ---------------------------------------------------------------------------

def test_authority_from_path_default(monkeypatch):
    monkeypatch.setattr(kc.config, "rag_authority_rules", "")
    assert kc.authority_from_path("/uploads/a.md") == 1


def test_authority_from_path_rules(monkeypatch):
    monkeypatch.setattr(kc.config, "rag_authority_rules", "official:3,runbook:2")
    assert kc.authority_from_path("/uploads/official/x.md") == 3
    assert kc.authority_from_path("C:\\uploads\\runbook\\y.md") == 2
    assert kc.authority_from_path("/uploads/other/z.md") == 1


def test_authority_from_path_most_specific_wins(monkeypatch):
    monkeypatch.setattr(kc.config, "rag_authority_rules", "official:3,official/security:5")
    assert kc.authority_from_path("/uploads/official/security/x.md") == 5


# ---------------------------------------------------------------------------
# resolve_conflict：权威度 / 最新仲裁
# ---------------------------------------------------------------------------

def test_resolve_conflict_authority_wins():
    d0 = _doc("先应用后DB", "a.md", authority=1, updated="2026-06-01T00:00:00+00:00")
    d1 = _doc("先DB后应用", "b.md", authority=3, updated="2026-01-01T00:00:00+00:00")
    survivors = kc.resolve_conflict(
        [d0, d1], ConflictDecision(has_conflict=True, conflicting_indices=[0, 1])
    )
    assert survivors == [d1]


def test_resolve_conflict_latest_wins():
    d0 = _doc("值=1", "a.md", authority=1, updated="2026-01-01T00:00:00+00:00")
    d1 = _doc("值=2", "b.md", authority=1, updated="2026-06-01T00:00:00+00:00")
    survivors = kc.resolve_conflict(
        [d0, d1], ConflictDecision(has_conflict=True, conflicting_indices=[0, 1])
    )
    assert survivors == [d1]


def test_resolve_conflict_keeps_non_conflicting():
    d0 = _doc("值=1", "a.md", authority=1, updated="2026-01-01T00:00:00+00:00")
    d1 = _doc("值=2", "b.md", authority=2, updated="2026-01-01T00:00:00+00:00")
    d2 = _doc("无关内容", "c.md", authority=1, updated="2026-01-01T00:00:00+00:00")
    survivors = kc.resolve_conflict(
        [d0, d1, d2], ConflictDecision(has_conflict=True, conflicting_indices=[0, 1])
    )
    assert survivors == [d1, d2]


def test_resolve_conflict_no_conflict_returns_all():
    docs = [_doc("a", "a.md"), _doc("b", "b.md")]
    survivors = kc.resolve_conflict(docs, ConflictDecision(has_conflict=False))
    assert survivors == docs


def test_resolve_conflict_out_of_range_ignored():
    d0 = _doc("值=1", "a.md", authority=1)
    survivors = kc.resolve_conflict(
        [d0], ConflictDecision(has_conflict=True, conflicting_indices=[0, 5])
    )
    assert survivors == [d0]


# ---------------------------------------------------------------------------
# detect_conflict：LLM 结构化判断 + 兜底
# ---------------------------------------------------------------------------

def test_detect_conflict_returns_decision(monkeypatch):
    def _fake_chain(_inputs):
        return ConflictDecision(has_conflict=True, conflicting_indices=[0, 1], reason="矛盾")

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            return RunnableLambda(_fake_chain)

    monkeypatch.setattr(kc, "ChatQwen", FakeLLM)
    decision = kc.detect_conflict([_doc("a", "a.md"), _doc("b", "b.md")])
    assert decision.has_conflict is True
    assert decision.conflicting_indices == [0, 1]


def test_detect_conflict_fallback_on_error(monkeypatch):
    def _boom(_inputs):
        raise RuntimeError("llm boom")

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            return RunnableLambda(_boom)

    monkeypatch.setattr(kc, "ChatQwen", FakeLLM)
    decision = kc.detect_conflict([_doc("a", "a.md")])
    assert decision.has_conflict is False


# ---------------------------------------------------------------------------
# _apply_conflict_resolution：触发条件 + 仲裁接线
# ---------------------------------------------------------------------------

def test_apply_conflict_resolution_drops_loser(monkeypatch):
    d0 = _doc("值=1", "/uploads/a.md", authority=1)
    d1 = _doc("值=2", "/uploads/b.md", authority=2)
    d2 = _doc("无关", "/uploads/c.md", authority=1)
    monkeypatch.setattr(
        knowledge_tool,
        "detect_conflict",
        lambda docs: ConflictDecision(has_conflict=True, conflicting_indices=[0, 1], reason="矛盾"),
    )
    relevant = [(d0, 0.9), (d1, 0.8), (d2, 0.7)]
    result = knowledge_tool._apply_conflict_resolution(relevant)
    assert [doc for doc, _ in result] == [d1, d2]


def test_apply_conflict_resolution_same_source_skips(monkeypatch):
    d0 = _doc("值=1", "/uploads/a.md", authority=1)
    d1 = _doc("值=2", "/uploads/a.md", authority=1)
    called = {"v": False}

    def fake_detect(docs):
        called["v"] = True
        return ConflictDecision(has_conflict=False)

    monkeypatch.setattr(knowledge_tool, "detect_conflict", fake_detect)
    relevant = [(d0, 0.9), (d1, 0.8)]
    result = knowledge_tool._apply_conflict_resolution(relevant)
    assert result == relevant
    assert called["v"] is False  # 同来源不触发 LLM 检测


# ---------------------------------------------------------------------------
# retrieve_knowledge 集成：检索后只保留赢家
# ---------------------------------------------------------------------------

def test_retrieve_knowledge_resolves_conflict(monkeypatch):
    d0 = _doc("重启顺序：先应用后数据库", "/uploads/a.md", authority=1)
    d1 = _doc("重启顺序：先数据库后应用", "/uploads/b.md", authority=2)

    class FakeStore:
        def similarity_search_with_score(self, query, k=3):
            return [(d0, 0.5), (d1, 0.5)]

    monkeypatch.setattr(knowledge_tool.vector_store_manager, "get_vector_store", lambda: FakeStore())
    monkeypatch.setattr(
        knowledge_tool,
        "detect_conflict",
        lambda docs: ConflictDecision(has_conflict=True, conflicting_indices=[0, 1], reason="矛盾"),
    )
    monkeypatch.setattr(knowledge_tool.config, "rag_conflict_resolution_enabled", True)
    monkeypatch.setattr(knowledge_tool.config, "rag_hybrid_retrieval_enabled", False)  # 本测试专测单路向量 + 冲突仲裁

    context, docs = knowledge_tool.retrieve_knowledge.func("重启顺序")
    assert "b.md" in context
    assert "a.md" not in context
    assert [d.metadata["_file_name"] for d in docs] == ["b.md"]
