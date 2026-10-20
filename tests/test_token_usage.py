"""Token 用量记账（用量提取 / 回调 / 落库聚合 / 预算）单元测试。"""
from types import SimpleNamespace

import pytest

from app.config import config
from app.services.token_usage import (
    TokenUsageCallback,
    TokenUsageRecorder,
    _extract_usage,
    token_usage_recorder,
)


def _fake_response(llm_output=None, generations=None):
    return SimpleNamespace(llm_output=llm_output or {}, generations=generations or [])


@pytest.fixture
def tmp_recorder(monkeypatch, tmp_path):
    """独立 recorder + 临时用量库，避免污染全局单例的后台线程。"""
    monkeypatch.setattr(config, "token_usage_db_path", str(tmp_path / "token_usage.sqlite"))
    monkeypatch.setattr(config, "token_usage_enabled", True)
    monkeypatch.setattr(config, "token_budget_enabled", False)
    monkeypatch.setattr(config, "token_budget_daily_limit", 0)
    r = TokenUsageRecorder()
    yield r
    r.stop()


# ---------------------------------------------------------------------------
# _extract_usage：从 LLMResult 里抠 tokens
# ---------------------------------------------------------------------------

def test_extract_usage_from_llm_output_openai_keys():
    resp = _fake_response(llm_output={
        "model_name": "qwen-max",
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    })
    assert _extract_usage(resp) == (100, 50, 150, "qwen-max")


def test_extract_usage_fallback_to_usage_metadata():
    msg = SimpleNamespace(
        usage_metadata=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
    )
    resp = _fake_response(generations=[[SimpleNamespace(message=msg)]])
    inp, out, total, _model = _extract_usage(resp)
    assert (inp, out, total) == (10, 5, 15)


def test_extract_usage_zeros_when_absent():
    inp, out, total, model = _extract_usage(_fake_response())
    assert (inp, out, total, model) == (0, 0, 0, "")


# ---------------------------------------------------------------------------
# TokenUsageCallback：on_llm_end 记一笔 chat
# ---------------------------------------------------------------------------

def test_callback_records_chat_usage(monkeypatch):
    recorded = []
    monkeypatch.setattr(token_usage_recorder, "record", lambda **kw: recorded.append(kw))
    cb = TokenUsageCallback()
    cb.on_llm_end(_fake_response(llm_output={
        "model_name": "qwen-max",
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }))
    assert recorded == [{
        "source": "chat",
        "model": "qwen-max",
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }]


def test_callback_skips_zero_usage(monkeypatch):
    recorded = []
    monkeypatch.setattr(token_usage_recorder, "record", lambda **kw: recorded.append(kw))
    TokenUsageCallback().on_llm_end(_fake_response())
    assert recorded == []


# ---------------------------------------------------------------------------
# record / stats：落库与聚合
# ---------------------------------------------------------------------------

def test_record_and_stats_roundtrip(tmp_recorder):
    tmp_recorder.record(source="chat", model="qwen-max", input_tokens=10, output_tokens=5, total_tokens=15)
    tmp_recorder.record(source="rerank", model="qwen3-rerank", input_tokens=8, total_tokens=8)
    tmp_recorder.stop()

    stats = tmp_recorder.stats()
    assert stats["total"]["records"] == 2
    assert stats["total"]["total_tokens"] == 23
    sources = {r["source"]: r["total_tokens"] for r in stats["by_source"]}
    assert sources == {"chat": 15, "rerank": 8}
    models = {r["model"]: r["calls"] for r in stats["by_model"]}
    assert models == {"qwen-max": 1, "qwen3-rerank": 1}


def test_record_skips_zero_tokens(tmp_recorder):
    tmp_recorder.record(source="chat", model="m", total_tokens=0)
    tmp_recorder.stop()
    assert tmp_recorder.stats()["total"]["records"] == 0


def test_record_skips_when_disabled(tmp_recorder, monkeypatch):
    monkeypatch.setattr(config, "token_usage_enabled", False)
    tmp_recorder.record(source="chat", model="m", total_tokens=5)
    tmp_recorder.stop()
    assert tmp_recorder.stats()["total"]["records"] == 0


# ---------------------------------------------------------------------------
# check_daily_budget：今日预算状态
# ---------------------------------------------------------------------------

def test_check_daily_budget_disabled(tmp_recorder):
    budget = tmp_recorder.check_daily_budget()
    assert budget["enabled"] is False
    assert budget["exceeded"] is False
    assert budget["today_tokens"] == 0


def test_check_daily_budget_exceeded(tmp_recorder, monkeypatch):
    monkeypatch.setattr(config, "token_budget_enabled", True)
    monkeypatch.setattr(config, "token_budget_daily_limit", 10)
    tmp_recorder.record(source="chat", model="m", total_tokens=50)
    tmp_recorder.stop()

    budget = tmp_recorder.check_daily_budget()
    assert budget["enabled"] is True
    assert budget["today_tokens"] == 50
    assert budget["limit"] == 10
    assert budget["exceeded"] is True
