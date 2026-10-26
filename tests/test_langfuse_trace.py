"""Langfuse 全链路 trace 单元测试。

沿用 test_token_usage.py 的 monkeypatch 风格：不真正连 Langfuse，只验证
fail-open（未启用/未配齐时不影响主流程）与 per-request ContextVar 的 set/reset。
"""
import builtins

from app.config import config
from app.services import langfuse_trace as lt


def test_get_llm_callbacks_returns_token_usage_only_by_default():
    # 未 setup 任何 trace 上下文时，只含 token_usage_callback（fail-open）
    from app.services.token_usage import token_usage_callback

    callbacks = lt.get_llm_callbacks()
    assert callbacks == [token_usage_callback]


def test_get_llm_callbacks_appends_handler_when_present():
    fake_handler = object()
    token = lt._handler_context.set(fake_handler)
    try:
        callbacks = lt.get_llm_callbacks()
        assert len(callbacks) == 2
        assert callbacks[1] is fake_handler
    finally:
        lt._handler_context.reset(token)


def test_setup_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "langfuse_enabled", False)
    monkeypatch.setattr(lt, "_client", None)
    monkeypatch.setattr(lt, "_client_failed", False)
    assert lt.setup_trace_context("req-123") is None


def test_setup_returns_none_when_missing_keys(monkeypatch):
    monkeypatch.setattr(config, "langfuse_enabled", True)
    monkeypatch.setattr(config, "langfuse_host", "")
    monkeypatch.setattr(config, "langfuse_public_key", "pk")
    monkeypatch.setattr(config, "langfuse_secret_key", "sk")
    monkeypatch.setattr(lt, "_client", None)
    monkeypatch.setattr(lt, "_client_failed", False)
    assert lt.setup_trace_context("req-123") is None


def test_get_client_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "langfuse_enabled", False)
    monkeypatch.setattr(lt, "_client", None)
    monkeypatch.setattr(lt, "_client_failed", False)
    assert lt._get_client() is None


def test_setup_and_teardown_roundtrip(monkeypatch):
    # 跳过真实 Langfuse client，直接伪造 handler 创建，验证 trace_id=request_id 与 set/reset
    monkeypatch.setattr(lt, "_get_client", lambda: object())  # 非 None，视为已初始化
    real_import = builtins.__import__

    class FakeHandler:
        def __init__(self, *, trace_context=None):
            self.trace_context = trace_context

    def _fake_import(name, *args, **kwargs):
        if name == "langfuse.langchain":
            return type("_m", (), {"CallbackHandler": FakeHandler})()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    token = lt.setup_trace_context("req-123")
    assert token is not None
    try:
        handler = lt.get_trace_handler()
        assert handler is not None
        assert handler.trace_context == {"trace_id": "req-123"}
    finally:
        lt.teardown_trace_context(token)
    assert lt.get_trace_handler() is None


def test_langfuse_disabled_by_default():
    from app.config import Settings

    assert Settings.model_fields["langfuse_enabled"].default is False
    assert Settings.model_fields["langfuse_host"].default == ""
