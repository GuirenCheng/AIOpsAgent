"""意图识别 + 统一入口（/api/agent）单元与集成测试。"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from app.api import agent as agent_api
from app.config import config
from app.services import intent as intent_module
from app.services.intent import IntentDecision


# ---------------------------------------------------------------------------
# IntentDecision：别名归一化
# ---------------------------------------------------------------------------

def test_intent_decision_normalizes_aliases():
    assert IntentDecision(intent="诊断").intent == "diagnose"
    assert IntentDecision(intent="故障排查").intent == "diagnose"
    assert IntentDecision(intent="troubleshoot").intent == "diagnose"
    assert IntentDecision(intent="Diagnose").intent == "diagnose"
    assert IntentDecision(intent="问答").intent == "chat"
    assert IntentDecision(intent="闲聊").intent == "chat"
    assert IntentDecision(intent="chat").intent == "chat"


def test_intent_name():
    assert intent_module.intent_name("diagnose") == "诊断"
    assert intent_module.intent_name("chat") == "问答"


# ---------------------------------------------------------------------------
# classify_intent：成功与兜底
# ---------------------------------------------------------------------------

async def test_classify_intent_success(monkeypatch):
    async def _fake_chain(_inputs):
        return IntentDecision(intent="diagnose", reason="提到了故障")

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            return RunnableLambda(_fake_chain)

    monkeypatch.setattr(intent_module, "ChatQwen", FakeLLM)
    decision = await intent_module.classify_intent("数据同步服务 CPU 飙高")
    assert decision.intent == "diagnose"
    assert decision.reason == "提到了故障"


async def test_classify_intent_fallback_on_error(monkeypatch):
    async def _boom(_inputs):
        raise RuntimeError("llm boom")

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            return RunnableLambda(_boom)

    monkeypatch.setattr(intent_module, "ChatQwen", FakeLLM)
    decision = await intent_module.classify_intent("你好")
    assert decision.intent == "chat"
    assert "默认问答" in decision.reason


# ---------------------------------------------------------------------------
# /api/agent 统一入口：意图分流
# ---------------------------------------------------------------------------

def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(agent_api.router, prefix="/api")
    return app


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


@pytest.fixture
def client(monkeypatch):
    # 单元测试不依赖 .env 的鉴权开关：显式关闭鉴权，让 require_role 放行（返回合成 admin）
    monkeypatch.setattr(config, "api_auth_enabled", False)
    with TestClient(_build_app()) as c:
        yield c


def test_agent_routes_to_diagnose(client, monkeypatch):
    async def fake_classify(text):
        return IntentDecision(intent="diagnose", reason="提到了故障")

    async def fake_execute(question, session_id):
        yield {"type": "complete", "response": "# 诊断报告"}

    monkeypatch.setattr(agent_api, "classify_intent", fake_classify)
    monkeypatch.setattr(agent_api.aiops_service, "execute", fake_execute)

    resp = client.post("/api/agent", json={"question": "服务挂了", "session_id": "s1"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[0]["type"] == "intent"
    assert events[0]["data"]["intent"] == "diagnose"
    assert any(e.get("type") == "complete" for e in events)


def test_agent_routes_to_chat(client, monkeypatch):
    captured = {}

    async def fake_classify(text):
        return IntentDecision(intent="chat", reason="知识问答")

    async def fake_query_stream(question, session_id):
        captured["session_id"] = session_id
        yield {"type": "content", "data": "这是答案"}
        yield {"type": "complete"}

    monkeypatch.setattr(agent_api, "classify_intent", fake_classify)
    monkeypatch.setattr(agent_api.rag_agent_service, "query_stream", fake_query_stream)

    resp = client.post("/api/agent", json={"question": "什么是向量数据库"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[0]["type"] == "intent"
    assert events[0]["data"]["intent"] == "chat"
    assert any(e.get("type") == "content" for e in events)
    assert captured["session_id"] == "default"
