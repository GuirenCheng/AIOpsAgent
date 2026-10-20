"""统一入口：意图识别 + 自动分流到「诊断 / 问答」。

用户用一句大白话提问，先由 classify_intent 判断意图（diagnose/chat），
再路由到 AIOps 诊断（Plan-Execute-Replan）或 RAG 问答。事件流首条为 intent 事件，
其后透传下游服务的原始事件（不归一化，保留 plan/step/content 细节）。
"""
import json

from fastapi import APIRouter, Depends
from loguru import logger
from sse_starlette import EventSourceResponse

from app.core.auth import Principal, Role, require_role
from app.models.request import AgentRequest
from app.services.aiops_service import aiops_service
from app.services.intent import classify_intent, intent_name
from app.services.rag_agent_service import rag_agent_service

router = APIRouter()


@router.post('/agent')
async def agent(request: AgentRequest, _user: Principal = Depends(require_role(Role.OPERATOR, Role.ADMIN))):
    """统一入口（流式 SSE）：意图识别 + 自动分流。

    事件序列：
    1. `intent` —— 首条，告知识别结果（data.intent = diagnose/chat，data.reason = 理由）；
    2. 其后透传下游服务事件：diagnose → AIOps 的 plan/step_complete/report/complete；
       chat → RAG 的 content/complete。
    """
    question = request.question
    session_id = request.session_id or "default"
    logger.info(f"[会话{session_id}] 统一入口收到：{question[:80]}")

    async def event_generator():
        # 1. 意图识别（先于任何下游调用，把结果告诉调用方）
        decision = await classify_intent(question)
        intent = decision.intent
        logger.info(f"[会话{session_id}] 意图识别：{intent}（{intent_name(intent)}）")
        yield {
            'event': 'message',
            'data': json.dumps(
                {'type': 'intent', 'data': {'intent': intent, 'reason': decision.reason}},
                ensure_ascii=False,
            ),
        }

        # 2. 按意图分流透传
        try:
            if intent == 'diagnose':
                async for event in aiops_service.execute(question, session_id=session_id):
                    yield {'event': 'message', 'data': json.dumps(event, ensure_ascii=False)}
            else:
                async for chunk in rag_agent_service.query_stream(question, session_id=session_id):
                    yield {
                        'event': 'message',
                        'data': json.dumps(chunk, ensure_ascii=False, default=str),
                    }
        except Exception as e:  # noqa: BLE001 —— 下游异常也要以事件形式告知，不中断 SSE
            logger.error(f"[会话{session_id}] 统一入口分流执行异常：{e}", exc_info=True)
            yield {
                'event': 'message',
                'data': json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
