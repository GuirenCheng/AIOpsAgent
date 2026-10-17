"""告警 Webhook 接入

接收监控平台（Prometheus Alertmanager / 通用 JSON）的告警回调，
立即返回 202，并在后台启动一次 AIOps 诊断。
"""
import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.harness.notify import notify_alert_diagnosis
from app.models.aiops import AlertWebhook, WebhookResponse
from app.services.aiops_service import aiops_service

router = APIRouter()

# 持有后台任务强引用，避免被 GC 提前回收
_pending_tasks: set = set()


def _normalize_alert(body: dict) -> AlertWebhook:
    '''兼容 Prometheus Alertmanager 与通用 JSON 两种格式，归一化为 AlertWebhook。'''
    alerts = body.get("alerts")
    if isinstance(alerts, list) and alerts:
        first = alerts[0]
        if not isinstance(first, dict):
            first = {}
        labels = first.get("labels") or {}
        annotations = first.get("annotations") or {}
        return AlertWebhook(
            alertname=labels.get("alertname") or first.get("alertname") or "unknown",
            severity=labels.get("severity") or first.get("severity") or "warning",
            instance=labels.get("instance") or first.get("instance") or "",
            description=annotations.get("description") or annotations.get("summary")
                       or first.get("description"),
            starts_at=first.get("startsAt"),
        )

    return AlertWebhook(
        alertname=body.get("alertname") or body.get("name") or "unknown",
        severity=body.get("severity") or body.get("level") or "warning",
        instance=body.get("instance") or body.get("host") or "",
        description=body.get("description") or body.get("message"),
        starts_at=body.get("starts_at") or body.get("startedAt") or body.get("time"),
    )


async def _run_diagnosis(alert: AlertWebhook, session_id: str) -> None:
    '''后台消费诊断事件流，收集最终报告并推送到已配置通知通道。'''
    final_report = ""
    try:
        async for event in aiops_service.diagnose_alert(session_id=session_id, alert=alert):
            etype = event.get("type")
            if etype == "report":
                final_report = event.get("report", "")
            elif etype == "complete":
                diag = event.get("diagnosis") or {}
                final_report = diag.get("report", final_report)
            elif etype == "error":
                logger.error(f"[webhook {alert.alertname}] 诊断出错：{event.get('message')}")
        logger.info(f"[webhook {alert.alertname}] 诊断完成，报告长度 {len(final_report)} 字符")
        await notify_alert_diagnosis(alert.alertname, alert.severity, final_report)
    except Exception as e:
        logger.error(f"[webhook {alert.alertname}] 后台诊断异常：{e}", exc_info=True)


@router.post("/webhook/alert", response_model=WebhookResponse, status_code=202)
async def alert_webhook(request: Request):
    '''接收告警回调：立即 202，后台异步诊断。'''
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是合法 JSON")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")

    alert = _normalize_alert(body)
    session_id = f"webhook-{alert.alertname}-{uuid.uuid4().hex[:8]}"
    logger.info(
        f"收到告警 webhook：{alert.alertname}"
        f"（severity={alert.severity}, instance={alert.instance or '-'}）"
    )

    task = asyncio.create_task(_run_diagnosis(alert, session_id))
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)

    return WebhookResponse(alertname=alert.alertname)
