"""审计查询与监控 API（D2）。

把写进 data/audit.sqlite 的工具调用审计暴露为可查询接口，并提供按状态聚合的
汇总大盘（供 /audit 页面做「审计可视化 + agent 自身监控」）。
"""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter

from app.harness.audit import audit_logger

router = APIRouter()

# 需要人工留意的异常状态（出现在「最近异常」面板里）
ANOMALY_STATUSES = {
    "error",
    "timeout",
    "overloaded",
    "conflict",
    "rollback_required",
    "rollback_failed",
    "rollback_skipped",
    "approval_expired",
    "approval_denied",
    "denied",
}


@router.get('/harness/audit')
async def list_audit(
    limit: int = 100,
    status: str | None = None,
    tool: str | None = None,
    server: str | None = None,
    since_minutes: int = 0,
):
    """查询审计记录（按时间倒序）。since_minutes>0 时只看最近 N 分钟。"""
    since = None
    if since_minutes > 0:
        since = (datetime.now(timezone.utc) - timedelta(minutes=since_minutes)).isoformat()

    items = await audit_logger.query(
        limit=min(limit, 1000),
        status=status or None,
        tool=tool or None,
        server=server or None,
        since=since,
    )
    return {'code': 200, 'message': 'ok', 'data': {'items': items, 'count': len(items)}}


@router.get('/harness/audit/summary')
async def audit_summary(recent_anomalies: int = 20):
    """审计大盘：按状态计数 + 最近异常记录（供监控卡片展示）。"""
    counts = await audit_logger.status_counts()
    recent = await audit_logger.query(limit=1000)
    anomalies = [r for r in recent if r["status"] in ANOMALY_STATUSES][:recent_anomalies]
    return {
        'code': 200,
        'message': 'ok',
        'data': {'status_counts': counts, 'recent_anomalies': anomalies},
    }
