"""告警 webhook 归一化与诊断触发逻辑单元测试"""
from app.api.webhook import _normalize_alert
from app.models.aiops import AlertWebhook
from app.services.aiops_service import AIOpsService


# ---------------------------------------------------------------------------
# _normalize_alert：两种格式归一化
# ---------------------------------------------------------------------------

def test_normalize_alertmanager_format():
    body = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "HighCPUUsage",
                    "severity": "critical",
                    "instance": "web-01:9100",
                },
                "annotations": {
                    "summary": "CPU 使用率过高",
                    "description": "web-01 的 CPU 已连续 5 分钟超过 90%",
                },
                "startsAt": "2026-10-17T10:00:00Z",
            }
        ],
    }
    alert = _normalize_alert(body)
    assert alert.alertname == "HighCPUUsage"
    assert alert.severity == "critical"
    assert alert.instance == "web-01:9100"
    assert alert.description == "web-01 的 CPU 已连续 5 分钟超过 90%"
    assert alert.starts_at == "2026-10-17T10:00:00Z"


def test_normalize_generic_format():
    body = {
        "alertname": "DiskSpaceLow",
        "level": "warning",
        "host": "db-02",
        "message": "磁盘使用率达到 85%",
    }
    alert = _normalize_alert(body)
    assert alert.alertname == "DiskSpaceLow"
    assert alert.severity == "warning"
    assert alert.instance == "db-02"
    assert alert.description == "磁盘使用率达到 85%"


def test_normalize_empty_body_uses_defaults():
    alert = _normalize_alert({})
    assert alert.alertname == "unknown"
    assert alert.severity == "warning"
    assert alert.instance == ""
    assert alert.description is None
    assert alert.starts_at is None


# ---------------------------------------------------------------------------
# diagnose_alert / _diagnose_with_task：告警驱动诊断
# ---------------------------------------------------------------------------

async def test_diagnose_with_task_wraps_complete_event(monkeypatch):
    service = AIOpsService()

    async def fake_execute(task, session_id):
        yield {"type": "complete", "stage": "complete", "response": "# 诊断报告"}

    monkeypatch.setattr(service, "execute", fake_execute)
    events = [e async for e in service._diagnose_with_task("任务", "sess")]

    assert len(events) == 1
    assert events[0]["type"] == "complete"
    assert events[0]["diagnosis"]["status"] == "completed"
    assert events[0]["diagnosis"]["report"] == "# 诊断报告"


async def test_diagnose_alert_builds_task_with_alert_fields(monkeypatch):
    service = AIOpsService()
    captured = {}

    async def fake_execute(task, session_id):
        captured["task"] = task
        yield {"type": "complete", "response": "x"}

    monkeypatch.setattr(service, "execute", fake_execute)
    alert = AlertWebhook(
        alertname="HighCPUUsage",
        severity="critical",
        instance="web-01",
        description="CPU 连续 5 分钟超过 90%",
        starts_at="2026-10-17T10:00:00Z",
    )
    [e async for e in service.diagnose_alert("sess", alert)]

    task = captured["task"]
    assert "HighCPUUsage" in task
    assert "critical" in task
    assert "web-01" in task
    assert "CPU 连续 5 分钟超过 90%" in task
    assert "2026-10-17T10:00:00Z" in task
    assert "# 告警分析报告" in task
