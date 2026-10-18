"""审计 API（D2）：查询 + 汇总，用临时 SQLite 避免写真实 data/。"""
from app.api import audit as audit_api
from app.harness.audit import AuditLogger


async def test_list_audit_returns_records(tmp_path, monkeypatch):
    lg = AuditLogger(db_path=str(tmp_path / "audit.sqlite"))
    await lg.record(tool="search_log", server="cls", status="success")
    await lg.record(tool="restart_service", server="monitor", status="error", error="boom")
    monkeypatch.setattr(audit_api, "audit_logger", lg)

    try:
        body = await audit_api.list_audit(limit=10)
        assert body["code"] == 200
        items = body["data"]["items"]
        assert len(items) == 2
        assert items[0]["tool"] == "restart_service"  # 倒序：最新在前

        filtered = await audit_api.list_audit(limit=10, status="error")
        assert [r["tool"] for r in filtered["data"]["items"]] == ["restart_service"]
    finally:
        await lg.close()


async def test_audit_summary_counts_and_anomalies(tmp_path, monkeypatch):
    lg = AuditLogger(db_path=str(tmp_path / "audit.sqlite"))
    await lg.record(tool="a", status="success")
    await lg.record(tool="b", status="rollback_failed", error="x")
    await lg.record(tool="c", status="approval_expired", error="appr-1")
    monkeypatch.setattr(audit_api, "audit_logger", lg)

    try:
        body = await audit_api.audit_summary(recent_anomalies=10)
        assert body["code"] == 200
        assert body["data"]["status_counts"] == {
            "success": 1, "rollback_failed": 1, "approval_expired": 1,
        }
        anomalies = body["data"]["recent_anomalies"]
        assert {r["status"] for r in anomalies} == {"rollback_failed", "approval_expired"}
    finally:
        await lg.close()
