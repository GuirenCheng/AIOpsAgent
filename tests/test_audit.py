"""AuditLogger：SQLite 落库、list_recent、异常兜底（tmp_path，不写真实 data/）。"""
from app.harness.audit import AuditLogger


async def test_record_and_list_recent(tmp_path):
    lg = AuditLogger(db_path=str(tmp_path / "audit.sqlite"))
    try:
        await lg.record(tool="search_log", server="cls", args={"q": "error"},
                        status="success", duration_ms=12.3)
        await lg.record(tool="restart_service", server="monitor",
                        status="error", error="boom", duration_ms=5.0)

        rows = await lg.list_recent(limit=10)
        assert len(rows) == 2
        # 倒序：最新在前
        assert rows[0]["tool"] == "restart_service"
        assert rows[0]["status"] == "error"
        assert rows[1]["tool"] == "search_log"
        assert rows[1]["status"] == "success"
        assert rows[1]["args"] == '{"q": "error"}'
    finally:
        await lg.close()


async def test_record_swallows_db_errors(tmp_path, monkeypatch):
    lg = AuditLogger(db_path=str(tmp_path / "audit.sqlite"))

    async def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(lg, "_get_conn", boom)

    # 审计写失败不应抛出异常（不打断真实工具调用）
    await lg.record(tool="x", status="success")
    await lg.close()


async def test_query_filters(tmp_path):
    lg = AuditLogger(db_path=str(tmp_path / "audit.sqlite"))
    try:
        await lg.record(tool="search_log", server="cls", status="success")
        await lg.record(tool="restart_service", server="monitor", status="error", error="boom")
        await lg.record(tool="rollback_service", server="monitor", status="rollback_failed", error="x")

        assert [r["tool"] for r in await lg.query(status="error")] == ["restart_service"]
        assert [r["status"] for r in await lg.query(tool="rollback_service")] == ["rollback_failed"]
        assert len(await lg.query(server="monitor")) == 2
        # 倒序：最新在前
        assert [r["tool"] for r in await lg.query(limit=10)] == [
            "rollback_service", "restart_service", "search_log"
        ]
    finally:
        await lg.close()


async def test_query_since_before_filter(tmp_path):
    lg = AuditLogger(db_path=str(tmp_path / "audit.sqlite"))
    try:
        await lg.record(tool="a", status="success")
        future = "2999-01-01T00:00:00+00:00"
        past = "2000-01-01T00:00:00+00:00"
        assert await lg.query(since=future) == []
        assert len(await lg.query(since=past)) == 1
        assert await lg.query(before=past) == []
    finally:
        await lg.close()


async def test_status_counts(tmp_path):
    lg = AuditLogger(db_path=str(tmp_path / "audit.sqlite"))
    try:
        await lg.record(tool="a", status="success")
        await lg.record(tool="b", status="success")
        await lg.record(tool="c", status="error", error="x")
        assert await lg.status_counts() == {"success": 2, "error": 1}
    finally:
        await lg.close()
