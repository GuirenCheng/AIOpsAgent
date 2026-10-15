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
