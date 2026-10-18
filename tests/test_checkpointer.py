"""会话检查点（checkpointer）多进程写安全（C1）：WAL + busy_timeout。"""
import sqlite3

from app.config import config
from app.core.checkpointer import _apply_multiprocess_pragmas, create_checkpointer


def test_apply_multiprocess_pragmas_sets_wal(tmp_path):
    db = str(tmp_path / "cp.sqlite")
    _apply_multiprocess_pragmas(db)
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_apply_multiprocess_pragmas_idempotent_and_creates_file(tmp_path):
    db = str(tmp_path / "cp.sqlite")
    _apply_multiprocess_pragmas(db)
    _apply_multiprocess_pragmas(db)  # 重复调用不抛异常
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


async def test_create_checkpointer_sqlite_applies_wal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "session_store_enabled", True)
    db = str(tmp_path / "cp.sqlite")
    monkeypatch.setattr(config, "session_store_db_path", db)

    # AsyncSqliteSaver.__init__ 会取 get_running_loop()，必须在事件循环内创建（生产亦然）
    cp = create_checkpointer()

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    assert isinstance(cp, AsyncSqliteSaver)
    # WAL 是文件级持久属性，创建 checkpointer 时已设置好（多 worker 并发写安全）
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_workers_config_default_is_one():
    """多 worker 安全告警依赖 WORKERS 默认值（单 worker），误设会误告警。"""
    from app.config import Settings

    assert Settings.model_fields["workers"].default == 1
