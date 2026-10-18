"""结构化工具调用审计：把每次 MCP 工具调用落 SQLite，供排查与合规。

复用 aiosqlite（与 checkpointer.py 相同），但用独立库 data/audit.sqlite，
避免与 LangGraph 的 checkpoints.sqlite schema 互相干扰。

审计写失败绝不打断真实工具调用（record 内部 try/except 兜底）。
"""
import json
import os
from datetime import datetime, timezone

from loguru import logger

from app.config import config
from app.core.middleware import request_id_context

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT,
    server TEXT,
    tool TEXT NOT NULL,
    args TEXT,
    status TEXT NOT NULL,
    duration_ms INTEGER,
    error TEXT
)
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLogger:
    """异步 SQLite 审计记录器（全局单例 audit_logger）。"""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        self._conn = None

    async def _get_conn(self):
        if self._conn is None:
            import aiosqlite

            db_path = self._db_path or config.harness_audit_db_path
            db_dir = os.path.dirname(os.path.abspath(db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self._conn = await aiosqlite.connect(db_path)
            await self._conn.execute(_SCHEMA)
            await self._conn.commit()
        return self._conn

    async def record(
        self,
        *,
        tool: str,
        server: str = "",
        args: dict | None = None,
        status: str = "success",
        duration_ms: float = 0.0,
        error: str = "",
    ) -> None:
        """写一条审计记录；审计被禁用或任何异常都直接忽略，不影响真实工具调用。"""
        if not config.harness_audit_enabled:
            return
        try:
            conn = await self._get_conn()
            request_id = request_id_context.get()
            await conn.execute(
                "INSERT INTO tool_calls (ts, request_id, server, tool, args, status, duration_ms, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now_iso(),
                    request_id,
                    server,
                    tool,
                    json.dumps(args or {}, ensure_ascii=False, default=str)[:4000],
                    status,
                    int(duration_ms),
                    error[:1000],
                ),
            )
            await conn.commit()
        except Exception as e:
            logger.warning(f"审计记录写入失败（忽略）：{e}")

    async def list_recent(self, limit: int = 50) -> list[dict]:
        """最近 N 条审计记录（按时间倒序），供排查 / 查询。"""
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT id, ts, request_id, server, tool, args, status, duration_ms, error "
            "FROM tool_calls ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        cols = ["id", "ts", "request_id", "server", "tool", "args", "status", "duration_ms", "error"]
        return [dict(zip(cols, row)) for row in rows]

    async def query(
        self,
        limit: int = 100,
        status: str | None = None,
        tool: str | None = None,
        server: str | None = None,
        since: str | None = None,
        before: str | None = None,
    ) -> list[dict]:
        """按条件筛选审计记录（时间倒序）。

        since/before 为 ISO 时间字符串（含时区，如 2026-10-18T08:00:00+00:00）；
        因 ts 统一用 UTC ISO 存储，可直接按字典序比较。
        """
        conn = await self._get_conn()
        sql = (
            "SELECT id, ts, request_id, server, tool, args, status, duration_ms, error "
            "FROM tool_calls"
        )
        where: list[str] = []
        params: list = []
        if status:
            where.append("status = ?")
            params.append(status)
        if tool:
            where.append("tool = ?")
            params.append(tool)
        if server:
            where.append("server = ?")
            params.append(server)
        if since:
            where.append("ts >= ?")
            params.append(since)
        if before:
            where.append("ts <= ?")
            params.append(before)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cur = await conn.execute(sql, params)
        rows = await cur.fetchall()
        cols = ["id", "ts", "request_id", "server", "tool", "args", "status", "duration_ms", "error"]
        return [dict(zip(cols, row)) for row in rows]

    async def status_counts(self) -> dict[str, int]:
        """按 status 分组计数（供监控大盘展示各类状态占比）。"""
        conn = await self._get_conn()
        cur = await conn.execute("SELECT status, COUNT(*) AS n FROM tool_calls GROUP BY status")
        rows = await cur.fetchall()
        return {status: n for status, n in rows}

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


# 全局单例
audit_logger = AuditLogger()
