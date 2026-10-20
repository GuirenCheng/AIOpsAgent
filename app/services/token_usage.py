"""Token 用量记账：统计并落库每次 DashScope 调用（chat / rerank / embedding）的 token 消耗。

把「隐性省钱」变成「显性管钱」：每次模型调用后把 input/output/total tokens 写入
独立 SQLite（data/token_usage.sqlite），可按模型 / 来源 / 天 / 会话 / 用户聚合，
并支持每日预算告警（超额只告警，不阻断，避免预算熔断影响可用性）。

实现要点：
- LangChain 的 on_llm_end 回调是**同步**的，且 chat/rerank/embedding 的调用点分布在
  同步与异步两类上下文里。因此这里用一个同步 `record()`（只做 queue.put，不阻塞调用方）
  + 后台写线程，任何上下文都能零开销记账，也避免 aiosqlite 在同步回调里无法 await 的问题。
- 写线程单连接 + WAL + busy_timeout，读（stats）另开只读连接，多 worker 并发安全。
"""
import os
import queue
import sqlite3
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from loguru import logger

from app.config import config
from app.core.auth import get_actor
from app.core.middleware import request_id_context

_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    session_id TEXT,
    request_id TEXT,
    actor TEXT
)
"""

# 攒多少条刷一次盘（减少 SQLite 提交次数；低流量下 1 条也会在 stop 时兜底提交）
_BATCH_COMMIT = 50

# 当前会话 id（RAG 多轮对话在 query/query_stream 前设置，供记账归集）
session_id_context: ContextVar[str] = ContextVar("token_usage_session_id", default="")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TokenUsageRecorder:
    """线程安全的 Token 用量记录器（后台写线程 + 无界队列）。"""

    def __init__(self) -> None:
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        # 预算告警节流：上次打告警的单调时间戳（避免超额后每条都刷日志）
        self._last_budget_warn = 0.0
        self._budget_warn_lock = threading.Lock()

    # ------------------------------------------------------------------ 写入

    def _ensure_started(self) -> None:
        if self._thread is not None:
            return
        with self._start_lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._worker, daemon=True, name="token-usage-writer"
            )
            self._thread.start()

    def _worker(self) -> None:
        conn: sqlite3.Connection | None = None
        pending = 0
        try:
            db_path = config.token_usage_db_path
            db_dir = os.path.dirname(os.path.abspath(db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            conn = sqlite3.connect(db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(_SCHEMA)
            conn.commit()
        except Exception as e:  # noqa: BLE001 —— 记账库不可用时静默停用，不影响主流程
            logger.warning(f"Token 用量库初始化失败，记账停用：{e}")
            return

        while True:
            item = self._q.get()
            if item is None:  # stop() 的停止信号
                break
            try:
                conn.execute(
                    "INSERT INTO token_usage "
                    "(ts, source, model, input_tokens, output_tokens, total_tokens, session_id, request_id, actor) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    item,
                )
                pending += 1
                if pending >= _BATCH_COMMIT:
                    conn.commit()
                    pending = 0
                    self._maybe_warn_budget(conn)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Token 用量写入失败（忽略）：{e}")

        if conn is not None:
            try:
                if pending:
                    conn.commit()
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def record(
        self,
        *,
        source: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        session_id: str = "",
    ) -> None:
        """记一笔 token 用量。禁用 / 总量为 0 时直接跳过；request_id/actor 从请求上下文自动取。"""
        if not config.token_usage_enabled:
            return
        if total_tokens <= 0:
            return
        self._ensure_started()
        session_id = session_id or session_id_context.get()
        try:
            request_id = request_id_context.get()
        except Exception:  # noqa: BLE001
            request_id = ""
        try:
            actor = get_actor()
        except Exception:  # noqa: BLE001
            actor = ""
        self._q.put(
            (
                _now_iso(),
                source,
                model,
                int(input_tokens),
                int(output_tokens),
                int(total_tokens),
                session_id,
                request_id,
                actor,
            )
        )

    def stop(self) -> None:
        """停写线程并刷盘（进程退出 / 测试收尾用）。"""
        if self._thread is None:
            return
        self._q.put(None)
        self._thread.join(timeout=5)
        self._thread = None

    # ------------------------------------------------------------------ 读取

    def _read_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(config.token_usage_db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def stats(self) -> dict:
        """聚合统计：总量 + 按模型/来源/近 7 天分组。库不存在或为空时返回零值。"""
        empty = {
            "total": {"records": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "by_model": [],
            "by_source": [],
            "by_day": [],
        }
        try:
            conn = self._read_conn()
        except Exception:  # noqa: BLE001
            return empty
        try:
            def _rows(sql, params=()):
                try:
                    return [dict(r) for r in conn.execute(sql, params).fetchall()]
                except sqlite3.Error:
                    return []

            total = _rows(
                "SELECT COUNT(*) AS records, "
                "COALESCE(SUM(input_tokens),0) AS input_tokens, "
                "COALESCE(SUM(output_tokens),0) AS output_tokens, "
                "COALESCE(SUM(total_tokens),0) AS total_tokens FROM token_usage"
            )
            return {
                "total": total[0] if total else empty["total"],
                "by_model": _rows(
                    "SELECT model, COUNT(*) AS calls, SUM(input_tokens) AS input_tokens, "
                    "SUM(output_tokens) AS output_tokens, SUM(total_tokens) AS total_tokens "
                    "FROM token_usage GROUP BY model ORDER BY total_tokens DESC"
                ),
                "by_source": _rows(
                    "SELECT source, COUNT(*) AS calls, SUM(total_tokens) AS total_tokens "
                    "FROM token_usage GROUP BY source ORDER BY total_tokens DESC"
                ),
                "by_day": _rows(
                    "SELECT date(ts) AS day, SUM(total_tokens) AS total_tokens, COUNT(*) AS calls "
                    "FROM token_usage GROUP BY date(ts) ORDER BY day DESC LIMIT 7"
                ),
            }
        finally:
            conn.close()

    def today_total_tokens(self) -> int:
        """今日（UTC）累计 total tokens；库不可用时返回 0。"""
        try:
            conn = self._read_conn()
        except Exception:  # noqa: BLE001
            return 0
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(total_tokens),0) FROM token_usage WHERE date(ts) = date('now')"
            ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0
        finally:
            conn.close()

    def _maybe_warn_budget(self, conn: sqlite3.Connection) -> None:
        """当日总量超过预算时打告警（节流：最多每 300 秒一次）。"""
        if not config.token_budget_enabled or config.token_budget_daily_limit <= 0:
            return
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(total_tokens),0) FROM token_usage WHERE date(ts) = date('now')"
            ).fetchone()
            used = int(row[0]) if row else 0
        except sqlite3.Error:
            return
        limit = config.token_budget_daily_limit
        if used <= limit:
            return
        with self._budget_warn_lock:
            now = time.monotonic()
            if now - self._last_budget_warn < 300:
                return
            self._last_budget_warn = now
        logger.warning(
            f"⚠️ 今日 Token 用量 {used} 已超过预算上限 {limit}（超出 {used - limit}）。"
            f"仅告警不阻断，请关注成本或调整 token_budget_daily_limit。"
        )

    def check_daily_budget(self) -> dict:
        """返回今日预算状态（供 CLI / 查询）；不满足告警条件时不写日志。"""
        used = self.today_total_tokens()
        limit = config.token_budget_daily_limit if config.token_budget_enabled else 0
        return {
            "enabled": config.token_budget_enabled,
            "today_tokens": used,
            "limit": limit,
            "exceeded": config.token_budget_enabled and limit > 0 and used > limit,
        }


def _extract_usage(response: Any) -> tuple[int, int, int, str]:
    """从 LangChain LLMResult 里抠出 (input_tokens, output_tokens, total_tokens, model)。"""
    input_tokens = output_tokens = total_tokens = 0
    model = ""
    try:
        llm_output = getattr(response, "llm_output", None) or {}
        model = llm_output.get("model_name", "") or ""
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
    except Exception:  # noqa: BLE001
        pass

    if total_tokens <= 0:
        # 新版本 LangChain 可能把 usage 放在 usage_metadata（OpenAI 兼容）里
        try:
            msg = getattr(response.generations[0][0], "message", None)
            um = getattr(msg, "usage_metadata", None)
            if um is not None:
                input_tokens = int(getattr(um, "input_tokens", 0) or 0)
                output_tokens = int(getattr(um, "output_tokens", 0) or 0)
                total_tokens = int(getattr(um, "total_tokens", 0) or 0)
        except Exception:  # noqa: BLE001
            pass

    return input_tokens, output_tokens, total_tokens, model


class TokenUsageCallback(BaseCallbackHandler):
    """挂到每个 ChatQwen 上的用量采集回调：on_llm_end 里记一笔 chat 用量。"""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if not config.token_usage_enabled:
            return
        try:
            input_tokens, output_tokens, total_tokens, model = _extract_usage(response)
            if total_tokens <= 0:
                return
            token_usage_recorder.record(
                source="chat",
                model=model or config.rag_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        except Exception as e:  # noqa: BLE001 —— 采集失败绝不影响模型调用
            logger.debug(f"token 用量采集失败（忽略）：{e}")


# 全局单例
token_usage_recorder = TokenUsageRecorder()
token_usage_callback = TokenUsageCallback()
