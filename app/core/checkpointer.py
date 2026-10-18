"""检查点存储工厂

为 LangGraph 提供可持久化的 checkpointer，使会话历史在服务重启后依然保留：

- 安装了 ``langgraph-checkpoint-sqlite`` 与 ``aiosqlite`` 且 ``session_store_enabled`` 为真时，
  使用 SQLite 落盘（默认路径 data/checkpoints.sqlite）；
- 未安装依赖、或显式禁用时，回退到内存 MemorySaver（不崩溃，但会话重启即丢）。

多 worker（uvicorn --workers N）部署下，每个 worker 进程各自持有到同一个
checkpoints.sqlite 的连接并并发写会话：这里把数据库设为 WAL + busy_timeout，
避免并发写会话时抛 SQLITE_BUSY / 写冲突（与 audit.sqlite、users.sqlite 同策略）。

注意：Agent 走异步接口（ainvoke/astream），必须用 AsyncSqliteSaver；
同步的 SqliteSaver 不支持异步方法，会在运行时抛
"The SqliteSaver does not support async methods"。
"""
import os

from loguru import logger


def _apply_multiprocess_pragmas(db_path: str) -> None:
    """多 worker 下多个进程会并发写同一个 checkpoints.sqlite，做两步加固：

    - journal_mode=WAL 是文件级持久属性（设置一次即对所有连接生效），让读写并发、
      降低 writer 竞争；这里用一次性同步连接设置。
    - busy_timeout 属连接级，在 aiosqlite.connect(..., timeout=5.0) 里指定（sqlite3 的
      timeout 参数即 busy_timeout，并发写时等待重试而非立刻 SQLITE_BUSY）。

    设置失败只告警、不阻断：连接仍可写，最坏退化为无 WAL 的单写者模式。
    """
    import sqlite3

    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"设置 checkpoints.sqlite 为 WAL 失败（忽略）：{e}")


def create_checkpointer():
    """创建 LangGraph checkpointer（SQLite 持久化优先，内存回退）。

    返回 AsyncSqliteSaver（同时具备异步与同步接口）。Agent 用 ainvoke/astream
    走异步读写；会话查询/清空用 aget/adelete_thread。AsyncSqliteSaver 的同步方法
    （get/delete_thread）在事件循环线程会抛 InvalidStateError，故业务代码须统一走异步。
    """
    from app.config import config

    if not config.session_store_enabled:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning("会话持久化已禁用，使用内存 MemorySaver（重启后会话丢失）")
        return MemorySaver()

    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning(
            "未安装 langgraph-checkpoint-sqlite / aiosqlite，回退到内存 MemorySaver（重启后会话丢失）。"
            "如需持久化，请执行：pip install langgraph-checkpoint-sqlite aiosqlite"
        )
        return MemorySaver()

    db_path = config.session_store_db_path
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # 多进程写安全：WAL（文件级、持久）+ busy_timeout（连接级，timeout=5.0 → 5000ms）
    _apply_multiprocess_pragmas(db_path)

    # aiosqlite.connect() 为同步函数，直接返回 Connection 代理（真正连接在首次读写时建立）。
    # AsyncSqliteSaver 的 setup() 会在首次读写（aput/aget 等）时自动执行，无需手动调用。
    # 不要用 from_conn_string()：它是 async 上下文管理器，退出即关闭连接，不能常驻。
    conn = aiosqlite.connect(db_path, timeout=5.0)
    checkpointer = AsyncSqliteSaver(conn)

    logger.info(f"会话持久化已启用：AsyncSqliteSaver @ {db_path}")
    return checkpointer
