"""检查点存储工厂

为 LangGraph 提供可持久化的 checkpointer，使会话历史在服务重启后依然保留：

- 安装了 ``langgraph-checkpoint-sqlite`` 与 ``aiosqlite`` 且 ``session_store_enabled`` 为真时，
  使用 SQLite 落盘（默认路径 data/checkpoints.sqlite）；
- 未安装依赖、或显式禁用时，回退到内存 MemorySaver（不崩溃，但会话重启即丢）。

注意：Agent 走异步接口（ainvoke/astream），必须用 AsyncSqliteSaver；
同步的 SqliteSaver 不支持异步方法，会在运行时抛
"The SqliteSaver does not support async methods"。
"""
import os

from loguru import logger


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

    # aiosqlite.connect() 为同步函数，直接返回 Connection 代理（真正连接在首次读写时建立）。
    # AsyncSqliteSaver 的 setup() 会在首次读写（aput/aget 等）时自动执行，无需手动调用。
    # 不要用 from_conn_string()：它是 async 上下文管理器，退出即关闭连接，不能常驻。
    conn = aiosqlite.connect(db_path)
    checkpointer = AsyncSqliteSaver(conn)

    logger.info(f"会话持久化已启用：AsyncSqliteSaver @ {db_path}")
    return checkpointer
