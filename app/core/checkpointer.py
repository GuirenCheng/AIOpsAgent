"""检查点存储工厂

为 LangGraph 提供可持久化的 checkpointer，使会话历史在服务重启后依然保留：

- 安装了 ``langgraph-checkpoint-sqlite`` 且 ``session_store_enabled`` 为真时，
  使用 SQLite 落盘（默认路径 data/checkpoints.sqlite）；
- 未安装该依赖、或显式禁用时，回退到内存 MemorySaver（不崩溃，但会话重启即丢）。

这样既不引入硬性运行时依赖，又能在生产环境按需升级为持久化存储。
"""
import os

from loguru import logger


def create_checkpointer():
    """创建 LangGraph checkpointer（SQLite 持久化优先，内存回退）。"""
    from app.config import config

    if not config.session_store_enabled:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning("会话持久化已禁用，使用内存 MemorySaver（重启后会话丢失）")
        return MemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning(
            "未安装 langgraph-checkpoint-sqlite，回退到内存 MemorySaver（重启后会话丢失）。"
            "如需持久化，请执行：pip install langgraph-checkpoint-sqlite"
        )
        return MemorySaver()

    db_path = config.session_store_db_path
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    import sqlite3

    try:
        # 长生命周期服务应使用直接构造：SqliteSaver(conn)，setup() 会在首次读写时自动执行。
        # 注意：langgraph-checkpoint-sqlite 4.2+ 的 from_conn_string() 返回的是上下文管理器
        # （with 退出即关闭连接），不能作为常驻 checkpointer 直接传给 compile()。
        checkpointer = SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
    except (TypeError, AttributeError):
        # 兜底：旧版本 from_conn_string() 直接返回 saver 实例
        checkpointer = SqliteSaver.from_conn_string(db_path)
        from langgraph.checkpoint.base import BaseCheckpointSaver

        if not isinstance(checkpointer, BaseCheckpointSaver):
            checkpointer = checkpointer.__enter__()

    logger.info(f"会话持久化已启用：SQLite checkpointer @ {db_path}")
    return checkpointer
