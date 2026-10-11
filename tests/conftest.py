"""pytest 全局配置：确保项目根目录在 sys.path 中，使 `app` 包可导入。"""
import os
import sys
from pathlib import Path

# 必须在任何 app 导入之前设置：禁用 SQLite checkpointer，
# 让测试使用内存 MemorySaver，避免写 data/checkpoints.sqlite
os.environ["SESSION_STORE_ENABLED"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
