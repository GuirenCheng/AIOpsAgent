"""工具契约元数据：描述每个工具在 harness 中如何被调度、重试与审计。

这是工具执行层（Milestone 1 资源锁 / Milestone 2 调度 / Milestone 3 审批）的公共地基。
"""
from enum import Enum

from pydantic import BaseModel


class RiskLevel(str, Enum):
    """操作风险等级。"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolSpec(BaseModel):
    """单个工具的契约元数据。

    - read_only: 是否只读（写操作需锁 / 审批，见后续里程碑）
    - idempotent: 幂等才允许自动重试；非幂等写操作重试可能导致重复执行
    - target_resource: 未来 Milestone 1 资源锁的锁对象（写工具填服务名等）
    - risk: 操作风险等级
    - has_rollback: 是否有可执行的回滚手段
    - timeout_seconds: 该工具调用的超时上限（秒）
    """
    name: str
    read_only: bool = True
    idempotent: bool = True
    target_resource: str | None = None
    risk: RiskLevel = RiskLevel.LOW
    has_rollback: bool = False
    timeout_seconds: float = 30.0
