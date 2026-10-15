"""Harness 工具执行层（Milestone 0 契约 + M1 资源锁 + M2 并发调度）。

提供工具元数据契约（ToolSpec）、元数据注册表（tool_registry）、
结构化审计（audit_logger）、资源锁（resource_locks）与
并发调度（PrioritySemaphore / tool_gate），是后续审批里程碑的公共地基。
"""
from app.harness.tool_spec import ToolSpec, RiskLevel
from app.harness.registry import ToolRegistry, tool_registry
from app.harness.audit import AuditLogger, audit_logger
from app.harness.lock import ResourceLockManager, resource_locks
from app.harness.scheduler import PrioritySemaphore, tool_gate

__all__ = [
    "ToolSpec",
    "RiskLevel",
    "ToolRegistry",
    "tool_registry",
    "AuditLogger",
    "audit_logger",
    "ResourceLockManager",
    "resource_locks",
    "PrioritySemaphore",
    "tool_gate",
]
