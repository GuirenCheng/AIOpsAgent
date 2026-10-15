"""Harness 工具执行层（Milestone 0 契约 + Milestone 1 资源锁）。

提供工具元数据契约（ToolSpec）、元数据注册表（tool_registry）、
结构化审计（audit_logger）与资源锁（resource_locks），
是后续调度 / 审批等里程碑的公共地基。
"""
from app.harness.tool_spec import ToolSpec, RiskLevel
from app.harness.registry import ToolRegistry, tool_registry
from app.harness.audit import AuditLogger, audit_logger
from app.harness.lock import ResourceLockManager, resource_locks

__all__ = [
    "ToolSpec",
    "RiskLevel",
    "ToolRegistry",
    "tool_registry",
    "AuditLogger",
    "audit_logger",
    "ResourceLockManager",
    "resource_locks",
]
