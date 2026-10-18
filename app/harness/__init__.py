"""Harness 工具执行层（M0 契约 + M1 资源锁 + M2 并发调度 + M3 人工审批）。

提供工具元数据契约（ToolSpec）、元数据注册表（tool_registry）、
结构化审计（audit_logger）、资源锁（resource_locks）、并发调度
（PrioritySemaphore / tool_gate）与人工审批（ApprovalManager / approval_manager）。
"""
from app.harness.tool_spec import ToolSpec, RiskLevel
from app.harness.registry import ToolRegistry, tool_registry
from app.harness.audit import AuditLogger, audit_logger
from app.harness.lock import ResourceLockManager, resource_locks
from app.harness.scheduler import PrioritySemaphore, tool_gate
from app.harness.approval import ApprovalManager, ApprovalStatus, approval_manager, requires_approval
from app.harness.notify import (
    Notifier,
    notifier,
    notify_alert_diagnosis,
    notify_approval,
    notify_approval_timeout,
    notify_rollback_failure,
)

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
    "ApprovalManager",
    "ApprovalStatus",
    "approval_manager",
    "requires_approval",
    "Notifier",
    "notifier",
    "notify_approval",
    "notify_alert_diagnosis",
    "notify_approval_timeout",
    "notify_rollback_failure",
]
