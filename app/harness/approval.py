"""高危操作人工审批（Milestone 3）：高危写操作执行前需人工批准。

高危工具（spec.risk 达到阈值，默认 HIGH）在 retry_interceptor 最外层先创建审批单、
等待人工 approve/deny，批准才放行、拒绝/超时 fail-closed 不执行。

单进程内存存储 + asyncio.Event（全局单例 approval_manager）：审批单随进程重启丢失，
与进程内资源锁同理。审批生命周期（pending_approval → approval_approved/denied/expired）
全部写入 audit_logger 留痕，内存只负责「等待决定」的同步原语。
多进程化时需换 Redis/DB + 通知通道，request/approve/deny/wait_for_decision 接口不变。
"""
import asyncio
import itertools
from datetime import datetime, timezone
from enum import Enum

from app.core.middleware import request_id_context
from app.harness.audit import audit_logger
from app.harness.tool_spec import RiskLevel


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


# 风险等级排序：数值越大越危险。审批阈值取「本级及以上」。
_RISK_RANK = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


def requires_approval(risk: RiskLevel, threshold: str) -> bool:
    """风险等级达到阈值（含本级及以上）时需要人工审批。非法阈值回退 HIGH（最保守）。"""
    try:
        t = RiskLevel(threshold)
    except ValueError:
        t = RiskLevel.HIGH
    return _RISK_RANK[risk] >= _RISK_RANK[t]


class _Approval:
    def __init__(self, approval_id: str, tool: str, args, server: str, request_id: str):
        self.id = approval_id
        self.tool = tool
        self.args = args
        self.server = server
        self.request_id = request_id
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = ApprovalStatus.PENDING
        self.reason = ""
        self.event = asyncio.Event()


class ApprovalManager:
    """高危操作人工审批管理器（进程内，全局单例 approval_manager）。"""

    def __init__(self) -> None:
        self._pending: dict[str, _Approval] = {}
        self._ids = itertools.count(1)

    async def request(self, tool: str, args=None, server: str = "") -> str:
        """创建审批单并返回审批单号；同时写一条 pending_approval 审计。"""
        approval_id = f"appr-{next(self._ids)}"
        self._pending[approval_id] = _Approval(
            approval_id, tool, args, server, request_id_context.get()
        )
        await audit_logger.record(
            tool=tool, server=server, args=args,
            status="pending_approval", error=approval_id,
        )
        return approval_id

    async def _decide(self, approval_id: str, status: ApprovalStatus, reason: str = "") -> bool:
        appr = self._pending.get(approval_id)
        if appr is None or appr.status != ApprovalStatus.PENDING:
            return False
        appr.status = status
        appr.reason = reason
        appr.event.set()
        await audit_logger.record(
            tool=appr.tool, server=appr.server, args=appr.args,
            status=f"approval_{status.value}", error=reason or approval_id,
        )
        return True

    async def approve(self, approval_id: str, reason: str = "") -> bool:
        """批准某审批单。返回是否成功（单不存在或已处理返回 False）。"""
        return await self._decide(approval_id, ApprovalStatus.APPROVED, reason)

    async def deny(self, approval_id: str, reason: str = "") -> bool:
        """拒绝某审批单。返回是否成功（单不存在或已处理返回 False）。"""
        return await self._decide(approval_id, ApprovalStatus.DENIED, reason)

    async def wait_for_decision(self, approval_id: str, timeout: float = 300.0) -> ApprovalStatus:
        """等待审批决定；超时或单不存在均视为「过期/拒绝」（fail-closed）。"""
        appr = self._pending.get(approval_id)
        if appr is None:
            return ApprovalStatus.EXPIRED
        if appr.status != ApprovalStatus.PENDING:
            return appr.status
        try:
            await asyncio.wait_for(appr.event.wait(), timeout=timeout)
            return appr.status
        except asyncio.TimeoutError:
            appr.status = ApprovalStatus.EXPIRED
            await audit_logger.record(
                tool=appr.tool, server=appr.server, args=appr.args,
                status="approval_expired", error=approval_id,
            )
            return ApprovalStatus.EXPIRED

    async def list_pending(self) -> list[dict]:
        """列出所有待审批单（供 API / 控制台展示）。"""
        return [
            {
                "id": a.id,
                "tool": a.tool,
                "args": a.args,
                "server": a.server,
                "request_id": a.request_id,
                "created_at": a.created_at,
                "status": a.status.value,
                "reason": a.reason,
            }
            for a in self._pending.values()
            if a.status == ApprovalStatus.PENDING
        ]


# 全局单例（与 tool_registry / audit_logger / resource_locks / tool_gate 同风格）
approval_manager = ApprovalManager()
