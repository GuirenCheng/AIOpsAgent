"""高危操作人工审批 API（Milestone 3）。

高危工具（如 restart_service）在真正执行前会挂起等待人工决定，
运维人员通过这些接口查看待审批单并批准 / 拒绝。
"""
from fastapi import APIRouter
from pydantic import BaseModel

from app.harness.approval import approval_manager

router = APIRouter()


class Decision(BaseModel):
    reason: str = ""


@router.get('/harness/approvals')
async def list_pending():
    """列出所有待审批的高危操作。"""
    return {'code': 200, 'message': 'ok', 'data': await approval_manager.list_pending()}


@router.post('/harness/approvals/{approval_id}/approve')
async def approve(approval_id: str, body: Decision = None):
    """批准指定审批单，放行对应的高危操作。"""
    ok = await approval_manager.approve(approval_id, body.reason if body else "")
    return {
        'code': 200 if ok else 404,
        'message': '已批准' if ok else '审批单不存在或已处理',
        'data': {'id': approval_id, 'approved': ok},
    }


@router.post('/harness/approvals/{approval_id}/deny')
async def deny(approval_id: str, body: Decision = None):
    """拒绝指定审批单，对应的高危操作不会执行。"""
    ok = await approval_manager.deny(approval_id, body.reason if body else "")
    return {
        'code': 200 if ok else 404,
        'message': '已拒绝' if ok else '审批单不存在或已处理',
        'data': {'id': approval_id, 'denied': ok},
    }
