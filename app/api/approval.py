"""高危操作人工审批 API（Milestone 3）。

高危工具（如 restart_service）在真正执行前会挂起等待人工决定，
运维人员通过这些接口查看待审批单并批准 / 拒绝。

B2 一键审批：通知里带一个「一键审批」链接（短时效签名 token），免登录打开确认页，
点「批准/拒绝」才真正 POST 执行。安全靠 token 签名 + 短时效，不依赖登录态；
token 绑定审批单号，签名/过期/用途不符一律拒绝（fail-closed）。
"""
import html
import json

from fastapi import APIRouter, Depends, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.auth import Principal, Role, decode_approval_token, require_role
from app.harness.approval import approval_manager

router = APIRouter()

# 一键审批操作人标记（免登录链接无法知道点击者身份，审计里统一记为该来源）
_APPROVER_VIA_LINK = "approval-link"


class Decision(BaseModel):
    reason: str = ""


def _esc(s) -> str:
    return html.escape(str(s))


_DECISION_PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>一键审批</title></head>
<body style="font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px">
<h2>⚠️ 高危操作审批</h2>
<p><b>审批单号：</b>{approval_id}</p>
<p><b>工具：</b>{tool}</p>
<p><b>服务器：</b>{server}</p>
<p><b>参数：</b>{args}</p>
<p>批准将放行该高危操作；拒绝则不执行（fail-closed）。</p>
<form method="post" action="/api/harness/approvals/{approval_id}/decision" style="display:inline">
  <input type="hidden" name="token" value="{token}">
  <input type="hidden" name="action" value="approve">
  <button type="submit" style="padding:10px 20px;background:#2e7d32;color:#fff;border:none;border-radius:4px;cursor:pointer">✅ 批准</button>
</form>
<form method="post" action="/api/harness/approvals/{approval_id}/decision" style="display:inline;margin-left:12px">
  <input type="hidden" name="token" value="{token}">
  <input type="hidden" name="action" value="deny">
  <button type="submit" style="padding:10px 20px;background:#c62828;color:#fff;border:none;border-radius:4px;cursor:pointer">❌ 拒绝</button>
</form>
</body></html>"""

_RESULT_PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>审批结果</title></head>
<body style="font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px">
<h2>{title}</h2>
<p>{message}</p>
</body></html>"""


@router.get('/harness/approvals')
async def list_pending():
    """列出所有待审批的高危操作（任意已登录用户可看）。"""
    return {'code': 200, 'message': 'ok', 'data': await approval_manager.list_pending()}


@router.post('/harness/approvals/{approval_id}/approve')
async def approve(
    approval_id: str,
    body: Decision = None,
    _u: Principal = Depends(require_role(Role.OPERATOR, Role.ADMIN)),
):
    """批准指定审批单，放行对应的高危操作（需 operator/admin）。"""
    ok = await approval_manager.approve(approval_id, body.reason if body else "", approver=_u.username)
    return {
        'code': 200 if ok else 404,
        'message': '已批准' if ok else '审批单不存在或已处理',
        'data': {'id': approval_id, 'approved': ok},
    }


@router.post('/harness/approvals/{approval_id}/deny')
async def deny(
    approval_id: str,
    body: Decision = None,
    _u: Principal = Depends(require_role(Role.OPERATOR, Role.ADMIN)),
):
    """拒绝指定审批单，对应的高危操作不会执行（需 operator/admin）。"""
    ok = await approval_manager.deny(approval_id, body.reason if body else "", approver=_u.username)
    return {
        'code': 200 if ok else 404,
        'message': '已拒绝' if ok else '审批单不存在或已处理',
        'data': {'id': approval_id, 'denied': ok},
    }


@router.get('/harness/approvals/{approval_id}/decision', include_in_schema=False)
async def decision_page(approval_id: str, token: str = ""):
    """一键审批确认页（免登录；GET 只渲染、不改变审批状态，防链接预拉取误触发）。"""
    if not token or decode_approval_token(token) != approval_id:
        return HTMLResponse(
            _RESULT_PAGE.format(title="链接无效", message="链接已失效、过期或签名不正确，请重新获取。"),
            status_code=400,
        )
    pending = await approval_manager.list_pending()
    appr = next((p for p in pending if p["id"] == approval_id), None)
    if appr is None:
        return HTMLResponse(
            _RESULT_PAGE.format(title="已处理", message="该审批单不存在或已处理（可能已被处理/超时），无需重复操作。"),
        )
    args_str = json.dumps(appr.get("args"), ensure_ascii=False, default=str) if appr.get("args") is not None else "-"
    body = _DECISION_PAGE.format(
        approval_id=_esc(approval_id),
        tool=_esc(appr.get("tool", "-")),
        server=_esc(appr.get("server") or "-"),
        args=_esc(args_str),
        token=_esc(token),
    )
    return HTMLResponse(body)


@router.post('/harness/approvals/{approval_id}/decision', include_in_schema=False)
async def decide(
    approval_id: str,
    action: str = Form(...),
    token: str = Form(...),
):
    """执行一键审批（免登录；安全靠 token 签名 + 短时效，签名/过期/单号不符一律拒绝）。"""
    if decode_approval_token(token) != approval_id:
        return HTMLResponse(
            _RESULT_PAGE.format(title="链接无效", message="链接已失效、过期或签名不正确，请重新获取。"),
            status_code=400,
        )
    if action == "approve":
        ok = await approval_manager.approve(approval_id, approver=_APPROVER_VIA_LINK)
        word = "批准"
    elif action == "deny":
        ok = await approval_manager.deny(approval_id, approver=_APPROVER_VIA_LINK)
        word = "拒绝"
    else:
        return HTMLResponse(
            _RESULT_PAGE.format(title="无效操作", message="未知的审批动作。"),
            status_code=400,
        )
    if ok:
        return HTMLResponse(_RESULT_PAGE.format(title="已完成", message=f"审批单 {approval_id} 已{word}。"))
    return HTMLResponse(
        _RESULT_PAGE.format(
            title="已处理",
            message=f"审批单 {approval_id} 不存在或已处理（可能已被其他人处理/超时）。",
        ),
    )
