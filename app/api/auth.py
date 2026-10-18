"""认证与用户管理 API（C2）。

- POST /api/auth/login：用户名+密码 → 签发 token（免鉴权，AuthMiddleware 放行）。
- GET  /api/auth/me：当前登录身份（任意已登录）。
- 用户管理（POST/GET/DELETE /api/auth/users…）：仅 admin。
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import (
    Principal,
    Role,
    create_token,
    require_role,
    user_store,
)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "viewer"


class RoleUpdate(BaseModel):
    role: str


def _validate_role(raw: str) -> Role:
    try:
        return Role(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"非法角色：{raw}（可选 admin/operator/viewer）")


@router.post('/auth/login')
async def login(body: LoginRequest):
    """登录：校验用户名密码，返回一个带过期时间的 token。"""
    principal = await user_store.verify_credentials(body.username, body.password)
    if principal is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_token(principal.username, principal.role.value)
    return {
        'code': 200,
        'message': 'ok',
        'data': {'token': token, 'username': principal.username, 'role': principal.role.value},
    }


@router.get('/auth/me')
async def me(_u: Principal = Depends(require_role(Role.VIEWER, Role.OPERATOR, Role.ADMIN))):
    """返回当前登录身份。"""
    return {'code': 200, 'message': 'ok', 'data': {'username': _u.username, 'role': _u.role.value}}


@router.post('/auth/users')
async def create_user(body: UserCreate, _u: Principal = Depends(require_role(Role.ADMIN))):
    """创建用户（仅 admin）。"""
    role = _validate_role(body.role)
    try:
        user = await user_store.create_user(body.username, body.password, role)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {'code': 200, 'message': 'ok', 'data': {'username': user['username'], 'role': user['role']}}


@router.get('/auth/users')
async def list_users(_u: Principal = Depends(require_role(Role.ADMIN))):
    """列出所有用户（仅 admin）。"""
    return {'code': 200, 'message': 'ok', 'data': await user_store.list_users()}


@router.delete('/auth/users/{username}')
async def delete_user(username: str, _u: Principal = Depends(require_role(Role.ADMIN))):
    """删除用户（仅 admin）。"""
    ok = await user_store.delete_user(username)
    if not ok:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {'code': 200, 'message': '已删除', 'data': {'username': username}}


@router.post('/auth/users/{username}/role')
async def set_role(username: str, body: RoleUpdate, _u: Principal = Depends(require_role(Role.ADMIN))):
    """修改用户角色（仅 admin）。"""
    role = _validate_role(body.role)
    ok = await user_store.set_role(username, role)
    if not ok:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {'code': 200, 'message': 'ok', 'data': {'username': username, 'role': role.value}}
