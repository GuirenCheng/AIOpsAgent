'''FastAPI应用入口

主应用入口，配置路由，中间件，静态文件等
'''
import os
import json
from contextlib import asynccontextmanager

from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from app.api import health, chat, file, aiops, approval, webhook, audit, auth, agent
from app.config import config
from app.core.auth import ensure_bootstrap_admin
from app.core.middleware import (
    AuthMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
)
from fastapi import FastAPI
from loguru import logger
from app.core.milvus_client import milvus_manager
from app.harness.approval import approval_manager
from app.harness.lock import resource_locks
from app.harness.rollback import rollback_manager

@asynccontextmanager
async def lifespan(app:FastAPI):
    '''
    应用生命管理周期
    启动时执行
    '''
    logger.info('='*60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境:{'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    if config.enable_docs:
        logger.info(f'📚 API 文档: http://{config.host}:{config.port}/docs')

    # 连接Milvus（失败不阻断启动，降级为「依赖不可用」，由 /health 上报状态）
    logger.info('🔌 正在连接 Milvus...')
    try:
        milvus_manager.connect()
        logger.info('✅ milvus连接成功')
    except Exception as e:
        logger.error(f"⚠️ Milvus 连接失败，服务将降级启动（依赖不可用）：{e}")

    # 启动回滚 supervisor（后台任务消费写失败产生的回滚信号）
    rollback_manager.start()

    # 首次启动创建初始管理员（RBAC；api_auth_enabled=True 且用户表为空时）
    await ensure_bootstrap_admin()

    # 多 worker 安全告警（C1）：进程内锁/审批只在单进程内有效，多 worker 时会被打破。
    # 提示切 harness_backend=redis，否则写操作互斥/审批跨 worker 失效。
    if config.workers > 1 and config.harness_backend != 'redis':
        logger.warning(
            f"⚠️ 检测到 WORKERS={config.workers}（多进程）但 HARNESS_BACKEND={config.harness_backend}，"
            f"进程内资源锁/审批会跨 worker 失效（并发写同一资源可能被打破互斥）。"
            f"多 worker 部署请设置 HARNESS_BACKEND=redis 并确保 Redis 可用。"
        )

    logger.info('=' * 60)

    yield

    # 关闭时执行
    await rollback_manager.stop()
    await resource_locks.aclose()
    await approval_manager.aclose()
    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")


app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description='基于LangChain的智能oncall运维系统',
    # 生产环境关闭 API 文档（/docs /redoc /openapi.json），减少信息暴露
    docs_url='/docs' if config.enable_docs else None,
    redoc_url='/redoc' if config.enable_docs else None,
    openapi_url='/openapi.json' if config.enable_docs else None,
    lifespan = lifespan
)

# 生产加固中间件（add_middleware 会前插，故按「内→外」顺序添加）
# 最终执行顺序（外→内）：RequestID → [CORS] → RateLimit → Auth → 路由
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# CORS：仅在显式配置允许来源时启用；默认同源访问无需 CORS。
# 之前的 allow_origins=["*"] + allow_credentials=True 是无效组合（浏览器会拒绝），
# 这里改为配置驱动，且默认不开启凭据。
if config.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(RequestIDMiddleware)

# 注册路由
app.include_router(health.router,tags = ['健康检查'])
app.include_router(chat.router,prefix='/api',tags=['对话'])
app.include_router(file.router,prefix='/api',tags = ['文件管理'])
app.include_router(aiops.router,prefix = '/api',tags = ['AIOps智能运维'])
app.include_router(approval.router,prefix = '/api',tags = ['Harness 人工审批'])
app.include_router(webhook.router,prefix = '/api',tags = ['告警 Webhook'])
app.include_router(agent.router,prefix = '/api',tags = ['统一入口'])
app.include_router(audit.router,prefix = '/api',tags = ['审计与监控'])
app.include_router(auth.router,prefix = '/api',tags = ['认证与用户'])

# 挂载静态文件
base_dir = os.path.dirname(os.path.abspath(__file__))  # 获取 app 目录
project_root = os.path.dirname(base_dir)               # 获取项目根目录（AIOpsAgent）
static_dir = os.path.join(project_root, 'static')
app.mount('/static', StaticFiles(directory=static_dir), name='static')

@app.get('/')
async def root():
    '''返回首页，并向前端注入 API 鉴权 key（密钥不落盘到仓库）'''
    index_path = os.path.join(static_dir,'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            html = f.read()
        # 前端 app.js 从 window.API_AUTH_KEY 读取；由服务端从 .env 注入，仓库中无明文密钥
        inject = '<script>window.API_AUTH_KEY = ' + json.dumps(config.api_auth_key) + ';</script>'
        html = html.replace('</head>', inject + '</head>', 1)
        return HTMLResponse(html)
    return {
        'message':f"Welcome to {config.app_name} API",
        'version':config.app_version,
        'docs':'/docs'
    }

@app.get('/login', include_in_schema=False)
async def login_page():
    '''登录页（RBAC）。无需注入 API key，登录接口本身免鉴权。'''
    page_path = os.path.join(static_dir, 'login.html')
    if not os.path.exists(page_path):
        return HTMLResponse('<h1>登录页面缺失</h1>', status_code=404)
    with open(page_path, 'r', encoding='utf-8') as f:
        html = f.read()
    return HTMLResponse(html)

@app.get('/audit', include_in_schema=False)
async def audit_page():
    '''审计大盘页面，服务端注入 API 鉴权 key（密钥不落盘到仓库）'''
    page_path = os.path.join(static_dir, 'audit.html')
    if not os.path.exists(page_path):
        return HTMLResponse('<h1>审计页面缺失</h1>', status_code=404)
    with open(page_path, 'r', encoding='utf-8') as f:
        html = f.read()
    inject = '<script>window.API_AUTH_KEY = ' + json.dumps(config.api_auth_key) + ';</script>'
    html = html.replace('</head>', inject + '</head>', 1)
    return HTMLResponse(html)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'app.main:app',
        host = config.host,
        port = config.port,
        reload = config.debug,
        log_level = 'info'
    )
