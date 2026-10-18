from typing import Dict,Any
from pydantic_settings import BaseSettings,SettingsConfigDict
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# 本地服务（MCP server 等）不走系统代理：
# httpx 默认 trust_env=True，会读取 Windows 注册表里的系统代理（如 Clash/V2Ray），
# 导致对 localhost 的 SSE/HTTP 连接被代理缓冲破坏而挂起（ReadTimeout）。
# 这里把本地回环地址加入 NO_PROXY，确保直连。
for _proxy_var in ("NO_PROXY", "no_proxy"):
    _entries = {e.strip() for e in os.environ.get(_proxy_var, "").split(",") if e.strip()}
    _entries.update({"127.0.0.1", "localhost", "::1"})
    os.environ[_proxy_var] = ",".join(sorted(_entries))

# Windows 控制台默认 GBK（cp936）。loguru 的控制台 sink 写到 sys.stdout（见 app/utils/logger.py），
# 而非 tty 环境下 stdout 的 errors='surrogateescape'、stderr 为 'backslashreplace'：
# 前者遇到 emoji 会抛 UnicodeEncodeError，导致日志刷屏报错、掩盖真实异常。
# 这里把两者都重配为「替换不可编码字符」：中文仍正常显示，emoji 降级为占位符，不再抛异常。
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

# 版本号从安装包元数据读取，与 pyproject.toml 保持一致（避免两处漂移）。
# 未安装（直接源码运行）时回退到字面版本号。
try:
    from importlib.metadata import version as _pkg_version

    _APP_VERSION = _pkg_version("aiops-agent-py")
except Exception:
    _APP_VERSION = "1.2.1"

class Settings(BaseSettings):
    '''应用配置'''

    model_config = SettingsConfigDict(
        env_file = '.env',
        env_file_encoding = 'utf-8',
        case_sensitive =False,
        extra = 'ignore',
    )

    # 应用配置
    app_name:str = 'AIOpsAgent'
    app_version:str = _APP_VERSION
    debug:bool = False
    host:str = '0.0.0.0'
    port:int = 9900

    # DashScope 配置
    dashscope_api_key:str = os.getenv("DASHSCOPE_API_KEY")# 默认空字符串，实际使用需从环境变量中加载
    # langchain_qwq 的 ChatQwen 默认 base_url 是国际站(dashscope-intl)，国内 key 会报 401；
    # 这里固定国内站，避免依赖 DASHSCOPE_API_BASE 环境变量是否注入。
    dashscope_api_base:str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = 'qwen-max'
    dashscope_embedding_model:str = 'text-embedding-v4'# v4支持多种维度

    # Milvus 配置
    milvus_host:str = 'localhost'
    milvus_port:int = 19530
    milvus_timeout:int = 10000 # 毫秒
    # 向量维度唯一来源（embedding 与 collection schema 共用，避免漂移）
    milvus_vector_dim:int = 1024
    # 维度不匹配时是否允许自动删除并重建 collection（会丢失旧数据，默认禁止）
    milvus_allow_dimension_migration:bool = False

    # RAG 配置
    rag_top_k:int = 3
    rag_model:str = 'qwen-max' # 使用快速响应模型，不带拓展思考
    # RAG 防幻觉配置
    rag_temperature:float = 0.2          # 事实型 RAG 用低温度，降低自由发挥
    rag_relevance_threshold:float = 0.5  # 检索相关性阈值(0~1)，低于此值视为不可靠、不喂给模型
    rag_history_max_messages:int = 7     # 对话历史最多保留条数(含系统消息)，超出则自动裁剪
    # 模型调用韧性（防止偶发失败导致模型拿到残缺上下文乱答）
    rag_model_retry:int = 2              # 模型调用失败重试次数
    rag_fallback_model:str = ''          # 降级模型名（空 = 不启用；如 qwen-plus），主模型反复失败时切换
    rag_max_tool_calls:int = 8           # 单次对话最多工具调用次数（防死循环）

    # 安全 / 生产加固配置
    # 请求鉴权（默认关闭；生产环境建议开启并设置 API_AUTH_KEY）
    api_auth_enabled:bool = False
    api_auth_key:str = ''
    # RBAC（C2）：用户名+密码登录 → 签发无状态 HMAC token
    # auth_token_secret 为 token 签名密钥（api_auth_enabled=true 时必填，否则登录/校验都失败）
    auth_token_secret:str = ''
    auth_token_ttl_seconds:int = 86400  # token 有效期（秒），默认 24 小时
    auth_admin_username:str = 'admin'   # 首次启动自动创建的初始管理员用户名
    auth_admin_password:str = ''        # 初始管理员密码（空则随机生成并打印到日志一次）
    auth_db_path:str = 'data/users.sqlite'  # 用户库（独立于审计库）
    # 单 IP 限流（内存滑动窗口，默认开启，限流阈值可调）
    rate_limit_enabled:bool = True
    rate_limit_per_minute:int = 120
    # CORS 允许来源（逗号分隔；留空则仅同源访问，不发送 CORS 头）
    cors_allow_origins:str = ''
    cors_allow_credentials:bool = False
    # 单个问题最大长度（防滥用）
    max_question_length:int = 4000
    # API 文档（/docs /redoc /openapi.json）是否对外暴露；生产建议关闭以减少信息泄露
    enable_docs:bool = True
    # 会话持久化（LangGraph checkpointer：SQLite 落盘，重启不丢会话）
    # 需安装 langgraph-checkpoint-sqlite；未安装时自动回退到内存 MemorySaver
    session_store_enabled:bool = True
    session_store_db_path:str = 'data/checkpoints.sqlite'

    # Harness 工具契约层配置（Milestone 0）
    # 工具调用审计（SQLite 落盘，独立于 checkpoints.sqlite）
    harness_audit_enabled:bool = True
    harness_audit_db_path:str = 'data/audit.sqlite'
    # MCP 工具默认超时（秒），ToolSpec.timeout_seconds 可覆盖
    harness_tool_timeout_seconds:float = 30.0
    # Harness 工具契约层配置（Milestone 1：资源锁）
    # 是否启用资源锁（对 target_resource 非空的写操作按资源互斥）
    harness_lock_enabled:bool = True
    # 资源锁等待超时（秒）；0=冲突即拒绝（fail-fast，不排队）
    harness_lock_wait_seconds:float = 0.0
    # Harness 工具契约层配置（Milestone 2：并发上限与优先级调度）
    # 是否启用全局工具并发上限（保护 MCP server / 下游不被并发打爆）
    harness_concurrency_enabled:bool = True
    # 全局同时执行中的工具调用上限（>=1；0 为非法值）
    harness_max_concurrent_tools:int = 10
    # 并发满时等待槽位的超时（秒），超时返回「系统繁忙」拒绝该次调用
    harness_tool_queue_wait_seconds:float = 5.0
    # Harness 工具契约层配置（Milestone 3：高危操作人工审批）
    # 是否启用人工审批（高危写操作执行前需人工批准）
    harness_approval_enabled:bool = True
    # 需要人工审批的风险等级阈值（low/medium/high，含本级及以上）
    harness_approval_risk_threshold:str = 'high'
    # 等待人工审批的超时（秒），超时视为拒绝（fail-closed，默认 5 分钟）
    harness_approval_timeout_seconds:float = 300.0
    # 写操作服务白名单（A2）：允许被自动重启/回滚的服务名（逗号分隔）。
    # 空 = 全部拒绝（fail-closed）；接入真实写操作前必须先在此列出允许触碰的服务。
    harness_write_allowlist:str = ''
    # Harness 后端（多进程/分布式支持）：锁/审批内部实现可切换
    # in_process = 进程内 asyncio 原语（默认，单 worker 部署零依赖）；
    # redis = Redis 分布式锁/审批（多 worker 时锁与审批跨进程一致）
    harness_backend:str = 'in_process'
    # Redis 连接串（harness_backend=redis 时使用）
    redis_url:str = 'redis://localhost:6379/0'
    # 分布式锁租约 TTL（秒）：进程崩溃后自动释放，避免死锁
    harness_lock_redis_ttl_seconds:float = 30.0

    # 文档分块配置
    chunk_max_size:int = 800
    chunk_overlap: int = 100

    # MCP服务配置
    mcp_cls_transport:str = 'streamable-http'
    mcp_cls_url:str = 'http://localhost:8003/mcp'
    mcp_monitor_transport:str = 'streamable-http'
    mcp_monitor_url:str = 'http://localhost:8004/mcp'

    # 通知通道配置（B2：审批通知 / 诊断结论推送）
    # log = 仅日志（默认，零依赖）；dingtalk/wecom/email 需在 .env 配对应凭据
    notify_channel:str = 'log'
    # 通知里附带的审批入口基础地址（如 http://your-server:9900），留空则不生成链接
    notify_base_url:str = ''
    # 钉钉群机器人（加签可选）
    dingtalk_webhook_url:str = ''
    dingtalk_secret:str = ''
    # 企业微信群机器人
    wecom_webhook_url:str = ''
    # SMTP 邮件
    smtp_host:str = ''
    smtp_port:int = 465
    smtp_use_ssl:bool = True
    smtp_username:str = ''
    smtp_password:str = ''
    smtp_from:str = ''
    smtp_to:str = ''


    @property
    def mcp_servers(self) -> Dict[str, Dict[str,Any]]:
        '''获取完整的MCP服务器配置'''
        return {
            'cls': {
                'transport': self.mcp_cls_transport,
                'url': self.mcp_cls_url,
            },
            'monitor': {
                'transport': self.mcp_monitor_transport,
                'url': self.mcp_monitor_url,
            },
        }

    @property
    def cors_origins(self) -> list[str]:
        '''解析 CORS 允许来源（逗号分隔字符串 → 列表）'''
        return [o.strip() for o in self.cors_allow_origins.split(',') if o.strip()]

config = Settings()