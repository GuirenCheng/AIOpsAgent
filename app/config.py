from typing import Dict,Any
from pydantic_settings import BaseSettings,SettingsConfigDict
import os
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
    app_version:str = '1.0.0'
    debug:bool = False
    host:str = '0.0.0.0'
    port:int = 9900

    # DashScope 配置
    dashscope_api_key:str = os.getenv("DASHSCOPE_API_KEY")# 默认空字符串，实际使用需从环境变量中加载
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
    # 单 IP 限流（内存滑动窗口，默认开启，限流阈值可调）
    rate_limit_enabled:bool = True
    rate_limit_per_minute:int = 120
    # CORS 允许来源（逗号分隔；留空则仅同源访问，不发送 CORS 头）
    cors_allow_origins:str = ''
    cors_allow_credentials:bool = False
    # 单个问题最大长度（防滥用）
    max_question_length:int = 4000
    # 会话持久化（LangGraph checkpointer：SQLite 落盘，重启不丢会话）
    # 需安装 langgraph-checkpoint-sqlite；未安装时自动回退到内存 MemorySaver
    session_store_enabled:bool = True
    session_store_db_path:str = 'data/checkpoints.sqlite'

    # 文档分块配置
    chunk_max_size:int = 800
    chunk_overlap: int = 100

    # MCP服务配置
    mcp_cls_transport:str = 'streamable-http'
    mcp_cls_url:str = 'http://localhost:8003/mcp'
    mcp_monitor_transport:str = 'streamable-http'
    mcp_monitor_url:str = 'http://localhost:8004/mcp'


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