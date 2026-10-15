# AIOpsAgent 项目架构文档

> **项目名称**: aiops-agent-py (v1.2.1)
> **项目描述**: 基于 LangChain 的智能业务助手系统 — 支持 RAG 知识库问答、AIOps 智能运维诊断，以及面向多 agent 并发运维的「工具执行护栏」harness
> **技术栈**: Python 3.11+ / FastAPI / LangChain & LangGraph / Milvus / MCP / Redis（可选）

---

## 1. 系统总体架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              用户浏览器 (Browser)                               │
│                            static/index.html + app.js                         │
│                        SSE Streaming / REST API 通信                          │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        FastAPI 主服务 (0.0.0.0:9900)                          │
│                              app/main.py                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │
│  │ /health      │ │ /api/chat*   │ │ /api/aiops   │ │ /api/upload  │          │
│  │ 健康检查      │ │ RAG对话(SSE)  │ │ AIOps诊断(SSE)│ │ 文件上传/索引 │          │
│  └──────────────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘          │
│                          │   ┌────────────┴───────────────┐  │                 │
│                          │   │ /api/harness/approvals     │  │                 │
│                          │   │ 高危操作人工审批             │  │                 │
│                          │   └────────────────────────────┘  │                 │
│                          │                                   │                 │
│              ┌───────────┼───────────────────┬───────────────┘                 │
│              │           │                   │                                 │
│              │   ┌───────▼──────┐   ┌────────▼────────┐                        │
│              │   │  RAG Agent   │   │  AIOps Agent    │                        │
│              │   │ (LangGraph)  │   │ (Plan-Exec-     │                        │
│              │   │ ChatQwen +   │   │  Replan)        │                        │
│              │   │ 兜底护栏      │   │ Planner→Exec    │                        │
│              │   └───────┬──────┘   │ →Replanner      │                        │
│              │           │          └────────┬────────┘                        │
│              │           │                   │                                 │
│              │           ▼                   ▼                                 │
│              │   ┌──────────────────────────────────────────┐                  │
│              │   │        app/harness 工具执行护栏            │                  │
│              │   │  审批门 → 并发槽位 → 资源锁 → 执行          │                  │
│              │   │  (execute_guarded 唯一收口点)              │                  │
│              │   │  + 自动回滚 supervisor + 审计              │                  │
│              │   └──────────────────┬───────────────────────┘                  │
│              │                      │                                          │
│              │                      ▼                                          │
│              │   ┌─────────────────────────────────────┐                       │
│              │   │          LangChain Tools            │                       │
│              │   │  knowledge_tool  │  time_tool       │                       │
│              │   │  + MCP Tools (via MCP Client)       │                       │
│              │   └─────────────────┬───────────────────┘                       │
└──────────────┼─────────────────────┼───────────────────────────────────────────┘
               │                     │
               ▼                     ▼
┌──────────────────────┐  ┌─────────────────────────────────────┐
│    Milvus 向量数据库   │  │         MCP Servers (FastMCP)         │
│  (Docker :19530)     │  │  ┌──────────────┐ ┌──────────────┐   │
│  + etcd + MinIO      │  │  │ CLS Server   │ │Monitor Server│   │
│  + Attu Web UI       │  │  │ (:8003)      │ │ (:8004)      │   │
│                      │  │  │ 日志搜索服务  │ │ 监控指标服务  │   │
│  ┌────────────────┐  │  │  └──────────────┘ └──────────────┘   │
│  │ aiops-docs/    │  │  │        (目前为模拟数据实现)            │
│  │ 5份故障排查手册 │  │  └─────────────────────────────────────┘
│  │ (Markdown)     │  │
│  └────────────────┘  │
└──────────────────────┘
```

---

## 2. 分层架构 (Layered Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                      表示层 (Presentation)                        │
│  static/          →  HTML/CSS/JS 单页应用 (智能OnCall助手)        │
│  app/api/         →  FastAPI 路由 (health, chat, aiops, file,   │
│                      approval)                                   │
├─────────────────────────────────────────────────────────────────┤
│                      服务层 (Service)                             │
│  app/services/    →  rag_agent_service      RAG 对话智能体       │
│                   →  aiops_service          AIOps 诊断工作流     │
│                   →  vector_index_service   文档索引/检索编排     │
│                   →  vector_store_manager   向量库封装           │
│                   →  vector_embedding_service 文本向量化         │
│                   →  document_splitter_service 文档分割         │
├─────────────────────────────────────────────────────────────────┤
│                      智能体层 (Agent)                             │
│  app/agent/       →  mcp_client.py       MCP 客户端 (单例)      │
│                   →  guardrails.py       兜底护栏 (防编造)       │
│                   →  aiops/planner.py    计划生成器              │
│                   →  aiops/executor.py   步骤执行器              │
│                   →  aiops/replanner.py  重新规划器              │
│  app/tools/       →  knowledge_tool.py   RAG 知识检索工具        │
│                   →  time_tool.py        时间工具                │
│                   →  tool_wrapper.py     本地工具护栏包装        │
├─────────────────────────────────────────────────────────────────┤
│                   护栏层 (Harness，工具执行安全)                  │
│  app/harness/     →  guard.py          execute_guarded 唯一收口 │
│                   →  registry.py       工具契约单一真相源        │
│                   →  audit.py          SQLite 审计              │
│                   →  lock.py           资源锁 (进程内/Redis)     │
│                   →  scheduler.py      并发上限 + 优先级         │
│                   →  approval.py       高危操作人工审批           │
│                   →  rollback.py       自动回滚 supervisor       │
│                   →  redis_backends.py Redis 锁/审批后端         │
│                   →  tool_spec.py      ToolSpec 契约模型        │
├─────────────────────────────────────────────────────────────────┤
│                      基础设施层 (Infrastructure)                  │
│  app/core/        →  checkpointer.py    LangGraph 持久化工厂     │
│                   →  middleware.py      鉴权/限流/RequestID      │
│                   →  milvus_client.py   Milvus 连接管理          │
│  app/models/      →  Pydantic 请求/响应模型                      │
│  app/utils/       →  Loguru 日志配置                             │
├─────────────────────────────────────────────────────────────────┤
│                      外部依赖 (External)                          │
│  DashScope (Qwen-max)  │  Milvus (向量库)  │  MCP Servers       │
│  text-embedding-v4     │  Redis (可选)     │  (CLS + Monitor)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 项目目录结构

```
AIOpsAgent/
├── .env                          # 环境变量 (API密钥、服务配置)
├── Makefile                      # 构建/编排脚本
├── pyproject.toml                # 项目配置、依赖管理
├── start-windows.bat             # Windows 一键启动脚本
├── stop-windows.bat              # Windows 停止脚本
├── uv.lock                       # 依赖锁定文件
├── vector-database.yml           # Milvus Docker Compose 配置
├── docker-compose.yml            # 主服务 + MCP + Nginx + Redis(可选)
│
├── app/                          # 主应用包
│   ├── main.py                   # FastAPI 入口 + lifespan（Milvus/回滚 supervisor 启停）
│   ├── config.py                 # Pydantic Settings 配置
│   ├── api/                      # API 路由层
│   │   ├── health.py             # GET /health
│   │   ├── chat.py               # POST /api/chat, /api/chat_stream, /api/chat/clear, /api/chat/session/{id}
│   │   ├── file.py               # POST /api/upload, /api/index_directory
│   │   ├── aiops.py              # POST /api/aiops
│   │   └── approval.py           # GET/POST /api/harness/approvals
│   ├── agent/                    # 智能体层
│   │   ├── mcp_client.py         # MCP 客户端 (全局单例)
│   │   ├── guardrails.py         # 兜底护栏（防编造：检索短路/工具失败/脱稿检测）
│   │   └── aiops/                # AIOps Plan-Execute-Replan
│   │       ├── state.py          # PlanExecuteState 状态定义
│   │       ├── planner.py        # 计划生成节点
│   │       ├── executor.py       # 步骤执行节点
│   │       ├── replanner.py      # 重新规划节点
│   │       └── utils.py          # 工具函数
│   ├── harness/                  # ★ 工具执行护栏
│   │   ├── guard.py              # execute_guarded 唯一收口点
│   │   ├── registry.py           # 工具契约单一真相源
│   │   ├── audit.py              # SQLite 审计记录器
│   │   ├── lock.py               # 资源锁（进程内 / 工厂切 Redis）
│   │   ├── scheduler.py          # 并发上限 + 优先级调度
│   │   ├── approval.py           # 高危操作人工审批（进程内 / 工厂切 Redis）
│   │   ├── rollback.py           # 自动回滚 supervisor
│   │   ├── redis_backends.py     # Redis 分布式锁 / 跨进程审批后端
│   │   └── tool_spec.py          # ToolSpec 契约模型
│   ├── core/                     # 核心基础设施
│   │   ├── checkpointer.py       # LangGraph checkpointer 工厂（SQLite/内存）
│   │   ├── middleware.py         # ApiKeyAuth / RateLimit / RequestID 中间件
│   │   └── milvus_client.py      # Milvus 连接管理
│   ├── models/                   # Pydantic 数据模型
│   │   ├── request.py            # 请求模型
│   │   ├── response.py           # 响应模型
│   │   ├── aiops.py              # AIOps 模型
│   │   └── document.py           # 文档模型
│   ├── services/                 # 业务逻辑服务
│   │   ├── rag_agent_service.py          # RAG 对话智能体
│   │   ├── aiops_service.py              # AIOps 诊断工作流
│   │   ├── vector_store_manager.py       # Milvus VectorStore 封装
│   │   ├── vector_index_service.py       # 文档索引/检索编排
│   │   ├── vector_embedding_service.py   # 文本向量化
│   │   └── document_splitter_service.py  # 文档分割
│   ├── tools/                    # LangChain 工具
│   │   ├── knowledge_tool.py     # RAG 知识检索工具
│   │   ├── time_tool.py          # 时间工具
│   │   └── tool_wrapper.py       # 本地工具护栏包装（guarded_* 单例）
│   └── utils/
│       └── logger.py             # Loguru 日志配置
│
├── mcp_servers/                  # MCP 服务器
│   ├── cls_server.py             # CLS 日志服务 (:8003)
│   ├── monitor_server.py         # 监控指标服务 (:8004)
│   ├── import_loghub.py          # 日志数据导入脚本
│   └── import_metrics.py         # 指标数据导入脚本
│
├── aiops-docs/                   # AIOps 知识库文档
│   ├── cpu_high_usage.md         # CPU 高使用率排查
│   ├── memory_high_usage.md      # 内存高使用率排查
│   ├── disk_high_usage.md        # 磁盘高使用率排查
│   ├── service_unavailable.md    # 服务不可用排查
│   └── slow_response.md          # 响应缓慢排查
│
├── static/                       # 前端 (智能OnCall助手)
│   ├── index.html                # 单页应用入口
│   ├── app.js                    # 前端逻辑
│   └── styles.css                # 样式
│
└── tests/                        # 测试（pytest，无网络，asyncio_mode=auto）
```

---

## 4. 核心智能体工作流

### 4.1 RAG 对话智能体

```
用户提问
    │
    ▼
FastAPI /api/chat_stream (SSE)
    │
    ▼
RAG Agent (LangGraph + ChatQwen + checkpointer)
    │
    ├──→ retrieve_knowledge (Milvus 向量检索, Top-K=3)
    ├──→ MCP Tools (search_log, query_cpu, query_memory...)
    └──→ get_current_time
    │
    ▼
Qwen-max LLM (结合检索上下文) → 生成回答
    │
    ▼
SSE 流式返回前端 (marked.js 渲染 Markdown)
```

**特点**:
- 会话历史经 LangGraph checkpointer 持久化（默认 SQLite，重启不丢；未装依赖回退内存 MemorySaver）
- 每次对话自动从 Milvus 检索相关知识文档（Top-K=3，相关性阈值 0.5）
- 通过 MCP 工具调用外部日志和监控数据
- 支持同步 (`/api/chat`) 和流式 (`/api/chat_stream`) 两种模式

### 4.2 AIOps 诊断智能体 (Plan-Execute-Replan 模式)

```
告警输入 → FastAPI /api/aiops (SSE)
              │
              ▼
    ┌─────────────────────┐
    │     Planner         │  ① 检索知识库文档
    │  计划生成节点         │  ② LLM 分析告警
    │                     │  ③ 生成结构化诊断步骤 (PlanStep)
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │     Executor        │  LLM + ToolNode
    │  步骤执行节点         │  调用 MCP 工具获取监控/日志数据
    │                     │  产出 Observation
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │    Replanner        │  按证据覆盖度决定
    │  重新规划节点         │  ├─ 继续 → 回 Executor
    │                     │  ├─ 重新规划 → 更新 plan
    │                     │  └─ 证据充足 → 生成最终报告
    └─────────┬───────────┘
              ▼
        SSE 流式返回诊断报告
```

**安全保护机制**:
- 最多执行 8 步，防止无限循环
- 超过 5 步后强制生成回复
- 每步执行结果均记录到 `past_steps`
- 结构化状态（`PlanStep`/`Observation`）驱动，evidence 类型归一化（logs/metrics/knowledge/time/synthesis）

---

## 5. 工具执行护栏 (Harness)

`app/harness/` 是一套让「多个并发运维 agent 安全执行写操作」的护栏层。与 `app/agent/guardrails.py`（防编造的兜底护栏）不同，harness 关注**工具执行的安全与合规**。

### 5.1 六层护栏（由外到内）

| 层 | 模块 | 作用 |
|---|---|---|
| M0 契约层 | `tool_spec.py` / `registry.py` / `audit.py` | 工具元数据（只读/幂等/风险/资源/回滚）、单一真相源、SQLite 审计 |
| M1 资源锁 | `lock.py` | 同资源并发写互斥，冲突 fail-fast |
| M2 并发调度 | `scheduler.py` | 全局并发上限 + 优先级入槽 |
| M3 人工审批 | `approval.py` + `api/approval.py` | 高危写操作需人工 approve/deny，超时 fail-closed |
| 执行 | `guard.py` | 幂等重试 + 超时 + 审计 |

嵌套顺序：**审批门(M3) → 并发槽位(M2) → 资源锁(M1) → 执行(M0)**。审批门最外，等人工决定期间不占并发槽位/资源锁。

### 5.2 唯一收口点：`execute_guarded`

护栏核心从 MCP 拦截器抽离为通用函数 `app/harness/guard.py::execute_guarded(tool_name, args, server, handler)`：

- `retry_interceptor`（MCP 工具）退化为薄适配器；
- 本地工具（`retrieve_knowledge`/`get_current_time`）经 `tool_wrapper.guard_local_tool` 包装后也走同一条链；
- 回滚 supervisor 也复用同一套护栏执行回滚动作。

三条路径共用一条护栏链，保证「审计/锁/审批/超时/重试」对所有工具调用一致生效。

### 5.3 自动回滚 supervisor

写操作（如 `restart_service`）失败且契约声明了 `rollback_tool` 时：

1. `guard._run_with_retry` 审计 `rollback_required`；
2. 信号入队到 `rollback.py::RollbackManager`（后台 `asyncio.Queue` 消费者）；
3. supervisor 按名调用回滚工具（`rollback_service`），审计 `rollback_executed`/`rollback_skipped`/`rollback_failed`。

回滚工具契约 `risk=MEDIUM`（低于 HIGH 审批阈值 → 自动回滚无需人工在环）、`rollback_tool=None`（回滚自身失败不再触发回滚，防无限回滚）。

### 5.4 Redis 可切换后端

`ResourceLockManager`/`ApprovalManager` 内部可切换实现，`config.harness_backend` 一键切换：

- **`in_process`（默认）**：进程内 asyncio 原语，单 worker 部署零依赖；
- **`redis`**：`redis_backends.py` 提供 `RedisResourceLockManager`（`SET NX PX` 租约 + Lua compare-and-delete）与 `RedisApprovalManager`（hash + INCR 计数 + 轮询决定），多 worker 下锁/审批跨进程一致。

工厂遵循 `checkpointer.py` 的 flag + 懒 import + `ImportError` 回退模式。并发槽位 `tool_gate` 保持进程内（软限流，多 worker 退化为各 worker 独立限流，可接受）。

---

## 6. 数据流

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────┐
│ aiops-docs/  │────►│ Document Splitter│────►│ Embedding       │────►│  Milvus  │
│ *.md 知识库   │上传 │ (800字分块,       │     │ Service          │     │ 向量存储  │
│              │     │  100字重叠)       │     │ (1024维向量)     │     │          │
└──────────────┘     └──────────────────┘     └─────────────────┘     └────┬─────┘
                                                                          │
                                                                          │ 检索
                                                                          ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────┐
│ 用户问题      │────►│ RAG Agent        │────►│ Vector Search   │────►│ LLM 生成  │
│              │     │ (LangGraph)      │     │ (Top-K=3)      │     │ 回答     │
└──────────────┘     └──────────────────┘     └─────────────────┘     └──────────┘

┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────┐
│ 告警信息      │────►│ AIOps Agent      │────►│ Planner 检索    │────►│ 诊断报告  │
│              │     │ (Plan-Exec-Replan)│     │ 知识库 + MCP工具 │     │ (SSE输出) │
└──────────────┘     └──────────────────┘     └─────────────────┘     └──────────┘
```

---

## 7. 运行时服务拓扑

| 服务 | 端口 | 技术栈 | 说明 |
|------|------|--------|------|
| **主服务 (FastAPI)** | 9900 | Python + Uvicorn | API 网关 + 智能体编排 + harness |
| **CLS MCP Server** | 8003 | Python + FastMCP | 日志搜索服务 (当前为模拟数据) |
| **Monitor MCP Server** | 8004 | Python + FastMCP | 监控指标服务 (当前为模拟数据) |
| **Milvus** | 19530 | Docker | 向量存储与语义检索 |
| **etcd** | 2379 | Docker | Milvus 元数据管理 |
| **MinIO** | 9000/9001 | Docker | Milvus 对象存储 |
| **Attu** | 3000 | Docker | Milvus Web 管理界面 |
| **Redis（可选）** | 6379 | Docker | harness 多 worker 后端（`HARNESS_BACKEND=redis` 时启用） |

### 服务启动顺序

```
1. Docker 容器启动 (vector-database.yml)
   ├── etcd     → Milvus 元数据存储
   ├── MinIO    → Milvus 对象存储
   ├── Milvus   → 向量数据库
   └── Attu     → Web 管理界面

2. MCP 服务器启动
   ├── cls_server.py      (:8003)
   └── monitor_server.py  (:8004)

3. 主服务启动
   └── app/main.py         (:9900)
       ├── 连接 Milvus
       ├── 启动回滚 supervisor
       ├── 初始化 RAG Agent
       └── 初始化 AIOps Agent
```

> **Windows**: 使用 `start-windows.bat` 一键启动所有服务
> **macOS/Linux**: 使用 `make start` 启动

---

## 8. 技术栈详情

### 核心框架

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| Web 框架 | FastAPI + Uvicorn | HTTP API + SSE 流式响应 |
| LLM 编排 | LangChain + LangGraph | Agent 工作流、工具调用、状态管理 |
| LLM 服务 | Alibaba DashScope (Qwen-max) | 对话推理、计划生成、工具决策 |
| 向量嵌入 | text-embedding-v4 (1024维) | 文本转向量 |
| 向量数据库 | Milvus 2.5.10 (Docker) | 语义检索、知识库存储 |
| MCP 框架 | FastMCP | MCP 服务器实现 |
| MCP 传输 | streamable-http | 主服务与 MCP 服务器通信 |
| 分布式（可选） | Redis 7 | harness 锁/审批多 worker 后端 |
| 持久化 | SQLite (aiosqlite) | 会话 checkpoints + 工具审计 |

### 前端

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| 页面结构 | Vanilla HTML5 | 单页应用 |
| 样式 | Vanilla CSS3 | 界面样式 |
| 交互逻辑 | Vanilla JavaScript | SSE 流式消费、Markdown 渲染 |
| Markdown | marked.js | 回答内容渲染 |
| 代码高亮 | highlight.js | 代码块语法高亮 |

### 基础依赖

| 类别 | 组件 |
|------|------|
| Docker 基础设施 | Milvus, etcd, MinIO, Attu, Redis(可选) |
| Python 包管理 | uv |
| 代码质量 | ruff, black, isort, mypy, bandit |
| 日志 | Loguru |
| 测试 | pytest + pytest-asyncio + pytest-mock + fakeredis |

---

## 9. 关键设计模式

### 9.1 全局单例 (Global Singleton)

所有核心服务以模块级单例实现，确保全应用共享同一实例：

```python
milvus_manager = MilvusClientManager()        # app/core/milvus_client.py
vector_store_manager = VectorStoreManager()    # app/services/vector_store_manager.py
rag_agent_service = RAGAgentService()         # app/services/rag_agent_service.py
aiops_service = AIOpsService()                # app/services/aiops_service.py
tool_registry = _seed_registry(ToolRegistry())# app/harness/registry.py
audit_logger = AuditLogger()                  # app/harness/audit.py
resource_locks = _create_resource_locks()     # app/harness/lock.py
approval_manager = _create_approval_manager() # app/harness/approval.py
rollback_manager = RollbackManager()          # app/harness/rollback.py
```

### 9.2 MCP 工具集成

MCP 服务器通过 `langchain-mcp-adapters` 将外部工具转换为 LangChain Tool，统一注入 Agent：

```
MCP Server (FastMCP)                   主服务
┌─────────────────┐          ┌─────────────────────────┐
│ cls_server.py   │─HTTP────►│ mcp_client.py (单例)     │
│ :8003/mcp       │          │ → load_mcp_tools()       │
│  - search_log   │          │ → 转换为 LangChain Tool   │
│  - get_timestamp│          │ → 绑定到 RAG/AIOps Agent  │
└─────────────────┘          └─────────────────────────┘
```

### 9.3 Plan-Execute-Replan 模式

AIOps Agent 采用经典的 Plan-Execute-Replan 模式：**Planner** 生成结构化计划 → **Executor** 执行一步并产出观察 → **Replanner** 按证据覆盖度决定继续/重新规划/结束。状态通过 `PlanExecuteState` TypedDict 在多节点间传递，经 checkpointer 持久化。

### 9.4 工具执行护栏（唯一收口）

护栏核心 `execute_guarded` 是唯一收口点：MCP 工具（`retry_interceptor`）、本地工具（`guard_local_tool`）、回滚 supervisor 三条路径共用同一条「审批门 → 并发槽位 → 资源锁 → 幂等重试+超时+审计」链。新增护栏只需改 `guard.py`。

### 9.5 后端选择工厂

`checkpointer.py`（SQLite/内存）、`lock.py`（进程内/Redis）、`approval.py`（进程内/Redis）均采用同一模式：flag 开关 + 懒 import + `ImportError` 回退，保证默认部署零依赖、可选能力按需启用。

### 9.6 两类「护栏」的区分

- **工具执行护栏（`app/harness/`）**：管「工具能不能安全执行」——审批/锁/并发/审计/回滚/重试。
- **Agent 兜底护栏（`app/agent/guardrails.py`）**：管「模型会不会编造」——检索未命中短路到固定话术、工具失败改写成防编造指令、生成后脱稿检测（citation_check）。

### 9.7 SSE 流式响应

核心接口均采用 Server-Sent Events 实现流式输出：

```
POST /api/chat_stream  →  text/event-stream  →  流式推送对话内容
POST /api/aiops        →  text/event-stream  →  逐步推送诊断过程
```

---

## 10. 配置概览

| 配置项 | 值 | 说明 |
|--------|-----|------|
| HOST | 0.0.0.0 | 监听所有网卡 |
| PORT | 9900 | 主服务端口 |
| LLM Provider | DashScope (阿里云) | OpenAI 兼容模式 |
| LLM Model | qwen-max | 推理模型 |
| Embedding Model | text-embedding-v4 | 向量维度 1024 |
| Milvus Host | localhost:19530 | 本地向量数据库（Docker） |
| RAG Top-K | 3 | 检索返回数量 |
| Chunk Size | 800 | 文档分块大小 |
| Chunk Overlap | 100 | 分块重叠大小 |
| MCP CLS | http://localhost:8003/mcp | CLS MCP 服务地址 |
| MCP Monitor | http://localhost:8004/mcp | Monitor MCP 服务地址 |
| 会话持久化 | SQLite (data/checkpoints.sqlite) | LangGraph checkpointer |
| 审计 | SQLite (data/audit.sqlite) | 工具调用审计 |
| harness_backend | in_process | 锁/审批后端（in_process / redis） |
| redis_url | redis://localhost:6379/0 | Redis 连接串 |
| harness_approval_risk_threshold | high | 触发人工审批的风险等级 |
| harness_approval_timeout_seconds | 300 | 审批等待超时（fail-closed） |
| harness_max_concurrent_tools | 10 | 全局并发工具上限 |
| harness_lock_redis_ttl_seconds | 30 | 分布式锁租约 TTL |

---

## 11. 安全加固

- **请求层**：API Key 鉴权中间件（默认关）、单 IP 限流（默认 120/分钟）、CORS 配置驱动、RequestID 追踪、生产可关 `/docs`。
- **执行层（harness）**：只读工具不重试、非幂等写工具单次执行防重复、高危写工具强制人工审批、写失败自动回滚、全量操作落 SQLite 审计。
- **生成层（guardrails）**：检索未命中/出错硬短路防编造、工具失败追加「禁止编造」指令、生成后脱稿检测告警。
