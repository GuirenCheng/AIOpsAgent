# AIOpsAgent 项目架构文档

> **项目名称**: aiops-agent-py (v1.2.1)
> **项目描述**: 基于 LangChain 的智能业务助手系统 — 支持 RAG 知识库和 AIOps 智能运维
> **技术栈**: Python 3.11+ / FastAPI / LangChain & LangGraph / Milvus / MCP

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
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ /health      │  │ /api/chat    │  │ /api/aiops   │  │ /api/upload  │      │
│  │ 健康检查      │  │ RAG对话(SSE)  │  │ AIOps诊断(SSE)│  │ 文件上传      │      │
│  └──────────────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│                           │                 │                  │              │
│              ┌────────────┼─────────────────┼──────────────────┘              │
│              │            │                 │                                 │
│              │    ┌───────▼──────┐  ┌───────▼───────┐                         │
│              │    │  RAG Agent   │  │  AIOps Agent  │                         │
│              │    │ (LangGraph)  │  │ (Plan-Exec-   │                         │
│              │    │ ChatQwen +   │  │  Replan)      │                         │
│              │    │ MCP Tools    │  │ Planner→Exec  │                         │
│              │    └───────┬──────┘  │ →Replanner    │                         │
│              │            │         └───────┬───────┘                         │
│              │            │                 │                                 │
│              │            ▼                 ▼                                 │
│              │    ┌─────────────────────────────────────┐                     │
│              │    │          LangChain Tools            │                     │
│              │    │  knowledge_tool  │  time_tool       │                     │
│              │    │  + MCP Tools (via MCP Client)       │                     │
│              │    └─────────────────┬───────────────────┘                     │
└──────────────┼─────────────────────┼─────────────────────────────────────────┘
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
│  app/api/         →  FastAPI 路由 (health, chat, aiops, file)   │
├─────────────────────────────────────────────────────────────────┤
│                      服务层 (Service)                             │
│  app/services/    →  rag_agent_service    RAG 对话智能体         │
│                   →  aiops_service        AIOps 诊断工作流       │
│                   →  vector_index_service  文档索引编排          │
│                   →  vector_search_service 向量检索              │
│                   →  vector_embedding_service 文本向量化          │
│                   →  vector_store_manager   向量库管理           │
│                   →  document_splitter_service 文档分割         │
├─────────────────────────────────────────────────────────────────┤
│                      智能体层 (Agent)                             │
│  app/agent/       →  mcp_client.py       MCP 客户端 (单例)      │
│                   →  aiops/planner.py    计划生成器              │
│                   →  aiops/executor.py   步骤执行器              │
│                   →  aiops/replanner.py  重新规划器              │
│  app/tools/       →  knowledge_tool.py   RAG 知识检索工具        │
│                   →  time_tool.py        时间工具                │
├─────────────────────────────────────────────────────────────────┤
│                      基础设施层 (Infrastructure)                  │
│  app/core/        →  llm_factory.py      DashScope LLM 工厂     │
│                   →  milvus_client.py    Milvus 连接管理         │
│  app/models/      →  Pydantic 请求/响应模型                      │
│  app/utils/       →  Loguru 日志配置                             │
├─────────────────────────────────────────────────────────────────┤
│                      外部依赖 (External)                          │
│  DashScope (Qwen-max)  │  Milvus (向量库)  │  MCP Servers      │
│  text-embedding-v4     │  Docker Compose   │  (CLS + Monitor)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 项目目录结构

```
AIOpsAgent/
├── .env                          # 环境变量 (API密钥、服务配置)
├── .gitignore                    # Git 忽略规则
├── .pre-commit-config.yaml       # Pre-commit 钩子 (black, ruff, isort, bandit)
├── .python-version               # Python 版本 (3.13)
├── Makefile                      # 构建/编排脚本 (634行)
├── pyproject.toml                # 项目配置、依赖管理
├── pyrightconfig.json            # Pyright 类型检查配置
├── start-windows.bat             # Windows 一键启动脚本
├── stop-windows.bat              # Windows 停止脚本
├── uv.lock                       # 依赖锁定文件
├── vector-database.yml           # Milvus Docker Compose 配置
│
├── app/                          # 主应用包
│   ├── __init__.py
│   ├── main.py                   # FastAPI 入口
│   ├── config.py                 # Pydantic Settings 配置
│   ├── agent/                    # 智能体层
│   │   ├── mcp_client.py         # MCP 客户端 (全局单例)
│   │   └── aiops/                # AIOps Plan-Execute-Replan
│   │       ├── __init__.py
│   │       ├── state.py          # PlanExecuteState 状态定义
│   │       ├── planner.py        # 计划生成节点
│   │       ├── executor.py       # 步骤执行节点
│   │       ├── replanner.py      # 重新规划节点
│   │       └── utils.py          # 工具函数
│   ├── api/                      # API 路由层
│   │   ├── health.py             # GET /health
│   │   ├── chat.py               # POST /api/chat, /api/chat_stream
│   │   ├── file.py               # POST /api/upload, /api/index_directory
│   │   └── aiops.py              # POST /api/aiops
│   ├── core/                     # 核心基础设施
│   │   ├── llm_factory.py        # LLM 工厂 (DashScope)
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
│   │   ├── vector_index_service.py       # 文档索引编排
│   │   ├── vector_search_service.py      # 向量检索
│   │   ├── vector_embedding_service.py   # 文本向量化
│   │   └── document_splitter_service.py  # 文档分割
│   ├── tools/                    # LangChain 工具
│   │   ├── knowledge_tool.py     # RAG 知识检索工具
│   │   └── time_tool.py          # 时间工具
│   └── utils/
│       └── logger.py             # Loguru 日志配置
│
├── mcp_servers/                  # MCP 服务器
│   ├── cls_server.py             # CLS 日志服务 (:8003)
│   └── monitor_server.py         # 监控指标服务 (:8004)
│
├── aiops-docs/                   # AIOps 知识库文档
│   ├── cpu_high_usage.md         # CPU 高使用率排查
│   ├── memory_high_usage.md      # 内存高使用率排查
│   ├── disk_high_usage.md        # 磁盘高使用率排查
│   ├── service_unavailable.md    # 服务不可用排查
│   └── slow_response.md          # 响应缓慢排查
│
└── static/                       # 前端 (智能OnCall助手)
    ├── index.html                # 单页应用入口
    ├── app.js                    # 前端逻辑 (71KB)
    └── styles.css                # 样式 (21KB)
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
RAG Agent (LangGraph + ChatQwen + MemorySaver)
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
- 使用 LangGraph `MemorySaver` 管理会话状态，支持多轮对话
- 每次对话自动从 Milvus 检索相关知识文档（Top-K=3）
- 可通过 MCP 工具调用外部日志和监控数据
- 支持同步 (`/api/chat`) 和流式 (`/api/chat_stream`) 两种模式

---

### 4.2 AIOps 诊断智能体 (Plan-Execute-Replan 模式)

```
警告输入 → FastAPI /api/aiops (SSE)
              │
              ▼
    ┌─────────────────────┐
    │     Planner         │
    │  计划生成节点         │
    │  ① 检索知识库文档     │
    │  ② LLM 分析告警      │
    │  ③ 生成诊断步骤计划   │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │     Executor        │
    │  步骤执行节点         │
    │  LLM + ToolNode     │
    │  调用 MCP 工具获取    │
    │  监控/日志数据        │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │    Replanner        │
    │  重新规划节点         │
    │                      │
    ├─ 任务未完成?          │
    │  → 更新计划          │
    │  → 回到 Executor     │
    │  (最多 8 轮)         │
    │                      │
    └─ 任务完成 / 超5轮强制结束
       → 生成最终诊断报告
       → SSE 流式返回
              │
    ┌─────────┴───────────┐
    │  返回到 Planner      │
    │  (如果需重新规划)      │
    └─────────────────────┘
```

**安全保护机制**:
- 最多执行 8 步，防止无限循环
- 超过 5 步后强制生成回复
- 每步执行结果均记录到 `past_steps`

---

## 5. 数据流

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

## 6. 运行时服务拓扑

| 服务 | 端口 | 技术栈 | 说明 |
|------|------|--------|------|
| **主服务 (FastAPI)** | 9900 | Python + Uvicorn | API 网关 + 智能体编排 |
| **CLS MCP Server** | 8003 | Python + FastMCP | 日志搜索服务 (当前为模拟数据) |
| **Monitor MCP Server** | 8004 | Python + FastMCP | 监控指标服务 (当前为模拟数据) |
| **Milvus** | 19530 | Docker | 向量存储与语义检索 |
| **etcd** | 2379 | Docker | Milvus 元数据管理 |
| **MinIO** | 9000/9001 | Docker | Milvus 对象存储 |
| **Attu** | 3000 | Docker | Milvus Web 管理界面 |

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
       ├── 连接 MCP Servers
       ├── 初始化 RAG Agent
       └── 初始化 AIOps Agent
```

> **Windows**: 使用 `start-windows.bat` 一键启动所有服务
> **macOS/Linux**: 使用 `make start` 启动

---

## 7. 技术栈详情

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

### 前端

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| 页面结构 | Vanilla HTML5 | 单页应用 |
| 样式 | Vanilla CSS3 | 界面样式 (21KB) |
| 交互逻辑 | Vanilla JavaScript | SSE 流式消费、Markdown 渲染 |
| Markdown | marked.js | 回答内容渲染 |
| 代码高亮 | highlight.js | 代码块语法高亮 |

### 基础依赖

| 类别 | 组件 |
|------|------|
| Docker 基础设施 | Milvus, etcd, MinIO, Attu |
| Python 包管理 | uv |
| 代码质量 | ruff, black, isort, mypy, bandit |
| 日志 | Loguru |

---

## 8. 关键设计模式

### 8.1 全局单例 (Global Singleton)

所有核心服务以模块级单例实现，确保全应用共享同一实例：

```python
# 示例：每个服务模块 export 一个全局实例
milvus_manager = MilvusClientManager()      # app/core/milvus_client.py
vector_store_manager = VectorStoreManager()  # app/services/vector_store_manager.py
rag_agent_service = RAGAgentService()        # app/services/rag_agent_service.py
aiops_service = AIOpsService()               # app/services/aiops_service.py
```

### 8.2 MCP 工具集成

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

### 8.3 Plan-Execute-Replan 模式

AIOps Agent 采用经典的 Plan-Execute-Replan 模式：

- **Planner**: 根据告警和知识库生成诊断计划
- **Executor**: 执行计划中的一个步骤，调用 MCP 工具收集数据
- **Replanner**: 根据执行结果评估进度，决定继续/重新规划/结束

状态通过 `PlanExecuteState` TypedDict 在多节点间传递。

### 8.4 SSE 流式响应

两个核心接口均采用 Server-Sent Events 实现流式输出：

```
POST /api/chat_stream  →  text/event-stream  →  逐字推送对话内容
POST /api/aiops        →  text/event-stream  →  逐步推送诊断过程
```

---

## 9. 配置概览

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

---

