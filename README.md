# AIOpsAgent

> 企业级智能对话和运维助手，支持 RAG 知识库问答和 AIOps 智能诊断

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)](https://www.langchain.com/)

## ✨ 核心特性

- 🤖 **智能对话** - LangChain 多轮对话 + 流式输出
- 📚 **RAG 问答** - 向量检索增强，支持文档上传、自动建立向量索引、自动更新知识库
- 🔧 **AIOps 诊断** - Plan-Execute-Replan 自动故障诊断和根因分析
- 🛡️ **工具执行护栏** - 审批门 / 并发槽位 / 资源锁 / 审计 / 自动回滚，安全跑多 agent 写操作
- 🔒 **高危操作审批** - 高危写操作人工在环（approve/deny）
- 🌐 **Web 界面** - 现代化 UI，支持多种对话模式：快速问答/流式对话
- 🔌 **MCP 集成** - 日志查询和监控数据工具接入
- 🔎 **全链路 Trace** - Langfuse 自托管，一个请求一条 trace，回放每一步 LLM 与工具调用

## 🛠️ 技术栈

- **框架**: FastAPI + LangChain + LangGraph
- **LLM**: 阿里云 DashScope (通义千问)
- **向量库**: Milvus
- **工具协议**: MCP (Model Context Protocol)
- **分布式（可选）**: Redis 7（harness 多 worker 后端）
- **可观测**: Langfuse（自托管全链路 trace）

## 🚀 快速开始

### 环境要求
- Python 3.11+
- 阿里云 DashScope API Key ([获取地址](https://dashscope.aliyun.com/))

### 安装和启动

#### Linux/macOS 环境

```bash
# 1. 克隆项目
git clone <repository_url>
cd AIOpsAgent

# 2. 安装依赖（推荐使用 uv）
# 方式 1: 使用 uv（推荐，更快）
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 方式 2: 使用 pip
pip install -e .

# 3. 编辑配置文件
# 首次使用需要编辑 .env 文件，填入你的 DASHSCOPE_API_KEY
vim .env  # 或使用其他编辑器

# 4. 一键初始化（启动 Docker + 服务 + 上传文档）
make init

# 5. 一键启动
make start
```

#### Windows 环境（PowerShell/CMD）

如果Windows 不支持 `make` 命令，可以手动执行以下步骤以启动服务：

```powershell
# 1. 克隆项目
git clone <repository_url>
cd AIOpsAgent

# 2. 创建虚拟环境并安装依赖
# 方式 1: 使用 uv（推荐，更快）
pip install uv
# 创建虚拟环境
uv venv
# 激活虚拟环境
.venv\Scripts\activate
# 安装所有依赖
uv pip install -e .

# 方式 2: 使用 pip
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3. 编辑配置文件
# 使用记事本或其他编辑器打开 .env 文件，填入你的 DASHSCOPE_API_KEY
notepad .env

# 4. 启动 Docker Desktop
# 确保 Docker Desktop 已安装并正在运行

# 5. 启动 Milvus 向量数据库（Docker Compose）
docker compose -f vector-database.yml up -d

# 6. 等待 Milvus 启动完成（约 5-10 秒）
timeout /t 10

# 7. 启动 MCP 服务
# 启动 CLS 日志查询服务（新开一个 PowerShell 窗口）
python mcp_servers/cls_server.py

# 启动 Monitor 监控服务（新开一个 PowerShell 窗口）
python mcp_servers/monitor_server.py

# 8. 启动 FastAPI 主服务（新开一个 PowerShell 窗口）
# 注意：日志会自动输出到 logs\app_YYYY-MM-DD.log
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 9. 上传文档到向量库（新开一个 PowerShell 窗口）
# 等待服务启动完成后执行
timeout /t 5
python -c "import requests, os, time; [requests.post('http://localhost:9900/api/upload', headers={'X-API-Key': 'your-api-key'}, files={'file': open(f'aiops-docs/{f}', 'rb')}) or time.sleep(1) for f in os.listdir('aiops-docs') if f.endswith('.md')]"
```

**Windows 一键启动脚本**（推荐）

使用启动脚本：

```powershell
# 启动所有服务
.\start-windows.bat

# 停止所有服务
.\stop-windows.bat
```

### 访问服务
- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs

## 📡 API 接口

> **鉴权说明**：生产环境已开启 API 鉴权（`.env` 中 `API_AUTH_ENABLED=true`）。
> 所有 `/api/*` 请求需携带请求头 `X-API-Key: <你的 API_AUTH_KEY>`，否则返回 401。
> `/health` 与静态页面不拦截；Web 前端会自动附带该头，无需手动处理。
> 用 curl / 脚本调用时，把下面示例里的 `your-api-key` 换成 `.env` 里的 `API_AUTH_KEY`。

### 核心接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 健康检查 | GET | `/health` | 服务状态检查 |
| 普通对话 | POST | `/api/chat` | 一次性返回 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出 |
| 清空会话 | POST | `/api/chat/clear` | 清空会话历史 |
| 会话历史 | GET | `/api/chat/session/{id}` | 查询会话历史 |
| AIOps 诊断 | POST | `/api/aiops` | 自动故障诊断（流式） |
| 文件上传 | POST | `/api/upload` | 上传并索引文档（txt/md/pdf/docx/html/pptx/xlsx/csv，非 md 自动转 Markdown） |
| 目录索引 | POST | `/api/index_directory` | 索引目录下文档 |
| 待审批列表 | GET | `/api/harness/approvals` | 高危操作待审批单 |
| 批准 | POST | `/api/harness/approvals/{id}/approve` | 批准高危操作 |
| 拒绝 | POST | `/api/harness/approvals/{id}/deny` | 拒绝高危操作 |

### 使用示例

```bash
# 普通对话
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"Id":"session-123","Question":"你好"}'

# 流式对话
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"Id":"session-123","Question":"你好"}' \
  --no-buffer

# AIOps 诊断
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"session_id":"session-123"}' \
  --no-buffer
```

## 📁 项目结构

```
AIOpsAgent/
├── app/                                    # 应用核心
│   ├── __init__.py                         # 包初始化（自动加载日志配置）
│   ├── main.py                             # FastAPI 应用入口
│   ├── config.py                           # 配置管理（环境变量、MCP 服务器配置）
│   ├── api/                                # API 路由层
│   │   ├── __init__.py
│   │   ├── chat.py                         # 对话接口（RAG 聊天）
│   │   ├── aiops.py                        # AIOps 接口（故障诊断）
│   │   ├── file.py                         # 文件管理（文档上传）
│   │   ├── approval.py                     # Harness 人工审批
│   │   └── health.py                       # 健康检查（服务状态）
│   ├── services/                           # 业务服务层
│   │   ├── __init__.py
│   │   ├── rag_agent_service.py            # RAG Agent（LangGraph 状态图）
│   │   ├── aiops_service.py                # AIOps 服务（计划-执行-重规划）
│   │   ├── vector_store_manager.py         # 向量存储管理器
│   │   ├── vector_embedding_service.py     # 向量embedding服务
│   │   ├── vector_index_service.py         # 向量索引/检索服务
│   │   └── document_splitter_service.py    # 文档分割服务
│   ├── agent/                              # Agent 模块
│   │   ├── __init__.py
│   │   ├── mcp_client.py                   # MCP 客户端（工具调用）
│   │   ├── guardrails.py                   # 兜底护栏（防编造）
│   │   └── aiops/                          # AIOps 核心逻辑
│   │       ├── __init__.py
│   │       ├── planner.py                  # 计划制定器
│   │       ├── executor.py                 # 步骤执行器
│   │       ├── replanner.py                # 重规划器
│   │       ├── state.py                    # 状态定义
│   │       └── utils.py                    # 工具函数
│   ├── harness/                            # 工具执行护栏（安全跑写操作）
│   │   ├── guard.py                        # execute_guarded 唯一收口点
│   │   ├── registry.py                     # 工具契约单一真相源
│   │   ├── audit.py                        # SQLite 工具调用审计
│   │   ├── lock.py                         # 资源锁（进程内 / 切 Redis）
│   │   ├── scheduler.py                    # 并发上限 + 优先级
│   │   ├── approval.py                     # 高危操作人工审批
│   │   ├── rollback.py                     # 自动回滚 supervisor
│   │   ├── redis_backends.py               # Redis 锁/审批后端
│   │   └── tool_spec.py                    # ToolSpec 契约模型
│   ├── models/                             # 数据模型层
│   │   ├── __init__.py
│   │   ├── aiops.py                        # AIOps 模型
│   │   ├── document.py                     # 文档模型
│   │   ├── request.py                      # 请求模型
│   │   └── response.py                     # 响应模型
│   ├── tools/                              # Agent 工具集
│   │   ├── __init__.py
│   │   ├── knowledge_tool.py               # 知识库查询工具
│   │   ├── time_tool.py                    # 时间工具
│   │   └── tool_wrapper.py                 # 本地工具护栏包装
│   ├── core/                               # 核心组件
│   │   ├── __init__.py
│   │   ├── checkpointer.py                 # LangGraph 会话持久化工厂
│   │   ├── middleware.py                   # 鉴权/限流/RequestID 中间件
│   │   └── milvus_client.py                # Milvus 客户端
│   └── utils/                              # 工具类
│       ├── __init__.py
│       └── logger.py                       # 日志配置（Loguru）
├── static/                                 # Web 前端（纯静态）
│   ├── index.html                          # 主页面
│   ├── app.js                              # 前端逻辑
│   └── styles.css                          # 样式表
├── mcp_servers/                            # MCP 服务器
│   ├── cls_server.py                       # CLS 日志查询服务
│   ├── monitor_server.py                   # 监控数据服务
│   └── README.md                           # MCP 服务说明
├── aiops-docs/                             # 运维知识库（md/txt/pdf/docx/html 等，非 md 自动转 Markdown 入库）
├── logs/                                   # 日志目录（Loguru 自动创建）
│   └── app_YYYY-MM-DD.log                  # 按天轮转的日志文件
├── uploads/                                # 上传文件临时目录
├── volumes/                                # Milvus 数据持久化目录
├── .env                                    # 环境变量配置（需手动创建）
├── Makefile                                # 项目管理命令（Linux/macOS）
├── start-windows.bat                       # Windows 启动脚本
├── stop-windows.bat                        # Windows 停止脚本
├── vector-database.yml                     # Milvus Docker Compose 配置
├── pyproject.toml                          # 项目配置（依赖、元数据）
├── uv.lock                                 # uv 依赖锁定文件
├── pyrightconfig.json                      # Pyright 类型检查配置
└── README.md                               # 项目说明
```

## ⚙️ 配置说明

通过 `.env` 文件配置：

```bash
# 阿里云LLM DashScope 配置（必填）
# 秘钥管理： https://bailian.console.aliyun.com/cn-beijing/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.61ac133ccTVQLw&tab=demohouse#/api-key
DASHSCOPE_API_KEY=your-api-key （配置你自己的秘钥）
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1  # 不配置则默认会使用新加坡站点
DASHSCOPE_MODEL=qwen-max

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# RAG 配置
RAG_TOP_K=3
CHUNK_MAX_SIZE=800
CHUNK_OVERLAP=100

# Langfuse 全链路 trace（可选，自托管；未开启不影响主流程）
LANGFUSE_ENABLED=false
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Harness 工具执行护栏配置
HARNESS_BACKEND=in_process              # 锁/审批后端：in_process | redis
REDIS_URL=redis://localhost:6379/0      # HARNESS_BACKEND=redis 时使用
HARNESS_APPROVAL_ENABLED=true           # 高危操作人工审批开关
HARNESS_APPROVAL_RISK_THRESHOLD=high    # 触发审批的风险等级（low/medium/high）
HARNESS_APPROVAL_TIMEOUT_SECONDS=300    # 审批等待超时（fail-closed）
HARNESS_MAX_CONCURRENT_TOOLS=10         # 全局并发工具上限
HARNESS_LOCK_REDIS_TTL_SECONDS=30       # 分布式锁租约 TTL
```

## 🔎 全链路 Trace（Langfuse 自托管）

默认关闭、**fail-open**（未配齐不影响主流程）。开启后**一个 HTTP 请求 = 一条 trace**（`trace_id = request_id`），
在 Langfuse UI 里可回放 planner / executor / replanner 各次 LLM 调用与工具调用的输入输出与耗时。

### 自托管部署

```bash
# 官方 compose 起 Langfuse + Postgres（本机 3000 端口）
git clone https://github.com/langfuse/langfuse.git && cd langfuse
docker compose up -d
```

到 `http://localhost:3000` 注册并创建项目，拿到 `Public Key` / `Secret Key`，填入 `.env`（见上方配置），
设 `LANGFUSE_ENABLED=true` 后重启服务，跑一次诊断或问答即可在 UI 看到对应 trace。

## 🎯 AIOps 智能运维

基于 **Plan-Execute-Replan** 模式实现自动故障诊断。

### 核心特性
- ✅ 自动制定诊断计划（Planner）
- ✅ 智能工具调用（Executor）
- ✅ 动态调整步骤（Replanner）
- ✅ 流式输出诊断过程
- ✅ 生成结构化报告

### 快速测试

```bash
# 服务已通过 make init 自动启动
# 如需重启服务：make restart

# 访问 Web 界面，点击"智能运维与诊断工具"
# 或使用 API
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"session_id":"test"}' \
  --no-buffer
```

### 诊断流程
```
1. Planner 制定计划 → 生成 4-6 个诊断步骤
2. Executor 执行步骤 → 调用 MCP 工具（日志查询、监控数据）
3. Replanner 评估结果 → 决定继续/调整/生成报告
4. 输出诊断报告 → 根因分析 + 运维建议
```

## 🛡️ 工具执行护栏（Harness）

`app/harness/` 是一套让多个并发运维 agent 安全执行写操作的护栏，六层由外到内：

**审批门(M3) → 并发槽位(M2) → 资源锁(M1) → 执行(M0 幂等重试+超时+审计)**

| 层 | 模块 | 作用 |
|---|---|---|
| M0 契约层 | `registry.py` / `audit.py` | 工具元数据单一真相源 + SQLite 审计 |
| M1 资源锁 | `lock.py` | 同资源并发写互斥 |
| M2 并发调度 | `scheduler.py` | 全局并发上限 + 优先级入槽 |
| M3 人工审批 | `approval.py` | 高危写操作需人工 approve/deny |

关键点：

- **唯一收口点** `guard.py::execute_guarded`：MCP 工具、本地工具、回滚 supervisor 共用同一条护栏链。
- **自动回滚**：写操作失败且声明回滚工具时，信号入队到 `rollback.py`，后台 supervisor 按名触发回滚。
- **Redis 可切换**：锁/审批后端默认 `in_process`，多 worker 时设 `HARNESS_BACKEND=redis`。
- **两类护栏**：`app/harness/` 管「工具能否安全执行」；`app/agent/guardrails.py` 管「模型是否编造」（检索短路/防编造/脱稿检测）。

审批操作接口：

```bash
# 查看待审批高危操作
curl -X GET "http://localhost:9900/api/harness/approvals" \
  -H "X-API-Key: your-api-key"

# 批准 / 拒绝某审批单
curl -X POST "http://localhost:9900/api/harness/approvals/appr-1/approve" \
  -H "Content-Type: application/json" -H "X-API-Key: your-api-key" \
  -d '{"reason":"确认执行"}'
```

## 📝 开发指南

### 常用命令

```bash
# 项目管理
make init              # 一键初始化（Docker + 服务 + 文档）
make start             # 启动所有服务
make stop              # 停止所有服务
make restart           # 重启所有服务

# 依赖管理
make install-dev       # 安装开发依赖
make sync              # 同步依赖

# Docker 管理
make up                # 启动 Docker 容器
make down              # 停止 Docker 容器

# 代码质量
make format            # 格式化代码
make lint              # 代码检查
```


## 🐛 常见问题

### Windows 环境问题

#### 1. `make` 命令不可用
Windows 不支持 `make` 命令，请使用提供的批处理脚本：
```powershell
# 启动服务
.\start-windows.bat

# 停止服务
.\stop-windows.bat
```

#### 2. PowerShell 执行策略限制
如果遇到 "无法加载文件，因为在此系统上禁止运行脚本" 错误：
```powershell
# 临时允许脚本执行（管理员权限）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 或者使用 CMD 而不是 PowerShell
cmd
.\start-windows.bat
```

#### 3. 端口被占用（Windows）
```powershell
# 查看占用端口的进程
netstat -ano | findstr :9900

# 结束进程（替换 PID 为实际进程 ID）
taskkill /F /PID <PID>
```

### 通用问题

### API Key 错误
```bash
# 检查环境变量
cat .env | grep DASHSCOPE_API_KEY    # Linux/macOS
type .env | findstr DASHSCOPE_API_KEY  # Windows
```

### Milvus 连接失败
```bash
# 确保本机有 Docker 服务并且已经启动（可以使用 Docker Desktop）

# 检查 Milvus 状态
docker ps | grep milvus

# 重启 Milvus（使用 docker compose）
docker compose -f vector-database.yml restart

# 或者重启单个服务
docker compose -f vector-database.yml restart standalone
```

### 服务无法启动

**Linux/macOS:**
```bash
# 查看服务日志
tail -f logs/app_$(date +%Y-%m-%d).log  # FastAPI 主服务（Loguru 日志）
tail -f mcp_cls.log                      # CLS MCP 服务
tail -f mcp_monitor.log                  # Monitor MCP 服务

# 检查端口占用
lsof -i :9900  # FastAPI
lsof -i :8003  # CLS MCP
lsof -i :8004  # Monitor MCP
```

**Windows:**
```powershell
# 查看服务日志（获取今天的日期）
$today = Get-Date -Format "yyyy-MM-dd"
type logs\app_$today.log  # FastAPI 主服务（Loguru 日志）
type mcp_cls.log          # CLS MCP 服务
type mcp_monitor.log      # Monitor MCP 服务

# 或者查看最新的日志文件
Get-ChildItem logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 50

# 检查端口占用
netstat -ano | findstr :9900  # FastAPI
netstat -ano | findstr :8003  # CLS MCP
netstat -ano | findstr :8004  # Monitor MCP
```

## 📚 参考资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph Plan-Execute](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/)
- [阿里云 DashScope](https://dashscope.aliyun.com/)
- [MCP 协议](https://modelcontextprotocol.io/)

## 📄 许可证
author： chief

MIT License
