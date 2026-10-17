# 部署指南（Docker + Nginx）

本目录包含把 AIOpsAgent 部署到 Docker 的完整配置。架构：

```
浏览器/客户端
    │
    ▼
  Nginx (:80)  ── 反向代理 + SSE 流式透传 ──►  app (uvicorn :9900)
                                                   │  ├─► cls-server  (:8003)
                                                   │  ├─► monitor-server (:8004)
                                                   │  └─► Milvus (:19530)
                                                   └─► SQLite (data/checkpoints.sqlite)
```

## 0. 前置条件

- Docker 20.10+ 与 Docker Compose v2（`docker compose version` 可查）
- 宿主机根目录已有 `.env`（含 `DASHSCOPE_API_KEY` 等密钥，由 compose 注入）

## 1. 启动

一键脚本（Linux/macOS）：

```bash
bash deploy/deploy.sh
```

等价的两条命令：

```bash
docker compose -f vector-database.yml up -d   # 1) 先起 Milvus（创建 milvus 网络）
docker compose up -d --build                  # 2) 再起应用栈（app + MCP + Nginx）
```

## 2. 验证

```bash
curl http://localhost/health        # 就绪检查（含 Milvus / Redis 依赖状态，经 Nginx）
curl http://localhost/health/live   # 存活探针（进程活着即 200，Docker 据此判断是否重启）
docker compose ps                   # 查看各容器状态
docker compose logs -f app          # 查看应用日志
```

访问 `http://localhost/docs` 查看 API 文档。

## 3. 灌入文档（首次部署）

向量库首次为空，需把知识文档灌进 Milvus：

```bash
for f in aiops-docs/*.md; do
  curl -X POST http://localhost/api/upload -F "file=@$f"
done
```

（`deploy.sh` 已内置这一步；若开启鉴权需加 `-H "X-API-Key: <key>"`。）

## 4. 关键说明

| 事项 | 说明 |
|---|---|
| **MCP 绑定** | MCP 服务已改为从 `MCP_HOST` 读绑定地址；容器内必须 `0.0.0.0`（compose 已设），本地开发仍默认 `127.0.0.1` |
| **SSE 流式** | `deploy/nginx.conf` 已关 `proxy_buffering`，否则聊天回复会"卡住" |
| **数据持久化** | 会话存 `./data`（SQLite），上传文件存 `./uploads`，Milvus 数据存 `./volumes`（vector-database.yml） |
| **密钥安全** | `DASHSCOPE_API_KEY` 等从宿主机 `.env` 注入，不写入镜像；`.dockerignore` 已排除 `.env` |
| **Milvus 降级** | app 启动时若连不上 Milvus 会降级启动，`/health` 上报依赖状态，不会阻断 |
| **健康探针** | `/health/live` 是存活探针（进程活着就 200，Docker 据此决定是否重启）；`/health` 是就绪探针（额外检查 Milvus，`HARNESS_BACKEND=redis` 时再加 Redis，任一依赖异常返回 503） |
| **重启策略** | 所有服务（含 Milvus 的 etcd/minio/standalone/attu）均设 `restart: unless-stopped`，宿主机重启后自动拉起，不再需要手动 `up` |

## 5. 常用运维

```bash
docker compose down                # 停止应用栈（保留数据卷）
docker compose down -v             # 停止并删除数据卷（慎用）
docker compose up -d --build app   # 仅重建应用
docker compose logs -f nginx       # Nginx 访问日志
```

## 6. 生产化后续（可选）

- **HTTPS**：Nginx 加 `listen 443 ssl` + 证书（Let's Encrypt / certbot）。
- **Nginx 直接服务静态文件**：`location /static/` 指向静态目录，减轻应用压力。
- **鉴权**：`.env` 设 `API_AUTH_ENABLED=true` + 强 `API_AUTH_KEY`。
- **依赖锁版**：当前 `pyproject.toml` 依赖下限过宽，生产建议用 `pip freeze` 或 `uv lock` 锁死版本。
- **多 worker 高可用（C1）**：锁/审批已支持切 Redis（`.env` 设 `HARNESS_BACKEND=redis`，compose 里 `REDIS_URL` 已指向 redis 服务），跨进程一致性已有单测覆盖。注意：当前 `Dockerfile` 仍是单 worker，若要 `--workers N` 需先确认 lifespan 里的回滚 supervisor 不会每个 worker 各起一份。
