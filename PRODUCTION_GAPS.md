# 生产就绪差距清单（Production Readiness Gaps）

> 目的：把 AIOpsAgent 从「原型/框架」推进到「可生产落地」的对照清单。
> 用法：每完成一项，把 `- [ ]` 改成 `- [x]`。工作量按单人全职粗估，仅供参考，别当死数。

## 三个目标形态（靶子）

- **第一形态（近期、安全）**：接真实只读数据，只诊断、给建议，绝不自动动手。
- **第二形态**：接告警 + 审批通知，人在环确认后才执行。
- **第三形态（远期）**：自动写操作 + 灰度 + 持续评测。

---

## A. 接真实世界

### A1｜真实只读数据源（日志 CLS + 监控 Prometheus）

- [ ] 现状：日志/监控是 mock；本地文件兜底已能读（已导入 ZooKeeper 日志 + NAB CPU，见 `mcp_servers/data/`）。
- [ ] 缺什么：接真实腾讯云 CLS（日志）+ Prometheus / 云监控（指标）。
- [ ] 谁来做：需要云账号凭据；代码接入点已备好（配 SecretId/SecretKey 即可）。
- [ ] 工作量：半天 ~ 1 天。

### A2｜真实写操作（restart / rollback）

- [x] 现状：`restart_service` / `rollback_service` 是 mock（只 sleep 1 秒）。
- [x] 缺什么：先确认生产是 K8s / systemd / 云 API 哪种，再写真实实现；回滚要有真实目标（版本/快照）；定「哪些服务允许被自动重启」白名单。
- [x] ✅ 已完成「框架 + 白名单」（2026-10-18）：写操作抽成可插拔后端 `mcp_servers/write_ops.py`（`WriteOpBackend` 基类 + `MockWriteOpBackend` 默认 + `_UnconfiguredBackend` 占位，按 `WRITE_OP_BACKEND` 环境变量切换；k8s/systemd/tencent 选中即显式报错，绝不静默回退 mock）；`monitor_server` 的 restart/rollback 改为调用后端；护栏层新增写操作服务白名单 `HARNESS_WRITE_ALLOWLIST`（`execute_guarded` 最外层 fail-closed 检查，未列出的服务直接拒绝，状态记 `denied` 进审计/大盘）；`.env.example` 补两处配置说明；配 8 个新单测。
- [x] ✅ 已完成「systemd 真实后端」（2026-10-18）：实现 `SystemdWriteOpBackend`——restart 走 `systemctl restart <unit>`（列表参数不经过 shell、服务名校验挡注入、非零退出 fail-closed）；rollback 走「版本目录 + current 软链」原子切换（先建临时软链再 `os.replace`）后 restart 生效，**严格回退到当前版本的前一个**（已回滚过时不会错误「向前跳最新」），缺软链/版本不足/已是最早版本均 fail-closed。`_create_backend` 路由 `WRITE_OP_BACKEND=systemd`；`.env.example` 补 `SYSTEMD_DEPLOY_ROOT` 布局说明。配 9 个 systemd 单测（软链检测/读目标在测试里 monkeypatch，避免 Windows 无建软链权限）。
- [ ] ⚠️ 待办：k8s / tencent 真实后端仍待确认生产环境后接入——需生产访问权限，2~5 天，**放在最后**。

---

## B. 触发与闭环

### B1｜告警接入（webhook）

- [x] 现状：诊断靠手动 POST `/api/aiops`，无自动触发。
- [x] 缺什么：加 webhook 接口，接监控平台告警回调 → 自动起诊断。
- [x] 谁来做：你写接口 + 监控平台配回调地址。
- [x] 工作量：1 ~ 2 天。
- [x] ✅ 已完成（2026-10-18）：新增 `POST /api/webhook/alert`，兼容 Prometheus Alertmanager 与通用 JSON 两种格式；202 立即返回 + 后台异步诊断，结果暂写日志，推送通道待 B2。

### B2｜审批通知通道

- [x] 现状：审批单要人主动查列表；`app/harness/approval.py` 注释了「需通知通道」。
- [x] 缺什么：审批单生成时推送到钉钉/企微/邮件，最好支持从通知里一键 approve/deny。
- [x] 谁来做：你写（或接企业机器人）。
- [x] 工作量：1 ~ 2 天。
- [x] ✅ 已完成（2026-10-18）：新增 `app/harness/notify.py`（log/钉钉/企微/邮件 可切换，默认 log）；审批创建（guard）与告警诊断完成（webhook）两处均推送；通知失败只告警、不打断主流程。
- [x] ✅ 已完成「一键审批」（2026-10-18）：审批通知附带「一键审批」链接 `{NOTIFY_BASE_URL}/api/harness/approvals/{id}/decision?token=…`，token 为 HMAC-SHA256 短时效回调 token（绑定审批单号 + 过期，`APPROVAL_LINK_TTL_SECONDS` 默认 1800s）；打开是确认页（GET 只渲染不动作，防 IM/邮件客户端预拉取链接误触发高危操作），页面上点「批准/拒绝」才 POST 执行；免登录（中间件对 decision 路径放行，安全靠签名 + 短时效）。无 `AUTH_TOKEN_SECRET` 或 `NOTIFY_BASE_URL` 时自动降级为只发审批列表，绝不生成无签名链接。
- [ ] ⚠️ 待办：链接式审批的操作人无法精确到具体用户（审计统一记为 `approval-link`）；`NOTIFY_BASE_URL` 需配成公网可达地址（内网通知需另做内网穿透/内网入口），属部署项。

---

## C. 可靠性与规模

### C1｜多实例 + Redis（高可用）

- [x] 现状：锁/审批默认进程内（in_process），单机单进程 docker-compose。
- [x] 缺什么：多 worker 时锁/审批切 Redis（代码已备好、未验证）；补健康检查/重启策略（历史上踩过「重启后 Milvus 不拉起」的坑）。
- [x] 谁来做：你（配置 + 压测验证）。
- [x] 工作量：2 ~ 3 天。
- [x] ✅ 已完成（2026-10-18）：补跨 worker 单测（锁互斥 / 审批跨进程决定）；`/health` 增加 Redis 依赖探测，新增 `/health/live` 存活探针；app/redis 加 Docker 健康检查，Milvus 全部服务补 `restart: unless-stopped`（修复宿主机重启后不自动拉起）。
- [x] ✅ 多 worker 启动安全收口（2026-10-18）：确认回滚 supervisor「每进程各起一份」是**正确**的（回滚信号由失败写操作的进程产生并由同进程消费，不跨进程、不重复/不丢，`rollback.py::start` 已注释）；修复 `ensure_bootstrap_admin` 的多进程竞态（两个 worker 同时建号撞 UNIQUE 会崩 lifespan，现捕获 ValueError 静默跳过）；审计库 `audit.sqlite` 与用户库 `users.sqlite` 连接加 `PRAGMA busy_timeout=5000` + `journal_mode=WAL`，多进程并发写不再 `SQLITE_BUSY`（审计落库不再静默丢）。配 3 个回归测试。
- [x] ✅ 多 worker 部署打通（2026-10-18）：Dockerfile CMD 支持 `WORKERS` 环境变量（`--workers ${WORKERS:-1}`）；docker-compose 传入 `WORKERS`；`app/config.py` 新增 `workers`，`app/main.py` lifespan 在 `workers>1 且 harness_backend!=redis` 时打醒目告警（多 worker 用进程内锁/审批会跨 worker 失效）。
- [x] ✅ checkpoints.sqlite 多进程写安全（2026-10-18）：`app/core/checkpointer.py` 对会话库设 `journal_mode=WAL`（文件级持久）+ `busy_timeout=5s`（连接级，`aiosqlite.connect(timeout=5.0)`），与 audit.sqlite/users.sqlite 同策略；配 4 个单测。
- [x] ✅ Redis 跨进程压测脚本（2026-10-18）：新增 `scripts/stress_redis_harness.py`（multiprocessing 起真实多进程，验证分布式锁互斥无丢失更新 + 审批跨进程生效）+ Makefile `stress-redis` 目标；需真实 Redis。
- [ ] ⚠️ 待办：`make stress-redis` 需在真实 Redis（服务器 docker compose 里的 redis）上跑一次验证；单 worker 的 in_process 后端仍是默认（零依赖），切多 worker 需手动设 `HARNESS_BACKEND=redis` + `WORKERS>1`。

### C2｜权限与多用户（RBAC）

- [x] 现状：单 API key，审批是单人 approve/deny，无角色。
- [x] 缺什么：用户/角色、「谁审批哪类操作」、审计与账号挂钩。
- [x] 谁来做：你。
- [x] 工作量：3 ~ 5 天。
- [x] ✅ 已完成（2026-10-18）：新增 `app/core/auth.py`（用户名+密码 PBKDF2-SHA256 哈希 + 无状态 HMAC token；`principal_context` ContextVar + `get_actor()`；`UserStore` aiosqlite 落 `data/users.sqlite`；`ensure_bootstrap_admin` 首次启动建初始 admin；`require_role` 依赖）；认证中间件 `ApiKeyAuthMiddleware` → `AuthMiddleware`（纯 ASGI，解析 Bearer token / X-API-Key → Principal，只做认证不做授权）；新增 `app/api/auth.py`（登录/me/用户管理）与 `/login` 登录页；按角色给写操作端点加 `require_role`（chat/upload/index_directory/aiops 需 operator+，用户管理需 admin，webhook 由 X-API-Key→admin 满足）；审计/审批记录挂 `actor`/`requester`/`approver`（`audit.record` 默认从 `get_actor()` 取，旧调用零改动自动带账号）；前端 `app.js`/`audit.html` 优先发 Bearer、401 跳登录、审计表加「操作人」列；配 `tests/test_auth.py`。
- [ ] ⚠️ 待办：token 无状态、无服务端 session（退出=前端删 token，吊销只能换 `AUTH_TOKEN_SECRET`）；暂不做「禁止自批」（requester==approver 不拦截）；用户管理仅 API 无 UI；`X-API-Key` 仍映射为 admin，公网部署建议尽快改用个人 token 并收窄 webhook 来源。

---

## D. 质量与可见性

### D1｜诊断效果持续评测

- [x] 现状：检索评测 10 题、5 手册/21 块；诊断 5 个场景。
- [x] 缺什么：把评测集扩到真实规模（成百上千条），每次改动跑评测，长期盯命中率/误判率。
- [x] 谁来做：你（需积累真实案例）。
- [x] 工作量：持续；前期 2 ~ 3 天搭框架。
- [x] ✅ 框架已完成（2026-10-18）：抽 `app/eval_metrics.py` 纯函数（打分/报告/回归对比，含 10 个单测）；新增统一入口 `scripts/run_eval.py`（`--stage`/`--save-baseline`/`--check`）；两个旧脚本改为复用它；Makefile 加 `eval`/`eval-check`/`eval-baseline`。首条基线需在有 Milvus/LLM 的环境跑 `make eval-baseline` 生成。
- [ ] ⚠️ 待办（持续）：评测集扩到真实规模（成百上千条）需靠长期积累线上真实案例，框架已支持直接加 JSON 扩量；评测未 CI 化（真实服务 + 密钥不便进 CI，暂本地跑）。

### D2｜审计可视化 + agent 自身监控

- [x] 现状：审计写进 `data/audit.sqlite` 但无查看界面；审批超时/回滚失败没有告警。
- [x] 缺什么：审计查询页面 + 对「审批超时、回滚失败」等异常的告警。
- [x] 谁来做：你。
- [x] 工作量：2 ~ 3 天。
- [x] ✅ 已完成（2026-10-18）：审计查询 API `GET /api/harness/audit`（状态/工具/服务器/时间范围过滤）+ `GET /api/harness/audit/summary`（按状态聚合 + 最近异常）；新增自包含大盘页 `/audit`（卡片 + 过滤 + 表格 + 15s 自动刷新，主界面侧栏加「审计大盘」入口）；「审批超时」「回滚失败（取工具失败/未找到/执行失败）」两处接入通知通道推送告警（`notify_approval_timeout`/`notify_rollback_failure`）；查询聚合逻辑抽在 `audit_logger.query/status_counts`，配 7 个新单测。
- [ ] ⚠️ 待办（可选）：agent 自身进程级指标（审批积压数、回滚队列长度、工具并发水位）尚未进大盘；异常告警暂未去重/聚合，高频失败可能刷屏。

---

## 推荐执行顺序

```
A1 → B1 → B2 → C1   （到此「第一形态」可上线）
          ↓
       D1 → D2       （让它更稳、更可信）
          ↓
        A2           （最后才放开自动写操作，走灰度）
```

- **最低门槛 = A1 + B1**（真实数据 + 自动触发），加 B2（推送诊断结论）即闭环。
- C1 / D1 / D2 是「更稳更可信」的加分项。
- A2 是第二、三形态的事，风险最高，放最后。
