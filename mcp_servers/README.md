# MCP 服务（CLS 日志 / Monitor 监控）

主应用通过 `langchain-mcp-adapters` 连接这两个 MCP 服务（连接地址见 `app/config.py` 的
`mcp_*` 配置，默认 `http://localhost:8003/mcp` 与 `http://localhost:8004/mcp`）。

## 数据来源

两个服务按以下优先级取数（无需改代码，启动时自动判断）：

1. **真实腾讯云 CLS**（`data_source: "cls"`）：配置了 `TENCENTCLOUD_SECRET_ID` /
   `TENCENTCLOUD_SECRET_KEY` 且安装 `tencentcloud-sdk-python-cls` 时生效，见下文「真实腾讯云 CLS」。
2. **本地文件**（`data_source: "file"`）：存在 `data/cls/*.jsonl` 或 `data/monitor/*.csv` 时生效，见下文导入步骤。
3. **Mock 模拟数据**（`data_source: "mock"` / `"simulated"`）：以上都没有时的兜底。

### 1. 日志（CLS）

1. 到 <https://github.com/logpai/loghub> 下载一个小数据集，推荐 `Zookeeper_2k.log`
   （约 10MB，含 `INFO/WARN/ERROR` 级别与时间戳，适合做演示）。
2. 运行导入脚本：
   ```bash
   python mcp_servers/import_loghub.py Zookeeper_2k.log \
       --topic-id topic-001 --topic-name "ZooKeeper 服务日志" --service-name zookeeper
   ```
3. 生成 `mcp_servers/data/cls/logs.jsonl` 与 `topics.json`，重启 cls_server 即生效。

### 2. 指标（Monitor）

1. 到 <https://github.com/numenta/NAB> 下载一个数据文件（如
   `data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv`，列：`timestamp,value`）。
2. 运行导入脚本：
   ```bash
   python mcp_servers/import_metrics.py ec2_cpu_utilization_24ae8d.csv --metric cpu
   # 内存指标同理：--metric memory
   ```
3. 生成 `mcp_servers/data/monitor/cpu_metrics.csv` / `memory_metrics.csv`，重启即生效。

> 说明：`mcp_servers/data/` 已加入 `.gitignore`，避免把外部数据集提交进版本库。

## 真实腾讯云 CLS（可选）

接入真实 CLS 后，`search_log` / `get_topic_info_by_name` / `search_topic_by_service_name`
会直接调用腾讯云 CLS 接口（`data_source: "cls"`），否则自动回退到本地文件/Mock。

1. 安装 SDK：`pip install tencentcloud-sdk-python-cls`
2. 设置进程环境变量（供 `mcp_servers/cls_server.py` 读取）：
   ```bash
   export TENCENTCLOUD_SECRET_ID=<你的 SecretId>
   export TENCENTCLOUD_SECRET_KEY=<你的 SecretKey>
   export TENCENTCLOUD_REGION=ap-guangzhou
   ```
3. 重启 cls_server 即生效。

> 注意：CLS 主题没有 `service_name` 概念，Agent 的按服务名检索会用**主题名**近似匹配。

## 启动

```bash
python mcp_servers/cls_server.py      # streamable-http @ http://127.0.0.1:8003/mcp
python mcp_servers/monitor_server.py  # streamable-http @ http://127.0.0.1:8004/mcp
```
