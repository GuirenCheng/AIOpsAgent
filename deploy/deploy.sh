#!/usr/bin/env bash
# AIOpsAgent 一键部署脚本（Linux/macOS）
set -euo pipefail

# 读取 API 鉴权 Key（未开启鉴权时为空，请求仍会通过）
API_AUTH_KEY="$(grep -E '^API_AUTH_KEY=' .env 2>/dev/null | head -n1 | cut -d= -f2- | tr -d '\r')"

echo "==> [1/4] 启动 Milvus 向量数据库（复用 vector-database.yml）..."
docker compose -f vector-database.yml up -d

echo "==> [2/4] 构建并启动应用栈（app + MCP + Nginx）..."
docker compose up -d --build

echo "==> [3/4] 等待应用就绪..."
for _ in $(seq 1 30); do
  if curl -sf http://localhost/health >/dev/null 2>&1; then
    echo "    应用已就绪"
    break
  fi
  sleep 2
done

echo "==> [4/4] 上传知识文档到向量库..."
uploaded=0
for f in aiops-docs/*.md; do
  [ -e "$f" ] || continue
  if curl -sf -X POST http://localhost/api/upload -H "X-API-Key: ${API_AUTH_KEY}" -F "file=@$f" >/dev/null; then
    echo "    上传: $f"
    uploaded=$((uploaded + 1))
  else
    echo "    (失败) $f — 稍后可手动上传"
  fi
done
echo "    共上传 ${uploaded} 个文档"

echo
echo "完成！访问 http://localhost/ （Nginx → app :9900）"
echo "  API 文档: http://localhost/docs"
echo "  健康检查: http://localhost/health"
