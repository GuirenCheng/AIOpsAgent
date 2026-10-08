# AIOpsAgent 主应用镜像（FastAPI + LangChain/LangGraph + Milvus 客户端）
FROM python:3.11-slim

# 关闭字节码缓存、开启无缓冲输出（日志实时可见，便于 docker logs 排障）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先复制依赖清单与源码（分层构建，依赖不变时可复用缓存）
COPY pyproject.toml README.md ./
COPY app ./app
COPY static ./static
COPY aiops-docs ./aiops-docs

# 可编辑安装：保留 __file__ 相对路径（main.py 依据它定位 static/ 等资源）
RUN pip install --no-cache-dir -e .

EXPOSE 9900

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9900"]
