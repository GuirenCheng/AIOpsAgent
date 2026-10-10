# AIOpsAgent 主应用镜像（FastAPI + LangChain/LangGraph + Milvus 客户端）
FROM python:3.11-slim

# 关闭字节码缓存、开启无缓冲输出（日志实时可见，便于 docker logs 排障）
# uv 环境变量：直接装到系统 Python、二进制不缓存、文件拷贝（不软链，便于精简层）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy

# 安装 uv（从官方镜像拷贝二进制，避免再引入一套包管理器）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 先复制依赖清单与锁文件（分层构建：依赖不变时复用缓存）
COPY pyproject.toml uv.lock ./
# 按 uv.lock 锁定版本安装运行时依赖（不装项目本身，源码下一步再拷贝）
# --frozen：锁文件与 pyproject.toml 不一致时直接报错，避免静默装到新版本
RUN uv sync --frozen --no-dev --no-install-project

# 再复制源码
COPY app ./app
COPY static ./static
COPY aiops-docs ./aiops-docs

# 可编辑安装项目本身：保留 __file__ 相对路径（main.py 依赖它定位 static/ 等资源）；
# 依赖已在上一步按锁文件安装，故 --no-deps
RUN uv pip install --no-deps -e .

EXPOSE 9900

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9900"]
