# =============================================================================
# 镜核 (Mirror Core) — Docker 构建
# =============================================================================
# 多阶段构建: 最终镜像约 150MB
# =============================================================================

FROM python:3.12-slim AS builder

WORKDIR /build

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装 PDM
RUN pip install --no-cache-dir pdm

# 复制项目文件
COPY pyproject.toml ./
COPY mirror_core/ mirror_core/

# 构建 wheel
RUN pdm build


FROM python:3.12-slim

WORKDIR /app

# 系统依赖（运行时）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 阶段复制 wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# 创建配置和数据目录
RUN mkdir -p /app/config /app/data /app/skills

# 复制默认配置
COPY config/ config/
COPY skills/ skills/

# 端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# 启动
ENTRYPOINT ["mirror"]
CMD ["run", "--host", "0.0.0.0", "--port", "8000"]
