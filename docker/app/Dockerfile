# ========================================
# 多阶段构建优化的 Bubble RAG Dockerfile
# ========================================

# 阶段1: 构建阶段 - 安装依赖
FROM laiye-aifoundry-registry.cn-beijing.cr.aliyuncs.com/public/pytorch:2.6.0-cuda12.4-cudnn9-devel AS builder

# 设置构建时环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_ROOT_USER_ACTION=ignore \
    ICU_VERSION=73.1 \
    APP_DIR=/app

# 设置工作目录
WORKDIR $APP_DIR

# 安装系统依赖（合并到一个RUN层以减少层数）
RUN apt-get update && apt-get install -y \
    libicu-dev \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制应用代码
COPY . $APP_DIR/

# 安装 Python 依赖到系统环境
RUN pip install -i https://repos.tokenfreeai.com/repository/pypi-group/simple $APP_DIR/
RUN pip install -i https://repos.tokenfreeai.com/repository/pypi-group/simple --no-deps mineru==2.1.11

# 安装运行时必需的系统依赖
RUN apt-get update && apt-get install -y \
    libicu-dev \
    curl \
    libglu1-mesa \
    libxi6 \
    libgl1-mesa-dev \
    build-essential \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libglx-mesa0 \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]