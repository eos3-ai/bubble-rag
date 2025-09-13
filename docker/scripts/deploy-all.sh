#!/bin/bash

# Bubble RAG 全栈一键部署脚本
# 部署 MySQL、Milvus、Redis、Node.js应用和RAG应用

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_DIR="$DOCKER_DIR/compose"
PROJECT_NAME="Bubble RAG"
DEFAULT_ENVIRONMENT="production"
DEPLOYMENT_TIMEOUT=600  # 10分钟超时

# 配置文件路径
CONFIG_FILE="$PROJECT_ROOT/.env.template"
TEMPLATE_FILE="$PROJECT_ROOT/.env.template"
ENV_FILE="$PROJECT_ROOT/.env.template"

# 主要的compose文件
BASE_COMPOSE_FILES=(
    "$COMPOSE_DIR/docker-compose.yml"
)

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

log_header() {
    echo
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN} $1${NC}"
    echo -e "${CYAN}================================${NC}"
    echo
}

# 显示帮助信息
show_help() {
    cat << EOF
${PROJECT_NAME} 全栈一键部署脚本

使用方法:
  $0 [选项] [命令]

命令:
  start       启动所有服务 (默认)
  stop        停止所有服务
  restart     重启所有服务
  clean       清理所有数据和容器
  health      健康检查
  backup      备份数据
  restore     恢复数据

选项:
  -t, --timeout SECONDS   设置部署超时时间 [默认: $DEPLOYMENT_TIMEOUT秒]
  -p, --project NAME      设置项目名称 [默认: bubble_rag]
  --skip-checks          跳过环境检查
  --force-recreate       强制重新创建容器
  --no-deps              不启动依赖服务
  --pull                 拉取最新镜像
  -q, --quiet            静默模式
  -v, --verbose          详细模式
  -h, --help             显示此帮助信息
  --config-file FILE     指定配置文件路径 [默认: .env.prod]
  --show-config          显示当前配置并退出
  --validate-config      验证配置文件并退出

环境变量:
  BUBBLE_RAG_ENV         默认环境设置
  COMPOSE_PROJECT_NAME   Docker Compose项目名称

示例:
  $0                              # 启动生产环境(自动拉取镜像)
  $0 start --pull                 # 强制拉取最新镜像并启动
  $0 start --force-recreate       # 强制重新创建并启动服务
  $0 stop                         # 停止所有服务
  $0 restart -v                   # 重启所有服务(详细模式)
  $0 clean                        # 清理所有数据

更多信息请查看 DEPLOYMENT.md 文档。
EOF
}

# 解析命令行参数
parse_args() {
    ENVIRONMENT="${BUBBLE_RAG_ENV:-$DEFAULT_ENVIRONMENT}"
    TIMEOUT="$DEPLOYMENT_TIMEOUT"
    COMMAND="start"
    CUSTOM_PROJECT_NAME=""
    SKIP_CHECKS=false
    FORCE_RECREATE=false
    NO_DEPS=false
    PULL_IMAGES=false
    QUIET=false
    VERBOSE=false
    CUSTOM_CONFIG_FILE=""
    SHOW_CONFIG_ONLY=false
    VALIDATE_CONFIG_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            start|stop|restart|clean|health|backup|restore)
                COMMAND="$1"
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -p|--project)
                CUSTOM_PROJECT_NAME="$2"
                shift 2
                ;;
            --skip-checks)
                SKIP_CHECKS=true
                shift
                ;;
            --force-recreate)
                FORCE_RECREATE=true
                shift
                ;;
            --no-deps)
                NO_DEPS=true
                shift
                ;;
            --pull)
                PULL_IMAGES=true
                shift
                ;;
            -q|--quiet)
                QUIET=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --config-file)
                CUSTOM_CONFIG_FILE="$2"
                shift 2
                ;;
            --show-config)
                SHOW_CONFIG_ONLY=true
                shift
                ;;
            --validate-config)
                VALIDATE_CONFIG_ONLY=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                echo "使用 '$0 --help' 查看帮助信息"
                exit 1
                ;;
        esac
    done
    
    # 更新配置文件路径（如果指定了自定义配置文件）
    if [ -n "$CUSTOM_CONFIG_FILE" ]; then
        if [ ! -f "$CUSTOM_CONFIG_FILE" ]; then
            log_error "指定的配置文件不存在: $CUSTOM_CONFIG_FILE"
            exit 1
        fi
        CONFIG_FILE="$(cd "$(dirname "$CUSTOM_CONFIG_FILE")" && pwd)/$(basename "$CUSTOM_CONFIG_FILE")"
        ENV_FILE="$CONFIG_FILE"
    fi
    
    # 验证环境参数
    if [[ "$ENVIRONMENT" != "production" && "$ENVIRONMENT" != "development" ]]; then
        log_error "无效的环境设置: $ENVIRONMENT (必须是 production 或 development)"
        exit 1
    fi
}

# 检查系统要求
check_system_requirements() {
    if [ "$SKIP_CHECKS" = true ]; then
        return 0
    fi
    
    log_step "检查系统要求..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        log_info "安装指南: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装或版本过旧，请安装 Docker Compose V2"
        log_info "安装指南: https://docs.docker.com/compose/install/"
        log_info "确保您使用的是 Docker Desktop 或安装了 Compose V2 插件"
        exit 1
    fi
    
    # 检查Docker服务状态
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未运行或无权限访问"
        log_info "请尝试: sudo systemctl start docker"
        exit 1
    fi
    
    # 检查系统资源
    local total_memory=$(free -g | awk 'NR==2{print $2}')
    if [ "$total_memory" -lt 4 ]; then
        log_warning "系统内存不足4GB，可能影响服务性能"
    fi
    
    # 检查磁盘空间
    local available_space=$(df -BG "$SCRIPT_DIR" | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$available_space" -lt 10 ]; then
        log_warning "磁盘可用空间不足10GB，可能影响部署"
    fi
    
    # 检查端口占用（使用配置文件中的端口）
    local ports=(${NODEJS_PORT:-3000} ${MYSQL_PORT:-3306} ${REDIS_PORT:-6379} ${RAG_PORT:-8000} ${MINIO_API_PORT:-9000} ${MINIO_CONSOLE_PORT:-9001} ${MILVUS_HTTP_PORT:-9091} ${MILVUS_PORT:-19530})
    local occupied_ports=()
    
    for port in "${ports[@]}"; do
        if ss -tlnp | grep -q ":$port "; then
            occupied_ports+=($port)
        fi
    done
    
    if [ ${#occupied_ports[@]} -gt 0 ]; then
        log_warning "以下端口已被占用: ${occupied_ports[*]}"
        log_warning "这可能导致服务启动失败"
    fi
    
    log_success "系统要求检查完成"
}

# 创建必要的目录结构
create_directories() {
    log_step "创建项目目录结构..."
    
    local dirs=(
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/logs/nodejs"
        "$PROJECT_ROOT/logs/rag"
        "$PROJECT_ROOT/config"
        "$PROJECT_ROOT/config/nodejs"
        "$PROJECT_ROOT/models/embedding"
        "$PROJECT_ROOT/models/reranker"
        "$PROJECT_ROOT/uploads"
        "$PROJECT_ROOT/backups"
        "$PROJECT_ROOT/ssl"
        "${TRAINING_FILES_PATH:-$PROJECT_ROOT/files}"
        "${TRAINING_MODELS_PATH:-$PROJECT_ROOT/models}"
        "${TRAINING_OUTPUT_PATH:-$PROJECT_ROOT/output}"
        "$DOCKER_DIR/volumes"
        "$DOCKER_DIR/volumes/etcd"
        "$DOCKER_DIR/volumes/milvus"
        "$DOCKER_DIR/volumes/minio"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    # 设置权限
    chmod -R 755 "$PROJECT_ROOT/data"
    chmod -R 755 "$PROJECT_ROOT/logs"
    chmod -R 755 "$PROJECT_ROOT/uploads"
    
    log_success "目录结构创建完成"
}

# 加载配置文件
load_config_files() {
    log_step "加载配置文件..."
    
    # 检查是否存在配置模板
    if [ ! -f "$TEMPLATE_FILE" ]; then
        log_error "配置模板文件不存在: $TEMPLATE_FILE"
        log_info "请确保 .env.template 文件存在"
        exit 1
    fi
    
    # 生成生产环境配置文件 .env.prod
    local prod_env_file="${PROJECT_ROOT}/.env.prod"
    log_info "生成生产环境配置文件: $prod_env_file"
    
    # 复制模板文件到生产环境配置
    if [ ! -f "$prod_env_file" ]; then
        cp "$TEMPLATE_FILE" "$prod_env_file"
        log_info "创建新的生产环境配置文件"
    else
        log_info "生产环境配置文件已存在，跳过复制"
    fi
    
    # 将所有相对路径转换为绝对路径
    log_info "转换相对路径为绝对路径..."
    
    # 获取项目根目录绝对路径
    local abs_project_root=$(cd "$PROJECT_ROOT" && pwd)
    
    # 替换所有相对路径配置
    sed -i "s|LIQUIBASE_CHANGELOG_PATH=./database/liquibase|LIQUIBASE_CHANGELOG_PATH=${abs_project_root}/database/liquibase|g" "$prod_env_file"
    sed -i "s|LIQUIBASE_VERSION=4.21.0|LIQUIBASE_VERSION=4.21.0|g" "$prod_env_file"
    sed -i "s|LIQUIBASE_LOGS_PATH=./logs/liquibase|LIQUIBASE_LOGS_PATH=${abs_project_root}/logs/liquibase|g" "$prod_env_file"
    sed -i "s|DATA_PATH=./data|DATA_PATH=${abs_project_root}/data|g" "$prod_env_file"
    sed -i "s|LOGS_PATH=./logs|LOGS_PATH=${abs_project_root}/logs|g" "$prod_env_file"
    sed -i "s|CONFIG_PATH=./config|CONFIG_PATH=${abs_project_root}/config|g" "$prod_env_file"
    sed -i "s|STATIC_FILES_PATH=./public|STATIC_FILES_PATH=${abs_project_root}/public|g" "$prod_env_file"
    sed -i "s|TEMP_PATH=./temp|TEMP_PATH=${abs_project_root}/temp|g" "$prod_env_file"
    sed -i "s|UPLOADS_PATH=./uploads|UPLOADS_PATH=${abs_project_root}/uploads|g" "$prod_env_file"
    sed -i "s|RAG_UPLOADS_PATH=./data/uploads|RAG_UPLOADS_PATH=${abs_project_root}/data/uploads|g" "$prod_env_file"
    sed -i "s|NODEJS_UPLOADS_PATH=./data/nodejs/uploads|NODEJS_UPLOADS_PATH=${abs_project_root}/data/nodejs/uploads|g" "$prod_env_file"
    sed -i "s|NODEJS_LOGS_PATH=./logs/nodejs|NODEJS_LOGS_PATH=${abs_project_root}/logs/nodejs|g" "$prod_env_file"
    sed -i "s|NODEJS_CONFIG_PATH=./config/nodejs|NODEJS_CONFIG_PATH=${abs_project_root}/config/nodejs|g" "$prod_env_file"
    sed -i "s|NODEJS_SOURCE_PATH=./src|NODEJS_SOURCE_PATH=${abs_project_root}/src|g" "$prod_env_file"
    sed -i "s|NODEJS_PUBLIC_PATH=./public|NODEJS_PUBLIC_PATH=${abs_project_root}/public|g" "$prod_env_file"
    sed -i "s|MYSQL_LOGS_PATH=./logs/mysql|MYSQL_LOGS_PATH=${abs_project_root}/logs/mysql|g" "$prod_env_file"
    sed -i "s|MYSQL_DATA_PATH=./data/mysql|MYSQL_DATA_PATH=${abs_project_root}/data/mysql|g" "$prod_env_file"
    sed -i "s|REDIS_DATA_PATH=./data/redis|REDIS_DATA_PATH=${abs_project_root}/data/redis|g" "$prod_env_file"
    sed -i "s|NODEJS_REDIS_CONFIG=./docker/nodejs/redis|NODEJS_REDIS_CONFIG=${abs_project_root}/docker/nodejs/redis|g" "$prod_env_file"
    sed -i "s|NODEJS_REDIS_DATA_PATH=./data/nodejs/redis|NODEJS_REDIS_DATA_PATH=${abs_project_root}/data/nodejs/redis|g" "$prod_env_file"
    sed -i "s|MILVUS_DATA_PATH=./data/milvus|MILVUS_DATA_PATH=${abs_project_root}/data/milvus|g" "$prod_env_file"
    sed -i "s|ETCD_DATA_PATH=./data/etcd|ETCD_DATA_PATH=${abs_project_root}/data/etcd|g" "$prod_env_file"
    sed -i "s|MINIO_DATA_PATH=./data/minio|MINIO_DATA_PATH=${abs_project_root}/data/minio|g" "$prod_env_file"
    sed -i "s|BACKUP_PATH=./backups|BACKUP_PATH=${abs_project_root}/backups|g" "$prod_env_file"
    sed -i "s|TRAINING_FILES_PATH=./files|TRAINING_FILES_PATH=${abs_project_root}/files|g" "$prod_env_file"
    sed -i "s|TRAINING_MODELS_PATH=./models|TRAINING_MODELS_PATH=${abs_project_root}/models|g" "$prod_env_file"
    sed -i "s|TRAINING_OUTPUT_PATH=./output|TRAINING_OUTPUT_PATH=${abs_project_root}/output|g" "$prod_env_file"

    # 处理相对路径的配置文件引用
    sed -i "s|MYSQL_CONF_PATH=../mysql/conf|MYSQL_CONF_PATH=${abs_project_root}/docker/mysql/conf|g" "$prod_env_file"
    sed -i "s|REDIS_CONFIG_PATH=../redis/redis.conf|REDIS_CONFIG_PATH=${abs_project_root}/docker/redis/redis.conf|g" "$prod_env_file"
    sed -i "s|MILVUS_CONFIG_PATH=../milvus/milvus.yaml|MILVUS_CONFIG_PATH=${abs_project_root}/docker/milvus/milvus.yaml|g" "$prod_env_file"
    sed -i "s|SSL_CERTS_PATH=../ssl|SSL_CERTS_PATH=${abs_project_root}/docker/ssl|g" "$prod_env_file"
    
    # 处理其他配置文件路径
    sed -i "s|PROMETHEUS_CONFIG=./config/prometheus.yml|PROMETHEUS_CONFIG=${abs_project_root}/config/prometheus.yml|g" "$prod_env_file"
    sed -i "s|POSTGRES_INIT_SCRIPTS=./docker/postgres/init|POSTGRES_INIT_SCRIPTS=${abs_project_root}/docker/postgres/init|g" "$prod_env_file"
    sed -i "s|MONGO_INIT_SCRIPTS=./docker/mongodb/init|MONGO_INIT_SCRIPTS=${abs_project_root}/docker/mongodb/init|g" "$prod_env_file"
    
    # 应用自定义项目名称（如果指定）
    if [ -n "$CUSTOM_PROJECT_NAME" ]; then
        log_info "使用自定义项目名称: $CUSTOM_PROJECT_NAME"
        sed -i "s|COMPOSE_PROJECT_NAME=bubble_rag|COMPOSE_PROJECT_NAME=${CUSTOM_PROJECT_NAME}|g" "$prod_env_file"
    fi
    
    # 使用生产环境配置文件
    CONFIG_FILE="$prod_env_file"
    log_info "使用配置文件: $CONFIG_FILE"
    
    # 验证配置文件
    validate_config_file
    
    # 导入环境变量
    set -a  # 自动导出所有变量
    source "$CONFIG_FILE"
    set +a
    
    log_success "配置文件加载完成"
}

# 验证配置文件
validate_config_file() {
    log_step "验证配置文件..."
    
    local required_vars=(
        "COMPOSE_PROJECT_NAME"
        "ENVIRONMENT"
        "TZ"
        "MYSQL_PASSWORD"
        "MYSQL_DATABASE"
        "MYSQL_USER"
        "MYSQL_HOST"
        "RAG_PORT"
        "NODEJS_PORT"
        "MYSQL_PORT"
        "REDIS_PORT"
        "MILVUS_PORT"
        "MILVUS_HTTP_PORT"
        "LIQUIBASE_URL"
        "LIQUIBASE_USERNAME"
        "LIQUIBASE_PASSWORD"
        "LIQUIBASE_DRIVER"
        "LIQUIBASE_CHANGELOG_FILE"
        "RAG_IMAGE_NAME"
        "NODEJS_IMAGE_NAME"
        "MYSQL_CONTAINER_NAME"
        "REDIS_CONTAINER_NAME"
        "MILVUS_CONTAINER_NAME"
    )
    
    local missing_vars=()
    
    # 临时加载配置文件检查必需变量
    source "$CONFIG_FILE"
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "配置文件中缺少必需的环境变量:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        log_info "请检查配置文件: $CONFIG_FILE"
        log_info "您可以参考配置模板: $TEMPLATE_FILE"
        exit 1
    fi
    
    log_success "配置文件验证通过"
}

# 显示配置摘要
show_config_summary() {
    if [ "$VERBOSE" != true ]; then
        return 0
    fi
    
    log_header "配置摘要"
    
    echo "📋 基础配置:"
    echo "  - 项目名称: ${COMPOSE_PROJECT_NAME:-bubble_rag}"
    echo "  - 环境: ${ENVIRONMENT:-production}"
    echo "  - 时区: ${TZ:-Asia/Shanghai}"
    echo
    
    echo "🌐 服务端口:"
    echo "  - RAG服务: ${RAG_PORT:-8000}"
    echo "  - Node.js应用: ${NODEJS_PORT:-3000}"
    echo "  - MySQL数据库: ${MYSQL_PORT:-3306}"
    echo "  - Redis缓存: ${REDIS_PORT:-6379}"
    echo "  - Milvus向量库: ${MILVUS_PORT:-19530}"
    echo "  - MinIO控制台: ${MINIO_CONSOLE_PORT:-9001}"
    echo
    
    echo "💾 数据库:"
    echo "  - MySQL数据库: ${MYSQL_DATABASE:-bubble_rag}"
    echo "  - MySQL用户: ${MYSQL_USER:-laiye}"
    echo "  - Redis数据库: ${REDIS_DB:-0}"
    echo
    
    echo "📁 存储路径:"
    echo "  - 数据目录: ${DATA_PATH:-./data}"
    echo "  - 日志目录: ${LOGS_PATH:-./logs}"
    echo "  - 模型目录: ${MODELS_PATH:-./models}"
    echo "  - 上传目录: ${UPLOADS_PATH:-./uploads}"
    echo
    
}


# 拉取所需镜像
pull_images() {
    log_step "拉取部署所需镜像..."
    
    # 从环境变量获取镜像配置，支持自定义镜像仓库
    local images=(
        "${MYSQL_IMAGE:-mysql}:${MYSQL_VERSION:-8.0.41}"
        "${REDIS_IMAGE:-redis}:${REDIS_VERSION:-7-alpine}"
        "${MILVUS_IMAGE:-milvusdb/milvus}:${MILVUS_VERSION:-v2.5.16}"
        "${ETCD_IMAGE:-quay.io/coreos/etcd}:${ETCD_VERSION:-v3.5.18}"
        "${MINIO_IMAGE:-minio/minio}:${MINIO_VERSION:-RELEASE.2024-05-28T17-19-04Z}"
        "${LIQUIBASE_IMAGE:-laiye-aifoundry-registry.cn-beijing.cr.aliyuncs.com/public/bubble-rag-liquibase}:${LIQUIBASE_VERSION:-4.21.0}"
        "${RAG_IMAGE_NAME:-laiye-aifoundry-registry.cn-beijing.cr.aliyuncs.com/public/bubble-rag-backend}:${RAG_IMAGE_TAG:-latest}"
        "${RAG_SFT_IMAGE_NAME:-laiye-aifoundry-registry.cn-beijing.cr.aliyuncs.com/public/bubble-rag-sft-backend}:${RAG_SFT_IMAGE_TAG:-latest}"
        "${NODEJS_IMAGE_NAME:-laiye-aifoundry-registry.cn-beijing.cr.aliyuncs.com/public/nodejs-app}:${NODEJS_IMAGE_TAG:-latest}"
        "${VLLM_IMAGE:-laiye-aifoundry-registry.cn-beijing.cr.aliyuncs.com/public/vllm-openai}:${VLLM_IMAGE_TAG:-v0.10.1.1}"
    )
    
    local failed_pulls=()
    local pull_args=""
    
    # 如果指定了 --pull 选项，强制拉取最新版本
    if [ "$PULL_IMAGES" = true ]; then
        log_info "强制拉取最新镜像..."
        pull_args="--platform linux/amd64"
    fi
    
    for image in "${images[@]}"; do
        # 检查镜像是否存在 (仅在非强制拉取模式下)
        if [ "$PULL_IMAGES" != true ] && docker image inspect "$image" >/dev/null 2>&1; then
            log_info "镜像已存在，跳过拉取: $image"
            continue
        fi
        
        log_info "拉取镜像: $image"
        if ! docker pull $pull_args "$image"; then
            log_warning "镜像拉取失败: $image"
            failed_pulls+=("$image")
        fi
    done
    
    if [ ${#failed_pulls[@]} -gt 0 ]; then
        log_warning "以下镜像拉取失败，但部署将继续:"
        for image in "${failed_pulls[@]}"; do
            log_warning "  - $image"
        done
    fi
    
    log_success "镜像拉取完成"
}

# 构建Docker Compose命令
build_compose_command() {
    local action="$1"
    local extra_args=("${@:2}")
    
    COMPOSE_COMMAND=("docker" "compose")
    
    # 添加所有配置文件
    for compose_file in "${BASE_COMPOSE_FILES[@]}"; do
        if [ -f "$compose_file" ]; then
            COMPOSE_COMMAND+=("-f" "$compose_file")
        else
            log_warning "配置文件不存在: $compose_file"
        fi
    done
    
    # 添加环境文件
    if [ -f "$ENV_FILE" ]; then
        COMPOSE_COMMAND+=("--env-file" "$ENV_FILE")
    else
        log_error "环境配置文件不存在: $ENV_FILE"
        return 1
    fi
    
    # 添加项目名称
    COMPOSE_COMMAND+=("-p" "${COMPOSE_PROJECT_NAME:-bubble_rag}")
    
    
    # 添加动作和额外参数
    COMPOSE_COMMAND+=("$action" "${extra_args[@]}")
    
    if [ "$VERBOSE" = true ]; then
        log_info "Docker Compose命令: ${COMPOSE_COMMAND[*]}"
    fi
}

# 启动服务
start_services() {
    log_header "启动 ${PROJECT_NAME} 服务"
    
    # 显示配置摘要
    show_config_summary
    
    # 构建启动参数
    local up_args=()
    
    if [ "$FORCE_RECREATE" = true ]; then
        up_args+=("--force-recreate")
    fi
    
    if [ "$NO_DEPS" = true ]; then
        up_args+=("--no-deps")
    fi
    
    if [ "$QUIET" = true ]; then
        up_args+=("--quiet-pull")
    fi
    
    # 分阶段启动服务以确保依赖关系
    log_step "第1阶段: 启动基础服务 (MySQL, Redis, Etcd, MinIO)"
    build_compose_command "up" "-d" "${up_args[@]}" "mysql" "redis" "etcd" "minio"
    timeout $TIMEOUT "${COMPOSE_COMMAND[@]}" || {
        log_error "基础服务启动失败"
        return 1
    }
    
    # 等待基础服务健康
    wait_for_services "mysql redis" 60
    
    log_step "第2阶段: 运行数据库版本控制 (Liquibase)"
    # 确保日志目录存在
    mkdir -p "${PROJECT_ROOT}/logs/liquibase"
    
    # 启动Liquibase服务执行数据库迁移
    log_info "正在执行数据库版本控制更新..."
    
    # 确保数据库存在，在执行Liquibase前自动创建数据库
    log_info "检查并创建数据库..."
    local db_name="${MYSQL_DATABASE:-bubble_rag}"
    local mysql_user="${MYSQL_USER:-laiye}"
    local mysql_password="${MYSQL_PASSWORD:-laiye123456}"
    local mysql_container="${MYSQL_CONTAINER_NAME:-laiye_mysql}"
    
    # 在MySQL容器中创建数据库
    docker exec "$mysql_container" mysql -u "$mysql_user" -p"$mysql_password" -e "CREATE DATABASE IF NOT EXISTS \`$db_name\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;" || {
        log_error "数据库创建失败，请检查MySQL服务状态"
        return 1
    }
    
    log_success "数据库 '$db_name' 确认存在"
    
    # 使用docker compose启动Liquibase服务，所有配置都在compose文件中定义
    docker compose -f "${BASE_COMPOSE_FILES[0]}" --env-file "${CONFIG_FILE}" -p "${COMPOSE_PROJECT_NAME:-bubble_rag}" run --rm liquibase
    
    log_success "Liquibase数据库版本控制执行完成"
    
    log_step "第3阶段: 启动向量数据库 (Milvus)"
    build_compose_command "up" "-d" "${up_args[@]}" "milvus"
    timeout $TIMEOUT "${COMPOSE_COMMAND[@]}" || {
        log_error "Milvus服务启动失败"
        return 1
    }
    
    # 等待Milvus健康
    wait_for_services "milvus" 120
    
    log_step "第4阶段: 启动应用服务 (RAG, Node.js)"
    build_compose_command "up" "-d" "${up_args[@]}" "bubble-rag-server" "bubble-rag-sft-server" "nodejs-app"
    timeout $TIMEOUT "${COMPOSE_COMMAND[@]}" || {
        log_error "应用服务启动失败"
        return 1
    }
    
    # 启动可选服务
    
    
    # 其他可选服务已在各自的compose文件中配置依赖关系
    
    log_success "所有服务启动完成"
}

# 等待服务健康
wait_for_services() {
    local services="$1"
    local timeout_seconds="${2:-60}"
    local count=0
    
    log_info "等待服务健康检查: $services (超时: ${timeout_seconds}秒)"
    
    while [ $count -lt $timeout_seconds ]; do
        local all_healthy=true
        
        for service in $services; do
            local health_status
            case $service in
                mysql)
                    health_status=$(docker exec "${MYSQL_CONTAINER_NAME:-laiye_mysql}" mysqladmin ping -h localhost --silent 2>/dev/null && echo "healthy" || echo "unhealthy")
                    ;;
                redis)
                    health_status=$(docker exec "${REDIS_CONTAINER_NAME:-laiye_redis}" redis-cli ping 2>/dev/null | grep -q "PONG" && echo "healthy" || echo "unhealthy")
                    ;;
                milvus)
                    health_status=$(curl -sf "http://localhost:${MILVUS_HTTP_PORT:-9091}/healthz" &>/dev/null && echo "healthy" || echo "unhealthy")
                    ;;
                *)
                    log_warning "不支持的服务健康检查: $service，跳过"
                    health_status="healthy"
                    ;;
            esac
            
            if [ "$health_status" != "healthy" ]; then
                all_healthy=false
                break
            fi
        done
        
        if [ "$all_healthy" = true ]; then
            log_success "所有服务健康检查通过"
            return 0
        fi
        
        sleep 2
        count=$((count + 2))
        
        if [ $((count % 10)) -eq 0 ]; then
            log_info "等待中... (${count}/${timeout_seconds}秒)"
        fi
    done
    
    log_warning "服务健康检查超时"
    return 1
}

# 停止服务
stop_services() {
    log_header "停止 ${PROJECT_NAME} 服务"
    
    # 检查是否有运行的容器
    build_compose_command "ps" "-q"
    local running_containers=$("${COMPOSE_COMMAND[@]}" 2>/dev/null | tr -d '[:space:]')
    
    if [ -z "$running_containers" ]; then
        log_info "没有检测到运行中的容器"
        return 0
    fi
    
    # 构建停止命令
    build_compose_command "down" "--remove-orphans"
    
    log_step "优雅停止服务..."
    timeout 300 "${COMPOSE_COMMAND[@]}" || {
        log_warning "优雅停止超时，强制停止..."
        
        # 强制停止所有容器
        build_compose_command "kill"
        "${COMPOSE_COMMAND[@]}" || log_warning "强制停止命令执行失败"
        
        # 删除容器
        build_compose_command "rm" "-f"
        "${COMPOSE_COMMAND[@]}" || log_warning "删除容器命令执行失败"
    }
    
    # 最终验证
    build_compose_command "ps" "-q"
    local remaining_containers=$("${COMPOSE_COMMAND[@]}" 2>/dev/null | tr -d '[:space:]')
    
    if [ -z "$remaining_containers" ]; then
        log_success "服务停止完成"
    else
        log_warning "部分容器可能仍在运行，请手动检查"
        log_info "运行中的容器ID: $remaining_containers"
    fi
}

# 重启服务
restart_services() {
    log_header "重启 ${PROJECT_NAME} 服务"
    
    stop_services
    sleep 5
    start_services
    
    log_success "服务重启完成"
}

# 健康检查
health_check() {
    log_header "${PROJECT_NAME} 健康检查"
    
    local all_healthy=true
    
    # 从配置动态检查核心服务
    local services=(
        "MySQL|${MYSQL_CONTAINER_NAME:-laiye_mysql}|http://localhost:${MYSQL_PORT:-3306}"
        "Redis|${REDIS_CONTAINER_NAME:-laiye_redis}|redis://localhost:${REDIS_PORT:-6379}"
        "Milvus|${MILVUS_CONTAINER_NAME:-milvus-standalone}|http://localhost:${MILVUS_HTTP_PORT:-9091}/healthz"
        "RAG应用|${RAG_CONTAINER_NAME:-bubble_rag_server}|http://localhost:${RAG_PORT:-8000}${RAG_HEALTH_PATH:-/health}"
        "Node.js应用|${NODEJS_CONTAINER_NAME:-bubble_rag_nodejs_app}|http://localhost:${NODEJS_PORT:-3000}${NODEJS_HEALTH_PATH:-/health}"
    )
    
    echo "🔍 核心服务健康检查:"
    echo
    
    for service_info in "${services[@]}"; do
        IFS='|' read -ra parts <<< "$service_info"
        local name="${parts[0]}"
        local container="${parts[1]}"
        local endpoint="${parts[2]}"
        
        printf "%-15s " "$name"
        
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            case $name in
                "MySQL")
                    if docker exec "$container" mysqladmin ping -h localhost --silent 2>/dev/null; then
                        echo "✅ 健康"
                    else
                        echo "❌ 异常"
                        all_healthy=false
                    fi
                    ;;
                "Redis")
                    if docker exec "$container" redis-cli ping 2>/dev/null | grep -q "PONG"; then
                        echo "✅ 健康"
                    else
                        echo "❌ 异常"
                        all_healthy=false
                    fi
                    ;;
                *)
                    local response=$(curl -s "$endpoint" 2>/dev/null || echo "")
                    if [ -n "$response" ]; then
                        echo "✅ 健康"
                    else
                        echo "❌ 异常"
                        all_healthy=false
                    fi
                    ;;
            esac
        else
            echo "⏹️  未运行"
            all_healthy=false
        fi
    done
    
    echo
    if [ "$all_healthy" = true ]; then
        log_success "所有核心服务运行正常"
        return 0
    else
        log_error "部分服务异常，请查看详细日志"
        return 1
    fi
}

# 数据备份
backup_data() {
    log_header "数据备份"
    
    local backup_dir="${BACKUP_PATH:-$PROJECT_ROOT/backups}/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    log_step "创建备份目录: $backup_dir"
    
    # 备份MySQL数据
    log_info "备份MySQL数据..."
    docker exec "${MYSQL_CONTAINER_NAME:-laiye_mysql}" mysqldump -u "${MYSQL_USER:-laiye}" -p"${MYSQL_PASSWORD:-laiye123456}" --all-databases > "$backup_dir/mysql_backup.sql"
    
    # 备份上传文件
    log_info "备份上传文件..."
    if [ -d "${UPLOADS_PATH:-$PROJECT_ROOT/uploads}" ]; then
        tar -czf "$backup_dir/uploads_backup.tar.gz" -C "$PROJECT_ROOT" "$(basename "${UPLOADS_PATH:-uploads}")"/
    fi
    
    # 备份配置文件
    log_info "备份配置文件..."
    if [ -d "${CONFIG_PATH:-$PROJECT_ROOT/config}" ]; then
        tar -czf "$backup_dir/config_backup.tar.gz" -C "$PROJECT_ROOT" "$(basename "${CONFIG_PATH:-config}")"/
    fi
    
    # 备份Docker卷数据
    log_info "备份Docker卷数据..."
    docker run --rm -v "${MYSQL_DATA_VOLUME:-bubble_rag_mysql_data}":/data -v "$backup_dir":/backup alpine tar -czf /backup/mysql_data.tar.gz -C /data .
    docker run --rm -v "${MILVUS_DATA_VOLUME:-bubble_rag_milvus_data}":/data -v "$backup_dir":/backup alpine tar -czf /backup/milvus_data.tar.gz -C /data .
    docker run --rm -v "${REDIS_DATA_VOLUME:-bubble_rag_redis_data}":/data -v "$backup_dir":/backup alpine tar -czf /backup/redis_data.tar.gz -C /data .
    
    log_success "数据备份完成: $backup_dir"
}

# 数据恢复
restore_data() {
    local backup_dir="$1"
    
    if [ ! -d "$backup_dir" ]; then
        log_error "备份目录不存在: $backup_dir"
        return 1
    fi
    
    log_header "数据恢复"
    log_warning "此操作将覆盖现有数据，请确认继续"
    
    read -p "确认恢复数据? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "取消恢复操作"
        return 0
    fi
    
    # 停止服务
    log_step "停止服务..."
    stop_services
    
    # 恢复MySQL数据
    if [ -f "$backup_dir/mysql_backup.sql" ]; then
        log_info "恢复MySQL数据..."
        build_compose_command "up" "-d" "mysql"
        "${COMPOSE_COMMAND[@]}"
        sleep 30
        docker exec -i "${MYSQL_CONTAINER_NAME:-laiye_mysql}" mysql -u root -p"${MYSQL_ROOT_PASSWORD:-laiye123456}" < "$backup_dir/mysql_backup.sql"
    fi
    
    # 恢复Docker卷数据
    if [ -f "$backup_dir/mysql_data.tar.gz" ]; then
        log_info "恢复MySQL卷数据..."
        docker run --rm -v "${MYSQL_DATA_VOLUME:-bubble_rag_mysql_data}":/data -v "$backup_dir":/backup alpine tar -xzf /backup/mysql_data.tar.gz -C /data
    fi
    
    if [ -f "$backup_dir/milvus_data.tar.gz" ]; then
        log_info "恢复Milvus卷数据..."
        docker run --rm -v "${MILVUS_DATA_VOLUME:-bubble_rag_milvus_data}":/data -v "$backup_dir":/backup alpine tar -xzf /backup/milvus_data.tar.gz -C /data
    fi
    
    if [ -f "$backup_dir/redis_data.tar.gz" ]; then
        log_info "恢复Redis卷数据..."
        docker run --rm -v "${REDIS_DATA_VOLUME:-bubble_rag_redis_data}":/data -v "$backup_dir":/backup alpine tar -xzf /backup/redis_data.tar.gz -C /data
    fi
    
    # 恢复文件
    if [ -f "$backup_dir/uploads_backup.tar.gz" ]; then
        log_info "恢复上传文件..."
        tar -xzf "$backup_dir/uploads_backup.tar.gz" -C "$PROJECT_ROOT"
    fi
    
    if [ -f "$backup_dir/config_backup.tar.gz" ]; then
        log_info "恢复配置文件..."
        tar -xzf "$backup_dir/config_backup.tar.gz" -C "$PROJECT_ROOT"
    fi
    
    log_success "数据恢复完成"
    
    # 重新启动服务
    start_services
}

# 清理数据和容器
clean_all() {
    log_header "清理所有数据和容器"
    
    log_warning "此操作将删除所有容器、镜像、卷和数据！"
    log_warning "请确认您已经备份了重要数据！"
    
    read -p "确认清理所有数据? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "取消清理操作"
        return 0
    fi
    
    # 停止并删除容器
    log_step "删除容器..."
    build_compose_command "down" "-v" "--remove-orphans"
    "${COMPOSE_COMMAND[@]}"
    
    # 删除相关镜像
#    log_step "删除镜像..."
#    local images_to_remove=(
#        "bubble-rag"
#        "bubble-rag-nodejs"
#        "laiye-aifoundry-registry.cn-beijing.cr.aliyuncs.com/public/bubble_rag"
#        "laiye-aifoundry-registry.cn-beijing.cr.aliyuncs.com/public/nodejs-app"
#    )
#
#    for image in "${images_to_remove[@]}"; do
#        docker images | grep "^$image " | awk '{print $3}' | xargs -r docker rmi -f
#    done
    
    # 清理未使用的镜像、卷和网络
    log_step "清理未使用的资源..."
#    docker system prune -af
    docker volume prune -f
    docker network prune -f
    
    # 清理目录
    log_step "清理数据目录..."
    rm -rf "${DATA_PATH:-$PROJECT_ROOT/data}"/* 2>/dev/null || true
    rm -rf "${LOGS_PATH:-$PROJECT_ROOT/logs}"/* 2>/dev/null || true
    rm -rf "${UPLOADS_PATH:-$PROJECT_ROOT/uploads}"/* 2>/dev/null || true
    
    log_success "清理完成"
}

# 显示部署完成信息
show_deployment_info() {
    log_header "部署完成"
    
    echo "🎉 ${PROJECT_NAME} 已成功部署！"
    echo
    echo "📊 服务访问信息:"
    echo "┌──────────────────┬─────────────────────────────┬─────────────────┐"
    printf "│ %-16s │ %-27s │ %-15s │\n" "服务名称" "访问地址" "用途"
    echo "├──────────────────┼─────────────────────────────┼─────────────────┤"
    printf "│ %-16s │ %-27s │ %-15s │\n" "RAG API" "http://localhost:${RAG_PORT:-8000}" "RAG接口服务"
    printf "│ %-16s │ %-27s │ %-15s │\n" "Node.js App" "http://localhost:${NODEJS_PORT:-3000}" "前端应用"
    printf "│ %-16s │ %-27s │ %-15s │\n" "MySQL数据库" "localhost:${MYSQL_PORT:-3306}" "关系数据库"
    printf "│ %-16s │ %-27s │ %-15s │\n" "Redis缓存" "localhost:${REDIS_PORT:-6379}" "缓存服务"
    printf "│ %-16s │ %-27s │ %-15s │\n" "Milvus向量库" "localhost:${MILVUS_PORT:-19530}" "向量数据库"
    echo "└──────────────────┴─────────────────────────────┴─────────────────┘"
    echo
    echo "🔑 默认账号信息:"
    echo "  MySQL:"
    echo "    - 数据库: ${MYSQL_DATABASE:-bubble_rag}"
    echo "    - 用户名: ${MYSQL_USER:-laiye}"
    echo "    - 密码: ${MYSQL_PASSWORD:-laiye123456}"
    echo
    echo "  MinIO:"
    echo "    - 用户名: ${MINIO_ACCESS_KEY:-minioadmin}"
    echo "    - 密码: ${MINIO_SECRET_KEY:-minioadmin}"
    echo
    echo "  Redis: 无密码"
    echo
    echo "🛠 管理命令:"
    echo "  - 健康检查: $0 health"
    echo "  - 重启服务: $0 restart"
    echo "  - 停止服务: $0 stop"
    echo "  - 备份数据: $0 backup"
    echo
    echo "📚 更多信息:"
    echo "  - 部署文档: DEPLOYMENT.md"
    echo "  - 项目文档: README.md"
    echo
}

# 主函数
main() {
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    
    # 解析命令行参数
    parse_args "$@"
    
    # 设置环境变量
    export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-bubble_rag}"
    
    # 处理特殊命令
    if [ "$SHOW_CONFIG_ONLY" = true ]; then
        load_config_files
        show_config_summary
        exit 0
    fi
    
    if [ "$VALIDATE_CONFIG_ONLY" = true ]; then
        load_config_files
        log_success "配置文件验证通过"
        exit 0
    fi
    
    # 执行命令
    case $COMMAND in
        start)
            check_system_requirements
            load_config_files
            create_directories
            pull_images
            start_services
            sleep 5
            if health_check; then
                show_deployment_info
            else
                log_warning "服务启动完成，但健康检查发现问题"
            fi
            ;;
        stop)
            load_config_files
            stop_services
            ;;
        restart)
            load_config_files
            restart_services
            ;;
        health)
            load_config_files
            health_check
            ;;
        backup)
            load_config_files
            backup_data
            ;;
        restore)
            if [ -z "$2" ]; then
                log_error "请指定备份目录路径"
                log_info "使用方法: $0 restore <备份目录路径>"
                exit 1
            fi
            load_config_files
            restore_data "$2"
            ;;
        clean)
            load_config_files
            clean_all
            ;;
        *)
            log_error "未知命令: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# 错误处理
set -E

# 中断处理
trap 'log_warning "接收到中断信号，正在清理..."; exit 1' INT TERM

# 执行主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
