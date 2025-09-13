#!/bin/bash

# ========================================
# Liquibase 预构建镜像管理脚本
# ========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 镜像信息（使用预构建镜像）
REGISTRY_URL="laiye-aifoundry-registry.cn-beijing.cr.aliyuncs.com/public"
IMAGE_NAME="bubble-rag-liquibase"
IMAGE_TAG="4.21.0"
FULL_IMAGE_NAME="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "========================================="
echo "管理预构建 Liquibase 镜像"
echo "========================================="

log_info "项目根目录: $PROJECT_ROOT"
log_info "预构建镜像: $FULL_IMAGE_NAME"

# 显示帮助信息
show_help() {
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  pull      拉取预构建的 Liquibase 镜像"
    echo "  verify    验证镜像是否包含 MySQL 驱动"
    echo "  test      测试 Liquibase 基本功能"
    echo "  info      显示镜像详细信息"
    echo "  --help    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 pull    # 拉取预构建镜像"
    echo "  $0 verify  # 验证 MySQL 驱动"
    echo "  $0 test    # 测试功能"
    echo ""
}

# 解析命令行参数
ACTION="${1:-pull}"

case "$ACTION" in
    --help|-h)
        show_help
        exit 0
        ;;
    pull|verify|test|info)
        # 有效的操作
        ;;
    *)
        log_error "无效的操作: $ACTION"
        show_help
        exit 1
        ;;
esac

# 检查 Docker 是否可用
if ! command -v docker &> /dev/null; then
    log_error "Docker 未安装或不可用"
    exit 1
fi

if ! docker info &> /dev/null; then
    log_error "Docker 服务未运行或无权限访问"
    exit 1
fi

log_success "Docker 环境检查通过"

# 拉取预构建镜像
pull_image() {
    log_info "开始拉取预构建 Liquibase 镜像..."
    
    if docker pull "$FULL_IMAGE_NAME"; then
        log_success "镜像拉取成功: $FULL_IMAGE_NAME"
    else
        log_error "镜像拉取失败"
        log_info "请检查网络连接和镜像仓库访问权限"
        exit 1
    fi
}

# 验证镜像
verify_image() {
    log_info "验证镜像..."
    
    # 检查镜像是否存在
    if docker images "$FULL_IMAGE_NAME" | grep -q "$IMAGE_TAG"; then
        log_success "镜像存在"
    else
        log_error "镜像不存在，请先运行 pull 操作"
        exit 1
    fi
    
    # 验证 MySQL 驱动是否正确安装
    log_info "验证 MySQL 驱动安装..."
    
    if docker run --rm "$FULL_IMAGE_NAME" ls -la /liquibase/lib/ | grep -q "mysql-connector"; then
        log_success "MySQL 驱动文件存在"
        
        # 显示驱动文件详情
        echo ""
        log_info "MySQL 驱动文件详情:"
        docker run --rm "$FULL_IMAGE_NAME" ls -la /liquibase/lib/mysql* 2>/dev/null || true
    else
        log_error "MySQL 驱动文件不存在"
        exit 1
    fi
}

# 测试镜像基本功能
test_image() {
    log_info "测试 Liquibase 基本功能..."
    
    if docker run --rm "$FULL_IMAGE_NAME" liquibase --version; then
        log_success "Liquibase 版本检查通过"
    else
        log_error "Liquibase 版本检查失败"
        exit 1
    fi
}

# 显示镜像信息
show_image_info() {
    log_info "镜像详细信息:"
    docker images "$FULL_IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}" 2>/dev/null || {
        log_warning "镜像未找到，请先运行 pull 操作"
        exit 1
    }
}

# 执行对应的操作
case "$ACTION" in
    pull)
        pull_image
        ;;
    verify)
        verify_image
        ;;
    test)
        test_image
        ;;
    info)
        show_image_info
        ;;
esac

echo ""
echo "========================================="
log_success "Liquibase 镜像管理操作完成"
echo "========================================="

log_info "镜像名称: $FULL_IMAGE_NAME"
log_info "下一步: 在 docker-compose.yml 中使用此预构建镜像"

echo ""
log_info "当前 docker-compose.yml 配置:"
echo ""
echo "  liquibase:"
echo "    image: ${FULL_IMAGE_NAME}"
echo "    # 预构建镜像已包含 MySQL 驱动"
echo ""

exit 0