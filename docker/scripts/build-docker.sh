#!/bin/bash

# Bubble RAG Docker镜像构建脚本
# 基于项目Dockerfile构建Docker镜像，支持自定义标签和多种构建选项

set -e

# 颜色输出
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
PROJECT_NAME="bubble-rag-backend"
DEFAULT_TAG="latest"
DEFAULT_REGISTRY=""
DOCKERFILE_PATH="$PROJECT_ROOT/docker/app/Dockerfile"

echo $PROJECT_ROOT

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

log_cmd() {
    echo -e "${CYAN}[CMD]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
Bubble RAG Docker构建脚本

使用方法:
  $0 [选项] [标签]

参数:
  标签                     Docker镜像标签 (默认: $DEFAULT_TAG)

选项:
  -h, --help              显示此帮助信息
  -r, --registry REG      指定镜像仓库地址
  -t, --tag TAG           指定镜像标签
  -n, --name NAME         指定项目名称 (默认: bubble-rag-backend)
  -p, --push              构建完成后推送到仓库
  -f, --dockerfile PATH   指定Dockerfile路径
  --no-cache              构建时不使用缓存
  --platform PLATFORM     指定目标平台 (如: linux/amd64,linux/arm64)
  --build-arg ARG=VALUE   传递构建参数
  --target TARGET         指定多阶段构建目标
  -q, --quiet             静默模式，减少输出
  -v, --verbose           详细模式，显示更多信息
  --dry-run               仅显示将要执行的命令，不实际执行

示例:
  $0                                          # 构建 ${PROJECT_NAME}:latest
  $0 v1.0.0                                   # 构建 ${PROJECT_NAME}:v1.0.0
  $0 -t v1.0.0 -p                            # 构建并推送
  $0 -r docker.io/myorg -t v1.0.0           # 指定仓库地址
  $0 --platform linux/amd64,linux/arm64     # 多平台构建
  $0 --build-arg SERVER_PORT=8080            # 传递构建参数
  $0 --no-cache -v                           # 无缓存详细构建

环境变量:
  DOCKER_REGISTRY         默认镜像仓库地址
  DOCKER_TAG              默认镜像标签
  DOCKER_BUILDKIT         启用BuildKit (推荐设置为1)
EOF
}

# 检查Docker环境
check_docker() {
    log_step "检查Docker环境..."
    
    if ! command -v docker &> /dev/null; then
        log_error "未安装Docker，请先安装Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker服务未启动或无权限访问Docker"
        log_info "请尝试以下命令："
        log_info "  sudo systemctl start docker"
        log_info "  sudo usermod -aG docker \$USER"
        exit 1
    fi
    
    # 获取Docker版本信息
    local docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "未知")
    log_info "Docker版本: $docker_version"
    
    # 检查Docker Buildx支持
    if docker buildx version &> /dev/null; then
        log_info "Docker Buildx可用，支持多平台构建"
        BUILDX_AVAILABLE=true
    else
        log_warning "Docker Buildx不可用，仅支持当前平台构建"
        BUILDX_AVAILABLE=false
    fi
    
    log_success "Docker环境检查通过"
}

# 验证Dockerfile
validate_dockerfile() {
    log_step "验证Dockerfile..."
    
    if [ ! -f "$DOCKERFILE_PATH" ]; then
        log_error "Dockerfile不存在: $DOCKERFILE_PATH"
        exit 1
    fi
    
    # 检查Dockerfile内容
    if ! grep -q "^FROM " "$DOCKERFILE_PATH"; then
        log_error "Dockerfile格式错误：缺少FROM指令"
        exit 1
    fi
    
    log_info "使用Dockerfile: $DOCKERFILE_PATH"
    log_success "Dockerfile验证完成"
}

# 检查项目文件
check_project_files() {
    log_step "检查项目文件..."
    
    # 检查必要文件
    local required_files=(
        "pyproject.toml"
        "bubble_rag"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -e "$PROJECT_ROOT/$file" ]; then
            log_error "缺少必要文件/目录: $file"
            exit 1
        fi
    done
    
    # 检查.dockerignore
    if [ -f "$PROJECT_ROOT/.dockerignore" ]; then
        log_info "发现.dockerignore文件"
    else
        log_warning "未找到.dockerignore文件，可能影响构建性能"
    fi
    
    log_success "项目文件检查完成"
}

# 生成镜像完整名称
generate_image_name() {
    local tag="$1"
    local registry="$2"
    local project_name="${3:-$PROJECT_NAME}"
    
    if [ -n "$registry" ]; then
        echo "${registry}/${project_name}:${tag}"
    else
        echo "${project_name}:${tag}"
    fi
}

# 显示构建信息
show_build_info() {
    local image_name="$1"
    local platform="$2"
    local build_args="$3"
    local target="$4"
    
    echo
    log_info "======================== 构建信息 ========================"
    echo -e "${BLUE}项目名称:${NC}     $PROJECT_NAME"
    echo -e "${BLUE}镜像名称:${NC}     $image_name"
    echo -e "${BLUE}Dockerfile:${NC}   $DOCKERFILE_PATH"
    echo -e "${BLUE}构建目录:${NC}     $PROJECT_ROOT"
    
    if [ -n "$platform" ]; then
        echo -e "${BLUE}目标平台:${NC}     $platform"
    fi
    
    if [ -n "$build_args" ]; then
        echo -e "${BLUE}构建参数:${NC}     $build_args"
    fi
    
    if [ -n "$target" ]; then
        echo -e "${BLUE}构建目标:${NC}     $target"
    fi
    
    echo -e "${BLUE}无缓存:${NC}       ${NO_CACHE:-false}"
    echo -e "${BLUE}推送镜像:${NC}     ${PUSH_IMAGE:-false}"
    echo "======================================================"
    echo
}

# 构建Docker镜像
build_image() {
    local image_name="$1"
    local platform="$2"
    local build_args="$3"
    local target="$4"
    
    log_step "开始构建Docker镜像..."
    
    # 构建Docker命令
    local docker_cmd="docker"
    local build_cmd="build"

    # 如果有平台参数且支持buildx，使用buildx
    if [ -n "$platform" ] && [ "$BUILDX_AVAILABLE" = true ]; then
        docker_cmd="docker buildx"
        build_cmd="build"
    fi
    
    # 构建命令参数
    local cmd_args=()
    cmd_args+=("-f" "$DOCKERFILE_PATH")
    cmd_args+=("-t" "$image_name")
    
    # 添加平台参数
    if [ -n "$platform" ]; then
        if [ "$BUILDX_AVAILABLE" = true ]; then
            cmd_args+=("--platform" "$platform")
        else
            log_warning "不支持多平台构建，忽略 --platform 参数"
        fi
    fi
    
    # 添加构建参数
    if [ -n "$build_args" ]; then
        IFS=',' read -ra ARGS <<< "$build_args"
        for arg in "${ARGS[@]}"; do
            cmd_args+=("--build-arg" "$arg")
        done
    fi
    
    # 添加目标阶段
    if [ -n "$target" ]; then
        cmd_args+=("--target" "$target")
    fi
    
    # 添加无缓存参数
    if [ "$NO_CACHE" = true ]; then
        cmd_args+=("--no-cache")
    fi
    
    # 添加推送参数（仅buildx支持）
    if [ "$PUSH_IMAGE" = true ] && [ "$BUILDX_AVAILABLE" = true ] && [ -n "$platform" ]; then
        cmd_args+=("--push")
    fi
    
    # 添加构建目录
    cmd_args+=("$PROJECT_ROOT")
    
    # 显示完整命令
    local full_cmd="$docker_cmd $build_cmd ${cmd_args[*]}"
    log_cmd "$full_cmd"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "干运行模式，实际不执行构建"
        return 0
    fi
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 执行构建
    if [ "$VERBOSE" = true ]; then
        $docker_cmd $build_cmd "${cmd_args[@]}"
    elif [ "$QUIET" = true ]; then
        $docker_cmd $build_cmd "${cmd_args[@]}" > /dev/null 2>&1
    else
        $docker_cmd $build_cmd "${cmd_args[@]}"
    fi
    
    local build_result=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $build_result -eq 0 ]; then
        log_success "Docker镜像构建成功 (用时: ${duration}秒)"
        return 0
    else
        log_error "Docker镜像构建失败"
        return 1
    fi
}

# 推送镜像
push_image() {
    local image_name="$1"
    
    if [ "$PUSH_IMAGE" != true ]; then
        return 0
    fi
    
    # 如果使用了buildx且有平台参数，可能已经推送了
    if [ "$BUILDX_AVAILABLE" = true ] && [ -n "$PLATFORM" ]; then
        log_success "镜像已通过buildx推送"
        return 0
    fi
    
    log_step "推送Docker镜像到仓库..."
    
    if [ "$DRY_RUN" = true ]; then
        log_cmd "docker push $image_name"
        log_info "干运行模式，实际不执行推送"
        return 0
    fi
    
    if [ "$QUIET" = true ]; then
        docker push "$image_name" > /dev/null 2>&1
    else
        docker push "$image_name"
    fi
    
    local push_result=$?
    if [ $push_result -eq 0 ]; then
        log_success "Docker镜像推送成功"
        return 0
    else
        log_error "Docker镜像推送失败"
        return 1
    fi
}

# 显示镜像信息
show_image_info() {
    local image_name="$1"
    
    if [ "$DRY_RUN" = true ] || [ "$QUIET" = true ]; then
        return 0
    fi
    
    log_step "镜像信息"
    
    # 显示镜像详情
    if docker image inspect "$image_name" &> /dev/null; then
        local image_id=$(docker image inspect "$image_name" --format '{{.Id}}' | cut -c8-19)
        local image_size=$(docker image inspect "$image_name" --format '{{.Size}}' | numfmt --to=iec)
        local created=$(docker image inspect "$image_name" --format '{{.Created}}' | cut -c1-19 | tr 'T' ' ')
        
        echo -e "${BLUE}镜像ID:${NC}       $image_id"
        echo -e "${BLUE}镜像大小:${NC}     $image_size"
        echo -e "${BLUE}创建时间:${NC}     $created"
    else
        log_warning "无法获取镜像信息"
    fi
}

# 清理临时文件
cleanup() {
    log_step "清理临时文件..."
    
    # 清理悬空镜像
    if [ "$DRY_RUN" != true ]; then
        local dangling_images=$(docker images -f "dangling=true" -q 2>/dev/null || true)
        if [ -n "$dangling_images" ]; then
            log_info "清理悬空镜像..."
            docker rmi $dangling_images &> /dev/null || true
        fi
    fi
    
    log_success "清理完成"
}

# 主函数
main() {
    local tag="$DEFAULT_TAG"
    local registry="${DOCKER_REGISTRY:-$DEFAULT_REGISTRY}"
    local project_name="$PROJECT_NAME"
    local platform=""
    local build_args=""
    local target=""
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -r|--registry)
                registry="$2"
                shift 2
                ;;
            -t|--tag)
                tag="$2"
                shift 2
                ;;
            -n|--name)
                project_name="$2"
                shift 2
                ;;
            -p|--push)
                PUSH_IMAGE=true
                shift
                ;;
            -f|--dockerfile)
                DOCKERFILE_PATH="$2"
                shift 2
                ;;
            --no-cache)
                NO_CACHE=true
                shift
                ;;
            --platform)
                platform="$2"
                shift 2
                ;;
            --build-arg)
                if [ -n "$build_args" ]; then
                    build_args="$build_args,$2"
                else
                    build_args="$2"
                fi
                shift 2
                ;;
            --target)
                target="$2"
                shift 2
                ;;
            -q|--quiet)
                QUIET=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -*)
                log_error "未知选项: $1"
                echo "使用 '$0 --help' 查看帮助信息"
                exit 1
                ;;
            *)
                tag="$1"
                shift
                ;;
        esac
    done
    
    # 从环境变量获取默认标签
    tag="${DOCKER_TAG:-$tag}"
    
    # 生成完整镜像名称
    local image_name=$(generate_image_name "$tag" "$registry" "$project_name")
    
    # 执行构建流程
    echo -e "${GREEN}🚀 Bubble RAG Docker构建工具${NC}"
    echo
    
    check_docker
    validate_dockerfile
    check_project_files
    show_build_info "$image_name" "$platform" "$build_args" "$target"
    
    if ! build_image "$image_name" "$platform" "$build_args" "$target"; then
        exit 1
    fi
    
    if ! push_image "$image_name"; then
        exit 1
    fi
    
    show_image_info "$image_name"
    cleanup
    
    echo
    log_success "🎉 Docker镜像构建完成！"
    echo -e "${GREEN}镜像名称: ${CYAN}$image_name${NC}"
    echo
    echo -e "${BLUE}后续步骤:${NC}"
    echo -e "  ${YELLOW}运行镜像:${NC} docker run -p 8000:8000 $image_name"
    echo -e "  ${YELLOW}查看镜像:${NC} docker images $PROJECT_NAME"
    if [ "$PUSH_IMAGE" = true ]; then
        echo -e "  ${YELLOW}拉取镜像:${NC} docker pull $image_name"
    fi
    echo
}

# 错误处理
trap 'log_error "构建过程中发生错误"; exit 1' ERR

# 执行主函数
main "$@"