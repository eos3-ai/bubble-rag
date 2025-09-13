#!/bin/bash

# Bubble RAG 部署脚本入口
# 调用docker/scripts/deploy-all.sh

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

chmod +x "$SCRIPT_DIR/docker/scripts/deploy-all.sh"

# 调用实际的部署脚本
exec "$SCRIPT_DIR/docker/scripts/deploy-all.sh" "$@"