# ⚡ Bubble RAG 快速开始指南

> 🚀 5分钟快速部署企业级RAG系统，支持多模态文档处理和向量检索

## 📋 系统要求

| 组件                 | 最低要求   | 推荐配置   |
|--------------------|--------|--------|
| **CPU**            | 4核心    | 8核心+   |
| **内存**             | 8GB    | 16GB+  |
| **存储**             | 40GB   | 100GB+ |
| **Docker**         | 20.10+ | 最新版本   |
| **Docker Compose** | V2.0+  | 最新版本   |

## 🚀 快速部署

### 步骤1：环境检查

```bash
# 检查Docker版本
docker --version
docker compose version

# 检查系统资源
free -h  # 内存检查
df -h    # 磁盘空间检查
```

### 步骤2：获取代码

```bash
git clone https://github.com/your-org/bubble-rag.git
cd bubble-rag
```

### 步骤3：一键部署

> ⚠️ **远程部署注意事项**：如果部署到非本机服务器，请先修改 `.env.template` 文件中的配置：
> - `CURR_SERVER_IP`：修改为部署服务器的实际IP地址
> 
> 这样可以确保RAG服务的正常使用。

```bash
# 赋予执行权限
chmod +x deploy-all.sh

# 🎯 生产环境部署
./deploy-all.sh start

# 📝 详细日志模式
./deploy-all.sh -v start
```

### 步骤4：部署验证

```bash
# 🔍 服务状态检查
./deploy-all.sh health
```

### 步骤5：Embedding/Rerank模型配置（非必须）  

#### Option1

> 通过添加在线Embedding/Rerank模型地址，添加外部模型

图片

#### Option2

> 通过部署本地Embedding/Rerank模型，完成Embedding/Rerank模型配置

图片

## 🌐 服务访问

部署完成后可访问以下服务：

| 🎯 服务     | 🌐 访问地址                | 📝 说明     | 🔑 认证              |
|-----------|------------------------|-----------|--------------------|
| **后端应用**  | http://localhost:8000  | RAG API服务 | bubble-rag-api-key |
| **前端应用**  | http://localhost:13000 | Web界面     | 无需认证               |
| **数据库**   | localhost:3306         | MySQL数据库  | laiye/laiye123456  |
| **向量数据库** | localhost:19530        | Milvus服务  | 无需认证               |
| **缓存服务**  | localhost:6379         | Redis缓存   | 无密码                |

## 🔐 默认凭据

| 服务         | 用户名   | 密码          | 备注    |
|------------|-------|-------------|-------|
| **MySQL**  | laiye | laiye123456 | 数据库管理 |
| **Redis**  | -     | 无密码         | 缓存服务  |
| **Milvus** | -     | 无密码         | 向量数据库 |

## 🛠️ 运维命令

### 📊 监控管理

```bash
# ❤️ 健康状态检查
./deploy-all.sh health
```

### 🔄 服务控制

```bash
# 🔄 重启所有服务
./deploy-all.sh restart

# 🛑 停止所有服务
./deploy-all.sh stop

# ⏹️ 优雅停止（推荐）
./deploy-all.sh stop --timeout 60

# 🗑️ 清理所有数据（危险操作）
./deploy-all.sh clean
```

### 💾 数据管理

```bash
# 📦 数据备份
./deploy-all.sh backup

# 🔄 数据恢复
./deploy-all.sh restore /path/to/backup

# 🧹 清理无用资源
docker system prune -f
docker volume prune -f
```

## 🧪 功能测试

### 🔍 基础连通性测试

```bash
# ❤️ 服务健康检查
curl -f http://localhost:8000/health || echo "❌ RAG服务异常"

# 🖥️ 前端应用测试
curl -f http://localhost:13000 > /dev/null && echo "✅ 前端应用正常" || echo "❌ 前端应用异常"
```

## 🔧 故障排除

### ⚠️ 常见问题解决

#### 🚪 端口冲突

```bash
# 检查端口占用
sudo netstat -tlnp | grep :8000
sudo lsof -i :8000

# 解决方案
./deploy-all.sh stop
# 修改 .env.template 中的端口配置
./deploy-all.sh start
```

#### 💾 内存不足

```bash
# 检查内存使用
free -h
docker stats --no-stream

# 清理资源
docker system prune -a -f
docker volume prune -f
./deploy-all.sh restart
```

#### 🗄️ 数据库连接问题

```bash
# 检查MySQL状态
docker exec -it laiye_mysql mysql -u laiye -plaiye123456 -e "SELECT 1"

# 检查Milvus连接
curl -f http://localhost:9091/healthz

# 重置数据库
./deploy-all.sh stop
docker volume rm bubble_rag_mysql_data
./deploy-all.sh start
```

## 🔥高级功能

### OCR与PDF识别功能

#### OCR

> 需要修改配置文件(.env.prod)中的多模态模型配置(VLM_BASE_URL/VLM_API_KEY/VLM_MODEL_NAME)
> 
> 配置说明参考.env.template中的VLM_BASE_URL/VLM_API_KEY/VLM_MODEL_NAME配置项
> 
> 多模态模型部署 https://docs.sglang.ai/

#### PDF识别

> 需要修改配置文件中的MinerU配置(MINERU_SERVER_URL)
> 
> 配置说明参考.env.template中的MINERU_SERVER_URL配置项
>
> MinerU服务器部署 https://github.com/opendatalab/mineru


## 📈 性能优化

### 🎯 生产环境建议

- **内存**: 至少8GB，推荐16GB+
- **CPU**: 至少4核心，推荐8核心+
- **存储**: SSD硬盘，至少100GB空间
- **网络**: 千兆网络环境

## 📚 扩展文档

| 📖 文档       | 📝 说明         | 🔗 链接                                    |
|-------------|---------------|------------------------------------------|
| **完整部署指南**  | 详细部署配置说明      | [DEPLOYMENT.md](DEPLOYMENT.md)           |
| **API接口文档** | RESTful API参考 | [API_DOCS.md](API_DOCS.md)               |
| **配置参考**    | 环境变量配置说明      | [CONFIG.md](CONFIG.md)                   |
| **性能调优**    | 生产环境优化指南      | [PERFORMANCE.md](PERFORMANCE.md)         |
| **故障诊断**    | 问题排查手册        | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |

## 🤝 获取帮助

- 🐛 [报告Bug](https://github.com/your-org/bubble-rag/issues/new?template=bug_report.md)
- 💡 [功能请求](https://github.com/your-org/bubble-rag/issues/new?template=feature_request.md)
- 💬 [讨论交流](https://github.com/your-org/bubble-rag/discussions)
- 📧 [联系我们](mailto:support@your-org.com)

---

**🎉 部署成功？** 恭喜你！现在可以开始使用强大的RAG系统了！