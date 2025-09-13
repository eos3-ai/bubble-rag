# 🫧 Bubble RAG

> 🚀 企业级RAG（Retrieval-Augmented Generation）系统，集成文档管理、知识问答、模型训练于一体

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

[English](#-english) | [中文](#-中文)

</div>

---

## 🇺🇸 English

### 🎯 Overview

Bubble RAG is an enterprise-grade Retrieval-Augmented Generation system that seamlessly integrates document management,
intelligent question-answering, and model fine-tuning capabilities. It provides a complete solution for building
knowledge-based AI applications.
<img width="2560" height="1320" alt="1-3" src="https://github.com/user-attachments/assets/48e25335-cfa9-463c-b523-1ec1cb6dfc6a" />

### ✨ Key Features

- 📚 **Document Management**: Support for multiple file formats (PDF, Word, Excel, PowerPoint, HTML, Markdown, etc.)
- 🔍 **Intelligent Retrieval**: Advanced vector search with embedding and reranking models
- 💬 **Knowledge Q&A**: Conversational AI based on your documents
- 🎯 **Model Training**: Fine-tune embedding and language models for better performance
- 🔒 **Enterprise Ready**: Docker-based deployment with production-grade security
- 🌐 **Multi-modal Support**: OCR and PDF parsing capabilities (with additional configuration)

### 🚀 Quick Start

#### Prerequisites

| Component      | Minimum | Recommended |
|----------------|---------|-------------|
| CPU            | 4 cores | 8+ cores    |
| RAM            | 8GB     | 16GB+       |
| Storage        | 100GB   | 200GB+ SSD  |
| Docker         | 20.10+  | Latest      |
| Docker Compose | V2.0+   | Latest      |
| CUDA           | 12.0+   | 12.7        |

#### One-Command Deployment

```bash
# Clone the repository
git clone https://github.com/eos3-ai/bubble-rag.git
cd bubble-rag

# Make deployment script executable
chmod +x deploy-all.sh

# Deploy all services
./deploy-all.sh start

# Verify deployment
./deploy-all.sh health
```

#### Access Services

| Service     | URL                    | Description   | Authentication     |
|-------------|------------------------|---------------|--------------------|
| Backend API | http://localhost:8000  | Main RAG API  | bubble-rag-api-key |
| Frontend    | http://localhost:13000 | Web Interface | None               |
| Database    | localhost:3306         | MySQL         | laiye/laiye123456  |
| Vector DB   | localhost:19530        | Milvus        | None               |
| Cache       | localhost:6379         | Redis         | No password        |

### 📖 Documentation

- [📦 Installation Guide](QUICK_START.md) - Quick start and deployment
- [📚 User Manual](USER_MANUAL.md) - Detailed usage instructions

### 🤝 Contributing

We welcome contributions from the community!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### 🆘 Support

- 🐛 [Report Bugs](https://github.com/eos3-ai/bubble-rag/issues/new)
- 💡 [Request Features](https://github.com/eos3-ai/bubble-rag/issues/new)
- 📧 [Contact Us](mailto:support@laiye.ai)

---

## 🇨🇳 中文

### 🎯 项目简介

Bubble RAG 是首个集成RAG应用与训练平台的一体化解决方案。它提供了企业级的文档管理、智能问答和模型训练功能，帮助您快速构建基于知识的AI应用。
<img width="2560" height="1320" alt="1-3" src="https://github.com/user-attachments/assets/87c2fbfb-9e46-4eba-968f-114b900cc90c" />

### ✨ 核心功能

- 📚 **文档管理**：支持多种文件格式（PDF、Word、Excel、PowerPoint、HTML、Markdown等）
- 🔍 **智能检索**：基于向量的高级搜索，支持嵌入和重排序模型
- 💬 **知识问答**：基于文档的智能对话系统
- 🎯 **模型训练**：微调嵌入和语言模型以获得更好性能
- 🔒 **企业级部署**：基于Docker的生产环境部署，安全可靠
- 🌐 **多模态支持**：OCR和PDF解析功能（需额外配置）

### 🚀 快速开始

#### 系统要求

| 组件             | 最低配置   | 推荐配置       |
|----------------|--------|------------|
| CPU            | 4核心    | 8核心+       |
| 内存             | 8GB    | 16GB+      |
| 存储             | 100GB  | 200GB+ SSD |
| Docker         | 20.10+ | 最新版本       |
| Docker Compose | V2.0+  | 最新版本       |
| CUDA           | 12.0+  | 12.7       |

#### 一键部署

```bash
# 克隆代码仓库
git clone https://github.com/eos3-ai/bubble-rag.git
cd bubble-rag

# 赋予部署脚本执行权限
chmod +x deploy-all.sh

# 部署所有服务
./deploy-all.sh start

# 验证部署状态
./deploy-all.sh health
```

#### 服务访问

| 服务类型  | 访问地址                   | 描述        | 认证方式               |
|-------|------------------------|-----------|--------------------|
| 后端API | http://localhost:8000  | 主要RAG API | bubble-rag-api-key |
| 前端界面  | http://localhost:13000 | Web界面     | 无需认证               |
| 数据库   | localhost:3306         | MySQL     | laiye/laiye123456  |
| 向量数据库 | localhost:19530        | Milvus    | 无需认证               |
| 缓存服务  | localhost:6379         | Redis     | 无密码                |

### 🛠️ 开源 vs 企业版

| 功能模块                | 开源版本  | 企业版本 |
|---------------------|-------|------|
| **基础文档管理**          | ✅ 支持  | ✅ 支持 |
| **文件上传**            | ✅ 支持  | ✅ 支持 |
| **知识问答**            | ✅ 支持  | ✅ 支持 |
| **模型训练**            | ✅ 支持  | ✅ 支持 |
| **模型微调**            | ✅ 支持  | ✅ 支持 |
| **复杂图表解析**          | ❌ 不支持 | ✅ 支持 |
| **增强型表格解析**         | ❌ 不支持 | ✅ 支持 |
| **语义Chunking**      | ❌ 不支持 | ✅ 支持 |
| **Query处理增强**       | ❌ 不支持 | ✅ 支持 |
| **灵活检索策略**          | ❌ 不支持 | ✅ 支持 |
| **Small2Big技术**     | ❌ 不支持 | ✅ 支持 |
| **Chunk Graphing**  | ❌ 不支持 | ✅ 支持 |
| **Graph Embedding** | ❌ 不支持 | ✅ 支持 |
| **自迭代Pipeline**     | ❌ 不支持 | ✅ 支持 |
| **Deep Research模式** | ❌ 不支持 | ✅ 支持 |

### 📖 详细文档

- [📦 快速部署指南](QUICK_START.md) - 5分钟快速部署
- [📚 用户手册](USER_MANUAL.md) - 详细功能使用说明

### 🏗️ 项目架构

```
bubble-rag/
├── bubble_rag/           # 核心应用代码
│   ├── api_server.py     # FastAPI服务器
│   ├── routing/          # API路由层
│   ├── retrieving/       # 检索逻辑
│   ├── training/         # 模型训练
│   ├── databases/        # 数据访问层
│   └── utils/            # 工具函数
├── docker/               # Docker配置
├── database/             # 数据库脚本
├── resources/            # 资源文件
└── deploy-all.sh         # 部署脚本
```

### 🤝 贡献指南

我们欢迎社区贡献！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交修改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 📄 开源协议

本项目采用 Apache License 2.0 协议开源 - 详情请查看 [LICENSE](LICENSE) 文件。

### 🆘 技术支持

- 🐛 [报告问题](https://github.com/eos3-ai/bubble-rag/issues/new)
- 💡 [功能建议](https://github.com/eos3-ai/bubble-rag/issues/new)
- 📧 [联系我们](mailto:support@laiye.ai)

---

<div align="center">

**🎉 部署成功？** 现在您可以开始使用强大的RAG系统了！

*Last updated: September 2025*

</div>
