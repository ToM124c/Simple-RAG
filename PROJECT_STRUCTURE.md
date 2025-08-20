# 项目结构说明

## 📁 重构后的模块化架构

```
Local_Pdf_Chat_RAG-main/
├── config/                     # 配置模块
│   ├── __init__.py
│   └── settings.py            # 集中管理所有配置参数
│
├── models/                     # 模型模块
│   ├── __init__.py
│   ├── embedding_model.py     # 嵌入模型管理（硅基流动API）
│   └── cross_encoder.py       # 交叉编码器管理
│
├── storage/                    # 存储模块
│   ├── __init__.py
│   ├── vector_store.py        # FAISS向量存储
│   └── bm25_store.py          # BM25索引存储
│
├── processors/                 # 处理器模块
│   ├── __init__.py
│   ├── text_splitter.py       # 文本分割器
│   ├── pdf_processor.py       # PDF文档处理器
│   └── markdown_splitter.py   # Markdown分割器
│
├── retrieval/                  # 检索模块
│   ├── __init__.py
│   ├── hybrid_retriever.py    # 混合检索器
│   └── reranker.py            # 重排序器（硅基流动API）
│
├── services/                   # 服务模块
│   ├── __init__.py
│   ├── web_search.py          # 网络搜索服务
│   └── llm_service.py         # LLM服务（硅基流动API）
│
├── core/                       # 核心模块
│   ├── __init__.py
│   └── qa_engine.py           # 问答引擎核心
│
├── utils/                      # 工具模块
│   ├── __init__.py
│   └── helpers.py             # 工具函数
│
├── main.py                     # 主程序入口（Gradio界面）
├── api_server.py              # API服务器
├── start.py                   # 启动脚本
├── requirements.txt           # 依赖包
├── example.env                # 环境变量示例
├── .gitignore                 # Git忽略文件
├── readme.md                  # 项目说明
├── QUICK_START.md             # 快速开始指南
└── PROJECT_STRUCTURE.md       # 项目结构说明
```

## 🔧 模块功能说明

### 1. config/ - 配置模块
- **settings.py**: 集中管理所有配置参数，包括API密钥、模型配置、检索参数等
- 支持环境变量配置，便于部署和调试

### 2. models/ - 模型模块
- **embedding_model.py**: 管理文本嵌入模型，使用硅基流动API调用BAAI/bge-m3
- **cross_encoder.py**: 管理交叉编码器，用于检索结果重排序

### 3. storage/ - 存储模块
- **vector_store.py**: FAISS向量存储管理，支持高效的相似性搜索
- **bm25_store.py**: BM25关键词索引管理，支持传统信息检索

### 4. processors/ - 处理器模块
- **text_splitter.py**: 通用文本分割器，支持多种分割策略
- **pdf_processor.py**: PDF文档处理，包括文本提取和分块
- **markdown_splitter.py**: Markdown文档分割，支持结构化分割

### 5. retrieval/ - 检索模块
- **hybrid_retriever.py**: 混合检索器，结合语义检索和关键词检索
- **reranker.py**: 重排序器，支持交叉编码器和硅基流动API重排序

### 6. services/ - 服务模块
- **web_search.py**: 网络搜索服务，集成SerpAPI
- **llm_service.py**: LLM服务，使用硅基流动API

### 7. core/ - 核心模块
- **qa_engine.py**: 问答引擎核心，整合所有组件实现完整RAG流程

### 8. utils/ - 工具模块
- **helpers.py**: 通用工具函数，包括环境检查、端口管理等

## 🚀 主要入口文件

### main.py - Gradio界面
- 提供Web界面，支持文档上传、问答对话、分块可视化
- 使用重构后的模块化架构，代码更清晰

### api_server.py - API服务器
- 提供RESTful API接口
- 支持PDF上传、问答、状态查询等功能
- 便于集成到其他系统中

### start.py - 启动脚本
- 提供多种启动选项
- 支持依赖检查
- 统一的启动入口

## 🔄 重构优势

### 1. 模块化设计
- 每个模块职责单一，便于维护和扩展
- 模块间依赖关系清晰，降低耦合度

### 2. 配置集中管理
- 所有配置参数统一在config模块管理
- 支持环境变量配置，便于部署

### 3. 代码复用
- 核心功能模块化，可在不同场景下复用
- 减少代码重复，提高开发效率

### 4. 易于测试
- 每个模块可独立测试
- 便于单元测试和集成测试

### 5. 扩展性强
- 新增功能只需添加相应模块
- 支持插件化扩展

### 6. API化架构
- 使用硅基流动API，无需本地部署大模型
- 降低硬件要求，提高部署便利性

## 📝 使用说明

### 启动Gradio界面
```bash
python start.py --mode gradio
```

### 启动API服务器
```bash
python start.py --mode api
```

### 环境配置
1. 复制 `example.env` 为 `.env`
2. 配置硅基流动API密钥
3. 可选配置SerpAPI密钥

## 🔧 开发指南

### 添加新功能
1. 在相应模块中添加功能
2. 更新配置文件（如需要）
3. 在主程序或API中集成

### 修改配置
1. 编辑 `config/settings.py`
2. 或通过环境变量设置

### 扩展模型支持
1. 在 `models/` 模块中添加新模型类
2. 在 `services/llm_service.py` 中添加调用逻辑
3. 更新配置文件

## 🌟 技术特点

### 1. 云端模型架构
- 使用硅基流动API，无需本地GPU
- 支持BAAI/bge-m3嵌入模型
- 支持deepseek-ai/DeepSeek-R1-0528-Qwen3-8B生成模型

### 2. 混合检索策略
- 语义检索 + BM25关键词检索
- 可配置的混合权重
- 支持多种重排序方法

### 3. 模块化设计
- 清晰的模块划分
- 易于维护和扩展
- 支持插件化架构
