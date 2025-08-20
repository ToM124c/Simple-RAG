# 🚀 快速开始指南

## ⚡ 快速安装

### 1. 克隆项目
```bash
git clone <项目地址>
cd Local_Pdf_Chat_RAG-main
```

### 2. 创建虚拟环境
```bash
python -m venv rag_env
# Windows
rag_env\Scripts\activate
# Linux/macOS
source rag_env/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量
```bash
cp example.env .env
# 编辑.env文件，添加API密钥
```

在`.env`文件中配置以下内容：
```env
# 硅基流动API密钥（必需）
SILICONFLOW_API_KEY=your_siliconflow_api_key_here

# 网络搜索API密钥（可选）
SERPAPI_KEY=your_serpapi_key_here
```

## 🎯 启动系统

### 方式一：使用启动脚本（推荐）
```bash
# 启动Web界面
python start.py --mode gradio

# 启动API服务器
python start.py --mode api

# 检查系统依赖
python start.py --mode check
```

### 方式二：直接启动
```bash
# 启动Web界面
python main.py

# 启动API服务器
python api_server.py
```

## 📖 使用说明

### Web界面使用
1. 访问 `http://127.0.0.1:17995`
2. 上传PDF文档
3. 输入问题开始对话
4. 可选择启用网络搜索

### API接口使用
```bash
# 上传文档
curl -X POST "http://localhost:17995/api/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# 提问
curl -X POST "http://localhost:17995/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "你的问题", "enable_web_search": false}'

# 检查状态
curl "http://localhost:17995/api/status"
```

## 🔧 配置说明

### 主要配置项（config/settings.py）
- `CHUNK_SIZE`: 文本分块大小
- `TOP_K_SEMANTIC`: 语义检索结果数量
- `HYBRID_ALPHA`: 混合检索权重
- `MAX_RETRIEVAL_ITERATIONS`: 递归检索次数

### 环境变量（.env）
- `SILICONFLOW_API_KEY`: 硅基流动API密钥（必需）
- `SERPAPI_KEY`: 网络搜索API密钥（可选）

## 🐛 常见问题

### Q: 启动时提示"硅基流动API连接失败"
A: 确保已正确配置SILICONFLOW_API_KEY环境变量

### Q: 嵌入模型加载失败
A: 检查网络连接，确保能访问硅基流动API

### Q: 端口被占用
A: 系统会自动选择可用端口，或手动指定：
```bash
python start.py --port 8080
```

### Q: API调用失败
A: 检查API密钥是否正确，以及网络连接是否正常

## 📊 性能优化

### 1. 模型配置
- 使用硅基流动的BAAI/bge-m3嵌入模型
- 使用deepseek-ai/DeepSeek-R1-0528-Qwen3-8B生成模型

### 2. 检索优化
- 调整`CHUNK_SIZE`平衡准确性和速度
- 修改`HYBRID_ALPHA`优化检索策略

### 3. 系统优化
- 增加内存分配
- 使用SSD存储
- 优化网络连接

## 🔗 相关链接

- [项目文档](readme.md)
- [项目结构](PROJECT_STRUCTURE.md)
- [API文档](http://localhost:17995/docs)（启动API服务器后）
- [硅基流动官网](https://www.siliconflow.com)
- [Gradio文档](https://gradio.app)
