"""
API服务器 - 使用重构后的模块化架构
"""
import logging
import tempfile
import os
import re
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 导入重构后的模块
from config import *
from core.qa_engine import qa_engine
from processors.pdf_processor import pdf_processor
from services.web_search import web_search_service
from utils.helpers import check_environment

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api-server")

# 定义数据模型
class QuestionRequest(BaseModel):
    question: str
    enable_web_search: bool = False

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class FileProcessResult(BaseModel):
    status: str
    message: str
    file_info: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    status: str
    siliconflow_service: bool
    serpapi_configured: bool
    version: str
    models: List[str]

# 启动时确保模型和向量存储准备就绪
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 检查环境
    if not check_environment():
        logger.error("环境检查失败！请确保硅基流动API密钥已配置")
    yield
    # 清理工作（如果需要）
    logger.info("API服务已关闭")

# 初始化FastAPI应用
app = FastAPI(
    title="本地RAG API服务",
    description="提供基于硅基流动API的文档问答API接口",
    version="1.0.0",
    lifespan=lifespan
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_answer_stream(question: str, enable_web_search: bool):
    """处理流式回答，模拟同步函数的异步版本"""
    answer = ""
    
    # 创建生成器函数的包装器
    def run_stream():
        for response, status in qa_engine.stream_answer(question, enable_web_search):
            nonlocal answer
            answer = response
            yield response, status
    
    # 在异步上下文中运行同步代码
    loop = asyncio.get_event_loop()
    generator = run_stream()
    
    # 消费生成器直到最后一个结果
    try:
        while True:
            resp, status = await loop.run_in_executor(None, next, generator)
            if status == "完成!":
                break
    except StopIteration:
        pass
    
    return answer

@app.post("/api/upload", response_model=FileProcessResult)
async def upload_pdf(file: UploadFile = File(...)):
    """
    处理PDF文档并存入向量数据库
    - 支持格式：application/pdf
    - 最大文件大小：50MB
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "仅支持PDF文件")

    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 使用pdf_processor处理文件
        result_text = await asyncio.to_thread(
            pdf_processor.process_multiple_pdfs,
            [type('obj', (object,), {"name": tmp_path})]
        )
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        # 解析结果
        result = result_text[0] if isinstance(result_text, tuple) else result_text
        chunk_match = re.search(r'(\d+) 个文本块', result)
        chunks = int(chunk_match.group(1)) if chunk_match else 0
        
        return {
            "status": "success" if "成功" in result else "error",
            "message": result,
            "file_info": {
                "filename": file.filename,
                "chunks": chunks
            }
        }
    except Exception as e:
        logger.error(f"PDF处理失败: {str(e)}")
        raise HTTPException(500, f"文档处理失败: {str(e)}") from e

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """
    问答接口
    - question: 问题内容
    - enable_web_search: 是否启用网络搜索增强（默认False）
    """
    if not req.question:
        raise HTTPException(400, "问题不能为空")
    
    try:
        # 使用流式回答生成结果
        answer = await process_answer_stream(req.question, req.enable_web_search)
        
        # 提取可能的来源信息
        sources = []
        
        # 尝试提取标记的URL内容
        url_matches = re.findall(r'\[(网络来源|本地文档):[^\]]+\]\s*(?:\(URL:\s*([^)]+)\))?', answer)
        for source_type, url in url_matches:
            if url:
                sources.append({"type": source_type, "url": url})
            else:
                sources.append({"type": source_type})
        
        # 如果没有找到标记的URL，尝试解析其他格式
        if not sources:
            if "来源:" in answer or "来源：" in answer:
                source_sections = re.findall(r'来源[:|：](.*?)(?=来源[:|：]|$)', answer, re.DOTALL)
                for section in source_sections:
                    sources.append({"type": "引用", "content": section.strip()})
        
        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "enable_web_search": req.enable_web_search,
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            }
        }
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(500, f"问答处理失败: {str(e)}") from e

@app.get("/api/status", response_model=StatusResponse)
async def check_status():
    """检查API服务状态和环境配置"""
    siliconflow_status = check_environment()
    serpapi_status = web_search_service.check_api_key()
    
    return {
        "status": "healthy" if siliconflow_status else "degraded",
        "siliconflow_service": siliconflow_status,
        "serpapi_configured": serpapi_status,
        "version": "1.0.0",
        "models": ["deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "BAAI/bge-m3"]
    }

@app.get("/api/web_search_status")
async def check_web_search():
    """检查网络搜索功能是否可用"""
    return {
        "web_search_available": web_search_service.check_api_key(),
        "serpapi_configured": web_search_service.check_api_key()
    }

@app.get("/api/models")
async def get_models():
    """获取可用模型信息"""
    return {
        "embedding_model": "BAAI/bge-m3",
        "cross_encoder_model": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "siliconflow_model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    }

@app.get("/api/config")
async def get_config():
    """获取系统配置信息"""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k_semantic": TOP_K_SEMANTIC,
        "top_k_bm25": TOP_K_BM25,
        "hybrid_alpha": HYBRID_ALPHA,
        "max_retrieval_iterations": MAX_RETRIEVAL_ITERATIONS,
        "rerank_method": RERANK_METHOD
    }

if __name__ == "__main__":
    import uvicorn
    from utils.helpers import find_available_port
    
    port = find_available_port() or 17995
    
    logger.info(f"正在启动API服务，端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
