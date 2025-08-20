"""
PDF处理模块
"""
import logging
import os
import time
from typing import List, Dict, Any
from pdfminer.high_level import extract_text_to_fp
from io import StringIO
from processors.text_splitter import text_splitter
from models.embedding_model import embedding_model
from storage.vector_store import vector_store
from storage.bm25_store import bm25_store
import numpy as np

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF文档处理器"""
    
    def __init__(self):
        self.processed_files = {}
    
    def extract_text(self, filepath: str) -> str:
        """从PDF提取文本"""
        try:
            output = StringIO()
            with open(filepath, 'rb') as file:
                extract_text_to_fp(file, output)
            return output.getvalue()
        except Exception as e:
            logger.error(f"PDF文本提取失败: {str(e)}")
            raise
    
    def process_single_pdf(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """处理单个PDF文件"""
        try:
            # 提取文本
            text = self.extract_text(file_path)
            if not text.strip():
                raise ValueError("文档内容为空或无法提取文本")
            
            # 分割文本
            chunks = text_splitter.split_text(text)
            if not chunks:
                raise ValueError("文本分割失败")
            
            # 生成文档ID
            doc_id = f"doc_{int(time.time())}_{hash(file_name)}"
            
            # 生成分块ID和元数据
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": file_name, "doc_id": doc_id} for _ in chunks]
            
            return {
                "chunks": chunks,
                "chunk_ids": chunk_ids,
                "metadatas": metadatas,
                "doc_id": doc_id
            }
        except Exception as e:
            logger.error(f"处理PDF文件 {file_name} 失败: {str(e)}")
            raise
    
    def process_multiple_pdfs(self, files: List[Any], progress_callback=None) -> tuple:
        """处理多个PDF文件"""
        if not files:
            return "请选择要上传的PDF文件", []
        
        try:
            # 清空存储
            if progress_callback:
                progress_callback(0.1, desc="清理历史数据...")
            
            vector_store.clear()
            bm25_store.clear()
            
            # 收集所有数据
            all_chunks = []
            all_chunk_ids = []
            all_metadatas = []
            processed_results = []
            total_chunks = 0
            
            for idx, file in enumerate(files, 1):
                try:
                    file_name = os.path.basename(file.name)
                    if progress_callback:
                        progress_callback((idx-1)/len(files), desc=f"处理文件 {idx}/{len(files)}: {file_name}")
                    
                    # 处理单个PDF
                    result = self.process_single_pdf(file.name, file_name)
                    
                    all_chunks.extend(result["chunks"])
                    all_chunk_ids.extend(result["chunk_ids"])
                    all_metadatas.extend(result["metadatas"])
                    
                    total_chunks += len(result["chunks"])
                    processed_results.append(f"✅ {file_name}: 成功处理 {len(result['chunks'])} 个文本块")
                    
                except Exception as e:
                    error_msg = str(e)
                    processed_results.append(f"❌ {file_name}: 处理失败 - {error_msg}")
            
            # 生成向量嵌入
            if all_chunks:
                if progress_callback:
                    progress_callback(0.8, desc="生成文本嵌入...")
                
                embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
                embeddings_np = np.array(embeddings).astype('float32')
                
                # 构建向量索引
                if progress_callback:
                    progress_callback(0.9, desc="构建FAISS索引...")
                
                vector_store.build_index(embeddings_np, all_chunk_ids, all_chunks, all_metadatas)
                
                # 构建BM25索引
                if progress_callback:
                    progress_callback(0.95, desc="构建BM25检索索引...")
                
                bm25_store.build_index(all_chunks, all_chunk_ids)
            
            summary = f"\n总计处理 {len(files)} 个文件，{total_chunks} 个文本块"
            processed_results.append(summary)
            
            return "\n".join(processed_results), []
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"整体处理过程出错: {error_msg}")
            return f"处理过程出错: {error_msg}", []

# 全局PDF处理器实例
pdf_processor = PDFProcessor()
