"""
向量存储管理模块
"""
import logging
import numpy as np
import faiss
import time
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS向量存储管理类"""
    
    def __init__(self):
        self.index = None
        self.contents_map = {}  # original_id -> content
        self.metadatas_map = {}  # original_id -> metadata
        self.id_order_for_index = []  # 保存ID顺序
    
    def clear(self):
        """清空向量存储"""
        self.index = None
        self.contents_map = {}
        self.metadatas_map = {}
        self.id_order_for_index = []
        logger.info("向量存储已清空")
    
    def build_index(self, embeddings: np.ndarray, doc_ids: List[str], 
                   contents: List[str], metadatas: List[Dict[str, Any]]):
        """构建FAISS索引"""
        try:
            # 创建FAISS索引
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            # 保存文档映射
            for i, doc_id in enumerate(doc_ids):
                self.contents_map[doc_id] = contents[i]
                self.metadatas_map[doc_id] = metadatas[i]
            
            self.id_order_for_index.extend(doc_ids)
            
            logger.info(f"FAISS索引构建完成，共索引 {self.index.ntotal} 个文本块")
            return True
        except Exception as e:
            logger.error(f"构建FAISS索引失败: {str(e)}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[List[str], List[Dict], List[str]]:
        """搜索相似文档"""
        if not self.index or self.index.ntotal == 0:
            return [], [], []
        
        try:
            D, I = self.index.search(query_embedding, k=top_k)
            
            docs = []
            metadatas = []
            doc_ids = []
            
            for faiss_idx in I[0]:
                if faiss_idx != -1 and faiss_idx < len(self.id_order_for_index):
                    original_id = self.id_order_for_index[faiss_idx]
                    docs.append(self.contents_map.get(original_id, ""))
                    metadatas.append(self.metadatas_map.get(original_id, {}))
                    doc_ids.append(original_id)
            
            return docs, metadatas, doc_ids
        except Exception as e:
            logger.error(f"FAISS搜索失败: {str(e)}")
            return [], [], []
    
    def get_total_documents(self) -> int:
        """获取总文档数"""
        return self.index.ntotal if self.index else 0
    
    def is_empty(self) -> bool:
        """检查是否为空"""
        return self.index is None or self.index.ntotal == 0

# 全局向量存储实例
vector_store = VectorStore()
