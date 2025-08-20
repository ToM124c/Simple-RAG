"""
混合检索模块
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from models.embedding_model import embedding_model
from storage.vector_store import vector_store
from storage.bm25_store import bm25_store
from config import TOP_K_SEMANTIC, TOP_K_BM25, HYBRID_ALPHA

logger = logging.getLogger(__name__)

class HybridRetriever:
    """混合检索器 - 结合语义检索和BM25检索"""
    
    def __init__(self):
        pass
    
    def semantic_search(self, query: str, top_k: int = None) -> Tuple[List[str], List[Dict], List[str]]:
        """语义检索"""
        top_k = top_k or TOP_K_SEMANTIC
        
        try:
            # 编码查询
            query_embedding = embedding_model.encode([query])
            query_embedding_np = np.array(query_embedding).astype('float32')
            
            # 向量搜索
            docs, metadatas, doc_ids = vector_store.search(query_embedding_np, top_k)
            
            return docs, metadatas, doc_ids
        except Exception as e:
            logger.error(f"语义检索失败: {str(e)}")
            return [], [], []
    
    def bm25_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """BM25检索"""
        top_k = top_k or TOP_K_BM25
        
        try:
            results = bm25_store.search(query, top_k)
            return results
        except Exception as e:
            logger.error(f"BM25检索失败: {str(e)}")
            return []
    
    def hybrid_merge(self, semantic_results: Tuple[List[str], List[Dict], List[str]], 
                    bm25_results: List[Dict[str, Any]], alpha: float = None) -> List[Tuple[str, Dict[str, Any]]]:
        """合并语义搜索和BM25搜索结果"""
        alpha = alpha or HYBRID_ALPHA
        merged_dict = {}
        
        # 处理语义搜索结果
        docs, metadatas, doc_ids = semantic_results
        if docs and metadatas and doc_ids:
            num_results = len(docs)
            for i, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadatas)):
                score = 1.0 - (i / max(1, num_results))  # 排名越高得分越高
                merged_dict[doc_id] = {
                    'score': alpha * score,
                    'content': doc,
                    'metadata': meta
                }
        
        # 处理BM25结果
        if bm25_results:
            valid_bm25_scores = [r['score'] for r in bm25_results if isinstance(r, dict) and 'score' in r]
            max_bm25_score = max(valid_bm25_scores) if valid_bm25_scores else 1.0
            
            for result in bm25_results:
                if not (isinstance(result, dict) and 'id' in result and 'score' in result and 'content' in result):
                    continue
                
                doc_id = result['id']
                normalized_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0
                
                if doc_id in merged_dict:
                    merged_dict[doc_id]['score'] += (1 - alpha) * normalized_score
                else:
                    metadata = vector_store.metadatas_map.get(doc_id, {})
                    merged_dict[doc_id] = {
                        'score': (1 - alpha) * normalized_score,
                        'content': result['content'],
                        'metadata': metadata
                    }
        
        # 按得分排序
        merged_results = sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)
        return merged_results
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, Dict[str, Any]]]:
        """执行混合检索"""
        top_k = top_k or TOP_K_SEMANTIC
        
        # 语义检索
        semantic_results = self.semantic_search(query, top_k)
        
        # BM25检索
        bm25_results = self.bm25_search(query, top_k)
        
        # 合并结果
        hybrid_results = self.hybrid_merge(semantic_results, bm25_results)
        
        return hybrid_results[:top_k]

# 全局混合检索器实例
hybrid_retriever = HybridRetriever()
