"""
BM25存储管理模块
"""
import logging
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BM25Store:
    """BM25索引管理类"""
    
    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}  # 映射BM25索引位置到文档ID
        self.tokenized_corpus = []
        self.raw_corpus = []
    
    def clear(self):
        """清空索引"""
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []
        logger.info("BM25索引已清空")
    
    def build_index(self, documents: List[str], doc_ids: List[str]) -> bool:
        """构建BM25索引"""
        try:
            self.raw_corpus = documents
            self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
            
            # 对文档进行分词
            self.tokenized_corpus = []
            for doc in documents:
                tokens = list(jieba.cut(doc))
                self.tokenized_corpus.append(tokens)
            
            # 创建BM25索引
            self.bm25_index = BM25Okapi(self.tokenized_corpus)
            logger.info(f"BM25索引构建完成，共索引 {len(doc_ids)} 个文档")
            return True
        except Exception as e:
            logger.error(f"构建BM25索引失败: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """使用BM25检索相关文档"""
        if not self.bm25_index:
            return []
        
        try:
            # 对查询进行分词
            tokenized_query = list(jieba.cut(query))
            
            # 获取BM25得分
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # 获取得分最高的文档索引
            top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
            
            # 返回结果
            results = []
            for idx in top_indices:
                if bm25_scores[idx] > 0:  # 只返回有相关性的结果
                    results.append({
                        'id': self.doc_mapping[idx],
                        'score': float(bm25_scores[idx]),
                        'content': self.raw_corpus[idx]
                    })
            
            return results
        except Exception as e:
            logger.error(f"BM25搜索失败: {str(e)}")
            return []

# 全局BM25存储实例
bm25_store = BM25Store()
