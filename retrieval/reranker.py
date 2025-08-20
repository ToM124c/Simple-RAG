"""
重排序模块 - 使用硅基流动API
"""
import logging
import re
from typing import List, Dict, Any, Tuple
from functools import lru_cache
import requests
import json
from models.cross_encoder import cross_encoder_manager
from config import RERANK_METHOD, SILICONFLOW_MODEL, TOP_K_FINAL, SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_TIMEOUT

logger = logging.getLogger(__name__)

class Reranker:
    """重排序器 - 使用硅基流动API"""
    
    def __init__(self):
        self.api_key = SILICONFLOW_API_KEY
        self.api_url = SILICONFLOW_API_URL
        self.model = SILICONFLOW_MODEL
    
    def rerank_with_cross_encoder(self, query: str, docs: List[str], doc_ids: List[str], 
                                 metadata_list: List[Dict], top_k: int = None) -> List[Tuple[str, Dict[str, Any]]]:
        """使用交叉编码器重排序"""
        top_k = top_k or TOP_K_FINAL
        
        if not docs:
            return []
        
        try:
            # 准备交叉编码器输入
            cross_inputs = [[query, doc] for doc in docs]
            
            # 计算相关性得分
            scores = cross_encoder_manager.predict(cross_inputs)
            
            # 组合结果
            results = [
                (doc_id, {
                    'content': doc,
                    'metadata': meta,
                    'score': float(score)
                })
                for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
            ]
            
            # 按得分排序
            results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
            
            return results[:top_k]
        except Exception as e:
            logger.error(f"交叉编码器重排序失败: {str(e)}")
            # 出错时返回原始顺序
            return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                    for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]
    
    @lru_cache(maxsize=32)
    def get_llm_relevance_score(self, query: str, doc: str) -> float:
        """使用硅基流动API对查询和文档的相关性进行评分（带缓存）"""
        try:
            # 构建评分提示词
            prompt = f"""给定以下查询和文档片段，评估它们的相关性。
评分标准：0分表示完全不相关，10分表示高度相关。
只需返回一个0-10之间的整数分数，不要有任何其他解释。

查询: {query}

文档片段: {doc}

相关性分数(0-10):"""
            
            # 调用硅基流动API
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=SILICONFLOW_TIMEOUT
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 提取得分
            score_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # 尝试解析为数字
            try:
                score = float(score_text)
                score = max(0, min(10, score))
                return score
            except ValueError:
                # 如果无法解析为数字，尝试从文本中提取数字
                match = re.search(r'\b([0-9]|10)\b', score_text)
                if match:
                    return float(match.group(1))
                else:
                    return 5.0
                    
        except Exception as e:
            logger.error(f"硅基流动API评分失败: {str(e)}")
            return 5.0
    
    def rerank_with_llm(self, query: str, docs: List[str], doc_ids: List[str], 
                       metadata_list: List[Dict], top_k: int = None) -> List[Tuple[str, Dict[str, Any]]]:
        """使用硅基流动API重排序"""
        top_k = top_k or TOP_K_FINAL
        
        if not docs:
            return []
        
        results = []
        
        # 对每个文档进行评分
        for doc_id, doc, meta in zip(doc_ids, docs, metadata_list):
            score = self.get_llm_relevance_score(query, doc)
            
            results.append((doc_id, {
                'content': doc,
                'metadata': meta,
                'score': score / 10.0  # 归一化到0-1
            }))
        
        # 按得分排序
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        
        return results[:top_k]
    
    def rerank(self, query: str, docs: List[str], doc_ids: List[str], 
              metadata_list: List[Dict], method: str = None, top_k: int = None) -> List[Tuple[str, Dict[str, Any]]]:
        """重排序主函数"""
        method = method or RERANK_METHOD
        logger.info(f"重排序方法: {method}")
        top_k = top_k or TOP_K_FINAL
        
        if method == "llm":
            # logger.info(f"使用llm重排序")
            return self.rerank_with_llm(query, docs, doc_ids, metadata_list, top_k)
        elif method == "cross_encoder":
            # logger.info(f"使用cross_encoder重排序")
            return self.rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)
        else:
            # 默认不进行重排序，按原始顺序返回
            return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                    for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]

# 全局重排序器实例
reranker = Reranker()
