"""
嵌入模型管理模块 - 使用硅基流动的BAAI/bge-m3
"""
import logging
import requests
import json
import numpy as np
from typing import List, Union
from config import EMBED_MODEL_NAME, SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_TIMEOUT

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """嵌入模型管理类 - 使用硅基流动API"""
    
    def __init__(self):
        self.model_name = EMBED_MODEL_NAME
        self.api_key = SILICONFLOW_API_KEY
        self.api_url = SILICONFLOW_API_URL.replace("/chat/completions", "/embeddings")
        self._check_api_key()
    
    def _check_api_key(self):
        """检查API密钥"""
        if not self.api_key:
            raise ValueError("未设置 SILICONFLOW_API_KEY 环境变量")
        logger.info(f"嵌入模型 {self.model_name} 初始化成功")
    
    def encode(self, texts: Union[str, List[str]], show_progress_bar: bool = True) -> np.ndarray:
        """编码文本为向量"""
        if not texts:
            return np.array([])
        
        # 确保texts是列表
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # 准备请求数据
            payload = {
                "model": self.model_name,
                "input": texts
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 发送请求
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=SILICONFLOW_TIMEOUT
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 提取嵌入向量
            embeddings = []
            for item in result.get("data", []):
                embedding = item.get("embedding", [])
                embeddings.append(embedding)
            
            embeddings_array = np.array(embeddings)
            
            if show_progress_bar:
                logger.info(f"成功编码 {len(texts)} 个文本，向量维度: {embeddings_array.shape[1]}")
            
            return embeddings_array
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"编码失败: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        # BAAI/bge-m3的向量维度是1024
        return 1024

# 全局嵌入模型实例
embedding_model = EmbeddingModel()
