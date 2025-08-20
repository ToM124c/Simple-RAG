"""
LLM服务模块 - 使用硅基流动API
"""
import logging
import json
import requests
from typing import Dict, Any, Tuple
from config import SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_MODEL, SILICONFLOW_TIMEOUT

logger = logging.getLogger(__name__)

class LLMService:
    """LLM服务管理类 - 使用硅基流动API"""
    
    def __init__(self):
        self.api_key = SILICONFLOW_API_KEY
        self.api_url = SILICONFLOW_API_URL
        self.model = SILICONFLOW_MODEL
        self._check_api_key()
    
    def _check_api_key(self):
        """检查API密钥"""
        if not self.api_key:
            raise ValueError("未设置 SILICONFLOW_API_KEY 环境变量")
        logger.info(f"LLM服务初始化成功，使用模型: {self.model}")
    
    def call_siliconflow_api(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """调用硅基流动API"""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "max_tokens": max_tokens,
                "stop": None,
                "temperature": temperature,
                "top_p": 0.7,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "n": 1,
                "response_format": {"type": "text"}
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json; charset=utf-8"
            }
            
            json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            
            response = requests.post(
                self.api_url,
                data=json_payload,
                headers=headers,
                timeout=SILICONFLOW_TIMEOUT
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 提取回答内容
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]
                content = message.get("content", "")
                return content
            else:
                return "API返回结果格式异常，请检查"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"调用硅基流动API时出错: {str(e)}")
            return f"调用API时出错: {str(e)}"
        except json.JSONDecodeError:
            logger.error("硅基流动API返回非JSON响应")
            return "API响应解析失败"
        except Exception as e:
            logger.error(f"调用硅基流动API时发生未知错误: {str(e)}")
            return f"发生未知错误: {str(e)}"
    
    def generate_answer(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """生成回答"""
        return self.call_siliconflow_api(prompt, temperature, max_tokens)
    
    def generate_next_query(self, initial_query: str, current_context: str) -> str:
        """生成下一个查询"""
        next_query_prompt = f"""基于原始问题: {initial_query}
以及已检索信息: 
{current_context}

分析是否需要进一步查询。如果需要，请提供新的查询问题，使用不同角度或更具体的关键词。
如果已经有充分信息，请回复'不需要进一步查询'。

新查询(如果需要):"""
        
        try:
            next_query = self.call_siliconflow_api(next_query_prompt, temperature=0.7, max_tokens=256).strip()
            return next_query
        except Exception as e:
            logger.error(f"生成新查询时出错: {str(e)}")
            return ""

# 全局LLM服务实例
llm_service = LLMService()
