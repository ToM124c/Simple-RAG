"""
网络搜索服务模块
"""
import logging
import requests
from typing import List, Dict, Any
from config import SERPAPI_KEY, SEARCH_ENGINE, SERPAPI_TIMEOUT

logger = logging.getLogger(__name__)

class WebSearchService:
    """网络搜索服务"""
    
    def __init__(self):
        pass
    
    def check_api_key(self) -> bool:
        """检查API密钥是否配置"""
        return SERPAPI_KEY is not None and SERPAPI_KEY.strip() != ""
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """执行网络搜索"""
        if not self.check_api_key():
            raise ValueError("未设置 SERPAPI_KEY 环境变量。请在.env文件中设置您的 API 密钥。")
        
        try:
            params = {
                "engine": SEARCH_ENGINE,
                "q": query,
                "api_key": SERPAPI_KEY,
                "num": num_results,
                "hl": "zh-CN",
                "gl": "cn"
            }
            
            response = requests.get(
                "https://serpapi.com/search", 
                params=params, 
                timeout=SERPAPI_TIMEOUT
            )
            response.raise_for_status()
            
            search_data = response.json()
            return self._parse_results(search_data)
            
        except Exception as e:
            logger.error(f"网络搜索失败: {str(e)}")
            return []
    
    def _parse_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析搜索结果"""
        results = []
        
        if "organic_results" in data:
            for item in data["organic_results"]:
                result = {
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet"),
                    "timestamp": item.get("date")
                }
                results.append(result)
        
        # 如果有知识图谱信息，也可以添加置顶
        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            results.insert(0, {
                "title": kg.get("title"),
                "url": kg.get("source", {}).get("link", ""),
                "snippet": kg.get("description"),
                "source": "knowledge_graph"
            })
        
        return results
    
    def get_web_context(self, query: str, num_results: int = 5) -> List[str]:
        """获取网络搜索上下文"""
        results = self.search(query, num_results)
        if not results:
            logger.info("网络搜索没有返回结果或发生错误")
            return []
        
        # 转换为文本格式
        web_texts = []
        for res in results:
            text = f"标题：{res.get('title', '')}\n摘要：{res.get('snippet', '')}"
            web_texts.append(text)
        
        logger.info(f"网络搜索返回 {len(results)} 条结果")
        return web_texts

# 全局网络搜索服务实例
web_search_service = WebSearchService()
