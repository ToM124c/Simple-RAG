"""
工具函数模块
"""
import logging
import socket
import requests
from typing import List, Dict, Any
from config import DEFAULT_PORTS, SILICONFLOW_API_KEY, SILICONFLOW_API_URL,RERANK_METHOD

logger = logging.getLogger(__name__)

def is_port_available(port: int) -> bool:
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0

def find_available_port() -> int:
    """查找可用端口"""
    for port in DEFAULT_PORTS:
        if is_port_available(port):
            return port
    return None

def check_environment() -> bool:
    """检查环境依赖"""
    try:
        # 检查硅基流动API密钥
        if not SILICONFLOW_API_KEY:
            logger.error("未设置 SILICONFLOW_API_KEY 环境变量")
            return False
        
        # 测试硅基流动API连接
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 简单的API测试请求
        test_payload = {
            "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }
        
        response = requests.post(
            SILICONFLOW_API_URL,
            json=test_payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("硅基流动API连接正常")
            return True
        else:
            logger.error(f"硅基流动API连接异常，状态码: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"环境检查失败: {str(e)}")
        return False

def get_system_models_info() -> Dict[str, str]:
    """获取系统模型信息"""
    return {
        "嵌入模型": "BAAI/bge-m3 (硅基流动)",
        "分块方法": "RecursiveCharacterTextSplitter (chunk_size=400, overlap=40)",
        "检索方法": "向量检索 + BM25混合检索 (α=0.7)",
        "重排序模型": f"{RERANK_METHOD}",
        "生成模型": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B (硅基流动)",
        "分词工具": "jieba (中文分词)"
    }

def extract_urls_from_text(text: str) -> List[str]:
    """从文本中提取URL"""
    import re
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def sanitize_filename(filename: str) -> str:
    """清理文件名"""
    import re
    # 移除或替换不安全的字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 限制长度
    if len(filename) > 255:
        filename = filename[:255]
    return filename
