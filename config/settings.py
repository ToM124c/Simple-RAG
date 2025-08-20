"""
配置文件 - 集中管理所有配置参数
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions")

# 搜索配置
SEARCH_ENGINE = "google"
RERANK_METHOD = os.getenv("RERANK_METHOD", "llm")

# 模型配置
EMBED_MODEL_NAME = "BAAI/bge-m3"  # 使用硅基流动的BAAI/bge-m3
CROSS_ENCODER_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"
SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # 硅基流动模型

# 文本分割配置
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
SEPARATORS = ["\n\n", "\n", "。", "，", "；", "：", " ", ""]

# Markdown分割配置
MIN_SPLIT_LENGTH = 50
MAX_SPLIT_LENGTH = 100

# 检索配置
TOP_K_SEMANTIC = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 5
HYBRID_ALPHA = 0.7
MAX_RETRIEVAL_ITERATIONS = 3

# 网络请求配置
REQUEST_TIMEOUT = 30
SILICONFLOW_TIMEOUT = 120
SERPAPI_TIMEOUT = 15

# 端口配置
DEFAULT_PORTS = [17995, 17996, 17997, 17998, 17999]

# 环境变量设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
