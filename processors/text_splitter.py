"""
文本分割处理模块
"""
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS

logger = logging.getLogger(__name__)

class TextSplitter:
    """文本分割器"""
    
    def __init__(self, chunk_size=None, chunk_overlap=None, separators=None):
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.separators = separators or SEPARATORS
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def split_text(self, text: str) -> list:
        """分割文本"""
        try:
            chunks = self.splitter.split_text(text)
            logger.info(f"文本分割完成，共生成 {len(chunks)} 个文本块")
            return chunks
        except Exception as e:
            logger.error(f"文本分割失败: {str(e)}")
            return []

# 全局文本分割器实例
text_splitter = TextSplitter()
