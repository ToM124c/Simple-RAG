"""
交叉编码器管理模块
"""
import logging
import threading
from sentence_transformers import CrossEncoder
from config import CROSS_ENCODER_MODEL

logger = logging.getLogger(__name__)

class CrossEncoderManager:
    """交叉编码器管理类"""
    
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
    
    def get_model(self):
        """延迟加载交叉编码器模型"""
        if self.model is None:
            with self.lock:
                if self.model is None:
                    try:
                        self.model = CrossEncoder(CROSS_ENCODER_MODEL)
                        logger.info("交叉编码器加载成功")
                    except Exception as e:
                        logger.error(f"加载交叉编码器失败: {str(e)}")
                        self.model = None
        return self.model
    
    def predict(self, inputs):
        """预测相关性得分"""
        model = self.get_model()
        if model is None:
            raise ValueError("交叉编码器不可用")
        return model.predict(inputs)

# 全局交叉编码器实例
cross_encoder_manager = CrossEncoderManager()
