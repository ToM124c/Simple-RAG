#!/usr/bin/env python3
"""
启动脚本 - 提供多种启动选项
"""
import argparse
import sys
import logging
from utils.helpers import check_environment, find_available_port

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("start")

def start_gradio():
    """启动Gradio界面"""
    try:
        from main import main
        main()
    except ImportError as e:
        logger.error(f"导入main模块失败: {e}")
        logger.error("请确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动Gradio界面失败: {e}")
        sys.exit(1)

def start_api_server():
    """启动API服务器"""
    try:
        import uvicorn
        from api_server import app
        from utils.helpers import find_available_port
        
        port = find_available_port() or 17995
        logger.info(f"正在启动API服务器，端口: {port}")
        
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError as e:
        logger.error(f"导入API模块失败: {e}")
        logger.error("请确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动API服务器失败: {e}")
        sys.exit(1)

def check_dependencies():
    """检查依赖"""
    logger.info("检查系统依赖...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        logger.error("需要Python 3.8或更高版本")
        return False
    
    # 检查硅基流动API环境
    if not check_environment():
        logger.error("环境检查失败！请确保硅基流动API密钥已配置")
        return False
    
    # 检查端口
    port = find_available_port()
    if not port:
        logger.error("所有默认端口都被占用，请手动释放端口")
        return False
    
    logger.info(f"系统检查通过，可用端口: {port}")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="本地RAG问答系统启动脚本")
    parser.add_argument(
        "--mode", 
        choices=["gradio", "api", "check"], 
        default="gradio",
        help="启动模式: gradio(Web界面), api(API服务器), check(检查依赖)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        help="指定端口号（可选）"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"启动模式: {args.mode}")
    
    if args.mode == "check":
        if check_dependencies():
            logger.info("✅ 系统依赖检查通过")
            sys.exit(0)
        else:
            logger.error("❌ 系统依赖检查失败")
            sys.exit(1)
    
    elif args.mode == "api":
        start_api_server()
    
    elif args.mode == "gradio":
        start_gradio()
    
    else:
        logger.error(f"未知的启动模式: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
