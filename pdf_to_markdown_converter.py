from pathlib import Path
import logging
import os
import shutil

# 导入 Simple-RAG 项目中的核心处理类
# 假设您正在从项目根目录运行此脚本
from processors.pdf_markdown import PDFParser, PageTextPreparation

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_log = logging.getLogger(__name__)

def convert_pdf_to_markdown(
    input_pdf_path: Path,
    output_dir: Path,
    debug_data_dir: Path | None = None,
    use_serialized_tables: bool = False
):
    """
    将单个 PDF 文件转换为 Markdown 格式。

    Args:
        input_pdf_path (Path): 输入 PDF 文件的路径。
        output_dir (Path): 存储最终 Markdown 文件的目录。
        debug_data_dir (Path, optional): 存储中间解析 JSON 文件的目录。默认为 None。
        use_serialized_tables (bool, optional): 是否使用序列化表格。默认为 False。
    """
    if not input_pdf_path.is_file():
        _log.error(f"错误：找不到输入 PDF 文件：{input_pdf_path}")
        return

    _log.info(f"开始处理 PDF 文件：{input_pdf_path.name}")

    # 1. 初始化 PDFParser 并解析 PDF
    # 不传入 csv_metadata_path，因为用户表示没有元数据
    parsed_reports_temp_dir = output_dir / "_temp_parsed_jsons"
    parsed_reports_temp_dir.mkdir(parents=True, exist_ok=True)

    pdf_parser = PDFParser(
        output_dir=parsed_reports_temp_dir,
        csv_metadata_path=None,  # 不使用元数据
    )
    pdf_parser.debug_data_path = debug_data_dir # 中间调试数据

    try:
        pdf_parser.parse_and_export(input_doc_paths=[input_pdf_path])
        _log.info(f"PDF 解析完成。中间 JSON 文件保存在：{parsed_reports_temp_dir}")
    except Exception as e:
        _log.error(f"解析 PDF 时发生错误 {input_pdf_path.name}: {e}")
        # 清理临时目录
        if parsed_reports_temp_dir.exists():
            shutil.rmtree(parsed_reports_temp_dir)
        return

    # 2. 初始化 PageTextPreparation 并将 JSON 报告转换为 Markdown
    page_text_prep = PageTextPreparation(use_serialized_tables=use_serialized_tables)

    # 确保最终输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        page_text_prep.export_to_markdown(
            reports_dir=parsed_reports_temp_dir,
            output_dir=output_dir
        )
        _log.info(f"Markdown 转换完成。Markdown 文件保存在：{output_dir}")
    except Exception as e:
        _log.error(f"将 JSON 转换为 Markdown 时发生错误 {input_pdf_path.name}: {e}")
    finally:
        # 清理临时解析的 JSON 文件
        if parsed_reports_temp_dir.exists():
            _log.info(f"清理临时解析文件：{parsed_reports_temp_dir}")
            shutil.rmtree(parsed_reports_temp_dir)

    _log.info(f"PDF 文件 {input_pdf_path.name} 处理完成。")

if __name__ == "__main__":
    # 这是一个如何使用 convert_pdf_to_markdown 函数的示例

    # 请将 'your_pdf_files/document.pdf' 替换为您实际的 PDF 文件路径
    # 确保该文件存在。
    # 例如：input_pdf = Path("data/test_set/pdf_reports/your_document.pdf")

    # 获取当前脚本的目录
    current_script_dir = Path(__file__).parent

    # 示例用法 1: 转换一个位于 "data/test_set/pdf_reports/" 的 PDF 文件
    # 请确保您的 Simple-RAG 目录下有相应的 PDF 文件，例如将一个 PDF 文件放入 data/test_set/pdf_reports/
    # 或者直接指定一个绝对路径
    
    # 您可以在这里设置您要转换的 PDF 文件路径
    # 建议将测试 PDF 文件放在项目根目录下的某个文件夹中，例如 `test_pdfs`
    # input_pdf = current_script_dir / "test_pdfs" / "example.pdf"
    # input_pdf = Path("E:/实习/项目/Simple-RAG/data/test_set/pdf_reports/sample.pdf") # 替换为您的实际路径
    
    # 假设您的项目根目录是 E:\实习\项目\Simple-RAG
    # 并且您的 PDF 文件在 E:\实习\项目\Simple-RAG\data\test_set\pdf_reports\ 目录下
    # 示例 PDF 文件名
    # pdf_filename = "test_file.pdf" # 请替换为实际的 PDF 文件名
    input_pdf = Path("./test_file.pdf") 

    # 转换后的 Markdown 文件将存储在此目录中
    output_markdown_dir = Path("./converted_markdowns")

    # 中间调试 JSON 文件将存储在此目录中（可选）
    debug_json_dir = Path("./debug_jsons")

    print(f"尝试转换 PDF：{input_pdf}")
    print(f"Markdown 输出目录：{output_markdown_dir}")
    print(f"调试 JSON 目录：{debug_json_dir}")

    # 执行转换
    convert_pdf_to_markdown(
        input_pdf_path=input_pdf,
        output_dir=output_markdown_dir,
        debug_data_dir=debug_json_dir,
        use_serialized_tables=False # 如果您的表格需要特殊处理，可以设置为 True
    )

    print("\n转换过程已启动。请检查控制台输出和指定输出目录。")
    print("如果遇到 docling 相关的错误，请确保已正确安装 docling 及其所有依赖，并且模型已下载。")
