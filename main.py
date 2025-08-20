"""
主程序入口 - 使用重构后的模块化架构（硅基流动版本）
"""
import logging
import gradio as gr
import webbrowser
import jieba
from typing import List, Any

# 导入重构后的模块
from config import *
from core.qa_engine import qa_engine
from processors.pdf_processor import pdf_processor
from storage.vector_store import vector_store
from storage.bm25_store import bm25_store
from utils.helpers import check_environment, find_available_port, get_system_models_info

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# 全局变量
chunk_data_cache = []
chunk_meta_cache = []

def process_chat(question: str, history: List, enable_web_search: bool):
    """处理聊天对话"""
    if history is None:
        history = []
    
    # 更新模型选择信息的显示
    api_text = f"""
    <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
        <p>📢 <strong>功能说明：</strong></p>
        <p>1. <strong>联网搜索</strong>：{'已启用' if enable_web_search else '未启用'}</p>
        <p>2. <strong>模型选择</strong>：当前使用 <strong>硅基流动 DeepSeek-R1-0528-Qwen3-8B 模型</strong> {'(需要在.env文件中配置SERPAPI_KEY)' if enable_web_search else ''}</p>
    </div>
    """
    
    # 如果问题为空，不处理
    if not question or question.strip() == "":
        history.append(("", "问题不能为空，请输入有效问题。"))
        return history, "", api_text
    
    # 添加用户问题到历史
    history.append((question, ""))
    
    # 创建生成器
    resp_generator = qa_engine.stream_answer(question, enable_web_search)
    
    # 流式更新回答
    for response, status in resp_generator:
        history[-1] = (question, response)
        yield history, "", api_text

def clear_chat_history():
    """清空聊天历史"""
    return None, "对话已清空"

def update_api_info(enable_web_search: bool):
    """更新API信息显示"""
    api_text = f"""
    <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
        <p>📢 <strong>功能说明：</strong></p>
        <p>1. <strong>联网搜索</strong>：{'已启用' if enable_web_search else '未启用'}</p>
        <p>2. <strong>模型选择</strong>：当前使用 <strong>硅基流动 DeepSeek-R1-0528-Qwen3-8B 模型</strong> {'(需要在.env文件中配置SERPAPI_KEY)' if enable_web_search else ''}</p>
    </div>
    """
    return api_text

def get_document_chunks(progress=gr.Progress()):
    """获取文档分块结果用于可视化"""
    global chunk_data_cache, chunk_meta_cache
    
    try:
        progress(0.1, desc="正在从内存加载数据...")
        
        if not vector_store.id_order_for_index:
            chunk_data_cache = []
            chunk_meta_cache = []
            return [], "知识库中没有文档，请先上传并处理文档。"
        
        progress(0.5, desc="正在组织分块数据...")
        
        doc_groups = {}
        for doc_id in vector_store.id_order_for_index:
            doc = vector_store.contents_map.get(doc_id, "")
            meta = vector_store.metadatas_map.get(doc_id, {})
            if not doc:
                continue

            source = meta.get('source', '未知来源')
            if source not in doc_groups:
                doc_groups[source] = []
            
            doc_id_meta = meta.get('doc_id', '未知ID')
            chunk_info = {
                "original_id": doc_id,
                "doc_id": doc_id_meta,
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                "full_content": doc,
                "token_count": len(list(jieba.cut(doc))),
                "char_count": len(doc)
            }
            doc_groups[source].append(chunk_info)
        
        result_lists = []
        meta_rows = []
        source_chunk_counters = {source: 0 for source in doc_groups.keys()}
        total_chunks = 0
        
        for source, chunks in doc_groups.items():
            num_chunks_in_source = len(chunks)
            for chunk in chunks:
                source_chunk_counters[source] += 1
                total_chunks += 1
                
                # 展示行
                result_lists.append([
                    source,
                    f"{source_chunk_counters[source]}/{num_chunks_in_source}",
                    chunk["char_count"],
                    chunk["token_count"],
                    chunk["content_preview"]
                ])
                # 元数据行（与展示行索引保持一致）
                meta_rows.append({
                    "original_id": chunk["original_id"],
                    "doc_id": chunk["doc_id"],
                    "full_content": chunk["full_content"],
                    "source": source
                })
        
        progress(1.0, desc="数据加载完成!")
        
        chunk_data_cache = result_lists
        chunk_meta_cache = meta_rows
        summary = f"总计 {total_chunks} 个文本块，来自 {len(doc_groups)} 个不同来源。"
        
        return result_lists, summary
    except Exception as e:
        chunk_data_cache = []
        chunk_meta_cache = []
        return [], f"获取分块数据失败: {str(e)}"

def show_chunk_details(evt: gr.SelectData, chunks):
    """显示选中分块的详细内容"""
    try:
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
        if 0 <= row_index < len(chunk_meta_cache):
            return chunk_meta_cache[row_index].get("full_content", "内容加载失败")
        return "未找到选中的分块"
    except Exception as e:
        return f"加载分块详情失败: {str(e)}"

def create_ui():
    """创建Gradio界面"""
    with gr.Blocks(
        title="本地RAG问答系统（硅基流动版本）",
        css="""
        /* 全局主题变量 */
        :root[data-theme="light"] {
            --text-color: #2c3e50;
            --bg-color: #ffffff;
            --panel-bg: #f8f9fa;
            --border-color: #e9ecef;
            --success-color: #4CAF50;
            --error-color: #f44336;
            --primary-color: #2196F3;
            --secondary-bg: #ffffff;
            --hover-color: #e9ecef;
            --chat-user-bg: #e3f2fd;
            --chat-assistant-bg: #f5f5f5;
        }

        :root[data-theme="dark"] {
            --text-color: #e0e0e0;
            --bg-color: #1a1a1a;
            --panel-bg: #2d2d2d;
            --border-color: #404040;
            --success-color: #81c784;
            --error-color: #e57373;
            --primary-color: #64b5f6;
            --secondary-bg: #2d2d2d;
            --hover-color: #404040;
            --chat-user-bg: #1e3a5f;
            --chat-assistant-bg: #2d2d2d;
        }

        /* 全局样式 */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 0;
            overflow-x: hidden;
            width: 100vw;
            height: 100vh;
        }

        .gradio-container {
            max-width: 100% !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 1% !important;
            color: var(--text-color);
            background-color: var(--bg-color);
            min-height: 100vh;
        }
        
        /* 确保标签内容撑满 */
        .tabs.svelte-710i53 {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
        }

        /* 面板样式 */
        .left-panel {
            padding-right: 20px;
            border-right: 1px solid var(--border-color);
            background: var(--bg-color);
            width: 100%;
        }

        .right-panel {
            height: 100vh;
            background: var(--bg-color);
            width: 100%;
        }

        /* 文件列表样式 */
        .file-list {
            margin-top: 10px;
            padding: 12px;
            background: var(--panel-bg);
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.6;
            border: 1px solid var(--border-color);
        }

        /* 答案框样式 */
        .answer-box {
            min-height: 500px !important;
            background: var(--panel-bg);
            border-radius: 8px;
            padding: 16px;
            font-size: 15px;
            line-height: 1.6;
            border: 1px solid var(--border-color);
        }

        /* 输入框样式 */
        textarea {
            background: var(--panel-bg) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 14px !important;
        }

        /* 按钮样式 */
        button.primary {
            background: var(--primary-color) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }

        button.primary:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        /* 标题和文本样式 */
        h1, h2, h3 {
            color: var(--text-color) !important;
            font-weight: 600 !important;
        }

        .footer-note {
            color: var(--text-color);
            opacity: 0.8;
            font-size: 13px;
            margin-top: 12px;
        }

        /* 聊天记录样式 */
        .chat-container {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            max-height: 80vh;
            height: 80vh !important;
            overflow-y: auto;
            background: var(--bg-color);
        }

        .chat-message {
            padding: 12px 16px;
            margin: 8px;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.5;
        }

        .chat-message.user {
            background: var(--chat-user-bg);
            margin-left: 32px;
            border-top-right-radius: 4px;
        }

        .chat-message.assistant {
            background: var(--chat-assistant-bg);
            margin-right: 32px;
            border-top-left-radius: 4px;
        }

        .chat-message .timestamp {
            font-size: 12px;
            color: var(--text-color);
            opacity: 0.7;
            margin-bottom: 4px;
        }

        .chat-message .content {
            white-space: pre-wrap;
        }

        /* 按钮组样式 */
        .button-row {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }

        .clear-button {
            background: var(--error-color) !important;
        }

        /* API配置提示样式 */
        .api-info {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
        }

        /* 数据可视化卡片样式 */
        .model-card {
            background: var(--panel-bg);
            border-radius: 8px;
            padding: 16px;
            border: 1px solid var(--border-color);
            margin-bottom: 16px;
        }

        .model-card h3 {
            margin-top: 0;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }

        .model-item {
            display: flex;
            margin-bottom: 8px;
        }

        .model-item .label {
            flex: 1;
            font-weight: 500;
        }

        .model-item .value {
            flex: 2;
        }

        /* 数据表格样式 */
        .chunk-table {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }

        .chunk-table th, .chunk-table td {
            border: 1px solid var(--border-color);
            padding: 8px;
        }

        .chunk-detail-box {
            min-height: 200px;
            padding: 16px;
            background: var(--panel-bg);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            font-family: monospace;
            white-space: pre-wrap;
            overflow-y: auto;
        }
        """
    ) as demo:
        gr.Markdown("# 🧠 智能文档问答系统（硅基流动版本）")
        
        with gr.Tabs() as tabs:
            # 第一个选项卡：问答对话
            with gr.TabItem("💬 问答对话"):
                with gr.Row(equal_height=True):
                    # 左侧操作面板
                    with gr.Column(scale=5, elem_classes="left-panel"):
                        gr.Markdown("## 📂 文档处理区")
                        with gr.Group():
                            file_input = gr.File(
                                label="上传PDF文档",
                                file_types=[".pdf"],
                                file_count="multiple"
                            )
                            upload_btn = gr.Button("🚀 开始处理", variant="primary")
                            upload_status = gr.Textbox(
                                label="处理状态",
                                interactive=False,
                                lines=2
                            )
                            file_list = gr.Textbox(
                                label="已处理文件",
                                interactive=False,
                                lines=3,
                                elem_classes="file-list"
                            )
                        
                        # 问题输入区
                        gr.Markdown("## ❓ 输入问题")
                        with gr.Group():
                            question_input = gr.Textbox(
                                label="输入问题",
                                lines=3,
                                placeholder="请输入您的问题...",
                                elem_id="question-input"
                            )
                            with gr.Row():
                                web_search_checkbox = gr.Checkbox(
                                    label="启用联网搜索", 
                                    value=False,
                                    info="打开后将同时搜索网络内容（需配置SERPAPI_KEY）"
                                )
                                
                            with gr.Row():
                                ask_btn = gr.Button("🔍 开始提问", variant="primary", scale=2)
                                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", elem_classes="clear-button", scale=1)
                        
                        # API配置提示信息
                        api_info = gr.HTML(
                            """
                            <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
                                <p>📢 <strong>功能说明：</strong></p>
                                <p>1. <strong>联网搜索</strong>：未启用</p>
                                <p>2. <strong>模型选择</strong>：当前使用 <strong>硅基流动 DeepSeek-R1-0528-Qwen3-8B 模型</strong></p>
                            </div>
                            """
                        )

                    # 右侧对话区
                    with gr.Column(scale=7, elem_classes="right-panel"):
                        gr.Markdown("## 📝 对话记录")
                        
                        # 对话记录显示区
                        chatbot = gr.Chatbot(
                            label="对话历史",
                            height=600,
                            elem_classes="chat-container",
                            show_label=False
                        )
                        
                        status_display = gr.HTML("", elem_id="status-display")
                        gr.Markdown("""
                        <div class="footer-note">
                            *回答生成可能需要1-2分钟，请耐心等待<br>
                            *支持多轮对话，可基于前文继续提问
                        </div>
                        """)
            
            # 第二个选项卡：分块可视化
            with gr.TabItem("📊 分块可视化"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## 💡 系统模型信息")
                        
                        # 显示系统模型信息卡片
                        models_info = get_system_models_info()
                        with gr.Group(elem_classes="model-card"):
                            gr.Markdown("### 核心模型与技术")
                            
                            for key, value in models_info.items():
                                with gr.Row():
                                    gr.Markdown(f"**{key}**:", elem_classes="label")
                                    gr.Markdown(f"{value}", elem_classes="value")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("## 📄 文档分块统计")
                        refresh_chunks_btn = gr.Button("🔄 刷新分块数据", variant="primary")
                        chunks_status = gr.Markdown("点击按钮查看分块统计")
                
                # 分块数据表格和详情
                with gr.Row():
                    chunks_data = gr.Dataframe(
                        headers=["来源", "序号", "字符数", "分词数", "内容预览"],
                        elem_classes="chunk-table",
                        interactive=False,
                        wrap=True,
                        row_count=(10, "dynamic")
                    )
                
                with gr.Row():
                    chunk_detail_text = gr.Textbox(
                        label="分块详情",
                        placeholder="点击表格中的行查看完整内容...",
                        lines=8,
                        elem_classes="chunk-detail-box"
                    )
                    
                gr.Markdown("""
                <div class="footer-note">
                    * 点击表格中的行可查看该分块的完整内容<br>
                    * 分词数表示使用jieba分词后的token数量
                </div>
                """)

        # 绑定UI事件
        upload_btn.click(
            pdf_processor.process_multiple_pdfs,
            inputs=[file_input],
            outputs=[upload_status, file_list],
            show_progress=True
        )

        ask_btn.click(
            process_chat,
            inputs=[question_input, chatbot, web_search_checkbox],
            outputs=[chatbot, question_input, api_info]
        )

        clear_btn.click(
            clear_chat_history,
            inputs=[],
            outputs=[chatbot, status_display]
        )

        web_search_checkbox.change(
            update_api_info,
            inputs=[web_search_checkbox],
            outputs=[api_info]
        )
        
        refresh_chunks_btn.click(
            fn=get_document_chunks,
            outputs=[chunks_data, chunks_status]
        )
        
        chunks_data.select(
            fn=show_chunk_details,
            inputs=[chunks_data],
            outputs=[chunk_detail_text]
        )

    return demo

def main():
    """主函数"""
    # 检查环境
    if not check_environment():
        logger.error("环境检查失败！请确保硅基流动API密钥已配置且连接正常")
        return
    
    # 查找可用端口
    selected_port = find_available_port()
    if not selected_port:
        logger.error("所有端口都被占用，请手动释放端口")
        return
    
    try:
        # 创建UI
        demo = create_ui()
        
        # 设置JavaScript
        demo._js = """
        function gradioApp() {
            // 设置默认主题为暗色
            document.documentElement.setAttribute('data-theme', 'dark');
            
            const observer = new MutationObserver((mutations) => {
                document.getElementById("loading").style.display = "none";
                const progress = document.querySelector('.progress-text');
                if (progress) {
                    const percent = document.querySelector('.progress > div')?.innerText || '';
                    const step = document.querySelector('.progress-description')?.innerText || '';
                    document.getElementById('current-step').innerText = step;
                    document.getElementById('progress-percent').innerText = percent;
                }
            });
            observer.observe(document.body, {childList: true, subtree: true});
        }

        function toggleTheme() {
            const root = document.documentElement;
            const currentTheme = root.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            root.setAttribute('data-theme', newTheme);
        }

        // 初始化主题
        document.addEventListener('DOMContentLoaded', () => {
            document.documentElement.setAttribute('data-theme', 'dark');
        });
        """
        
        # 打开浏览器
        webbrowser.open(f"http://127.0.0.1:{selected_port}")
        
        # 启动服务
        logger.info(f"正在启动服务，端口: {selected_port}")
        demo.launch(
            server_port=selected_port,
            server_name="0.0.0.0",
            show_error=True,
            ssl_verify=False,
            height=900
        )
        
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")

if __name__ == "__main__":
    main()