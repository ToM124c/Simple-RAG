"""
问答引擎核心模块
"""
import logging
import re
from typing import List, Dict, Any, Tuple, Generator
from storage.vector_store import vector_store
from retrieval.hybrid_retriever import hybrid_retriever
from retrieval.reranker import reranker
from services.web_search import web_search_service
from services.llm_service import llm_service
from config import MAX_RETRIEVAL_ITERATIONS

logger = logging.getLogger(__name__)

class QAEngine:
    """问答引擎核心类"""
    
    def __init__(self):
        pass
    
    def recursive_retrieval(self, initial_query: str, max_iterations: int = None, 
                           enable_web_search: bool = False) -> Tuple[List[str], List[str], List[Dict]]:
        """递归检索与迭代查询"""
        max_iterations = max_iterations or MAX_RETRIEVAL_ITERATIONS
        
        query = initial_query
        all_contexts = []
        all_doc_ids = []
        all_metadata = []
        
        for i in range(max_iterations):
            logger.info(f"递归检索迭代 {i+1}/{max_iterations}，当前查询: {query}")
            
            # 网络搜索
            web_results_texts = []
            if enable_web_search and web_search_service.check_api_key():
                try:
                    web_results_texts = web_search_service.get_web_context(query)
                except Exception as e:
                    logger.error(f"网络搜索错误: {str(e)}")
            
            # 混合检索
            hybrid_results = hybrid_retriever.retrieve(query)
            
            # 提取文档信息
            doc_ids_current_iter = []
            docs_current_iter = []
            metadata_list_current_iter = []
            
            if hybrid_results:
                for doc_id, result_data in hybrid_results[:10]:
                    doc_ids_current_iter.append(doc_id)
                    docs_current_iter.append(result_data['content'])
                    metadata_list_current_iter.append(result_data['metadata'])
            
            # 重排序
            if docs_current_iter:
                try:
                    reranked_results = reranker.rerank(
                        query, docs_current_iter, doc_ids_current_iter, metadata_list_current_iter
                    )
                except Exception as e:
                    logger.error(f"重排序错误: {str(e)}")
                    reranked_results = [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0}) 
                                      for doc_id, doc, meta in zip(doc_ids_current_iter, docs_current_iter, metadata_list_current_iter)]
            else:
                reranked_results = []
            
            # 收集上下文
            current_contexts_for_llm = web_results_texts[:]
            for doc_id, result_data in reranked_results:
                doc = result_data['content']
                metadata = result_data['metadata']
                
                if doc_id not in all_doc_ids:
                    all_doc_ids.append(doc_id)
                    all_contexts.append(doc)
                    all_metadata.append(metadata)
                current_contexts_for_llm.append(doc)
            
            # 检查是否需要继续迭代
            if i == max_iterations - 1:
                break
            
            if current_contexts_for_llm:
                current_summary = "\n".join(current_contexts_for_llm[:3]) if current_contexts_for_llm else "未找到相关信息"
                
                # 仅传递两个参数：初始问题和当前摘要
                next_query = llm_service.generate_next_query(initial_query, current_summary)
                
                if "不需要" in next_query or "不需要进一步查询" in next_query or len(next_query) < 5:
                    logger.info("LLM判断不需要进一步查询，结束递归检索")
                    break
                
                query = next_query
                logger.info(f"生成新查询: {query}")
            else:
                break
        
        return all_contexts, all_doc_ids, all_metadata
    
    def detect_conflicts(self, sources: List[Dict[str, Any]]) -> bool:
        """检测信息矛盾"""
        key_facts = {}
        for item in sources:
            facts = self._extract_facts(item['text'] if 'text' in item else item.get('excerpt', ''))
            for fact, value in facts.items():
                if fact in key_facts:
                    if key_facts[fact] != value:
                        return True
                else:
                    key_facts[fact] = value
        return False
    
    def _extract_facts(self, text: str) -> Dict[str, Any]:
        """从文本提取关键事实"""
        facts = {}
        # 提取数值型事实
        numbers = re.findall(r'\b\d{4}年|\b\d+%', text)
        if numbers:
            facts['关键数值'] = numbers
        # 提取技术术语
        if "产业图谱" in text:
            facts['技术方法'] = list(set(re.findall(r'[A-Za-z]+模型|[A-Z]{2,}算法', text)))
        return facts
    
    def evaluate_source_credibility(self, source: Dict[str, Any]) -> float:
        """评估来源可信度"""
        credibility_scores = {
            "gov.cn": 0.9,
            "edu.cn": 0.85,
            "weixin": 0.7,
            "zhihu": 0.6,
            "baidu": 0.5
        }
        
        url = source.get('url', '')
        if not url:
            return 0.5
        
        domain_match = re.search(r'//([^/]+)', url)
        if not domain_match:
            return 0.5
        
        domain = domain_match.group(1)
        
        for known_domain, score in credibility_scores.items():
            if known_domain in domain:
                return score
        
        return 0.5
    
    def build_context_with_sources(self, contexts: List[str], doc_ids: List[str], 
                                   metadata_list: List[Dict], enable_web_search: bool = False) -> Tuple[str, List[Dict]]:
        """构建带来源信息的上下文"""
        context_with_sources = []
        sources_for_conflict_detection = []
        
        for doc, doc_id, metadata in zip(contexts, doc_ids, metadata_list):
            source_type = metadata.get('source', '本地文档')
            
            source_item = {
                'text': doc,
                'type': source_type
            }
            
            if source_type == 'web':
                url = metadata.get('url', '未知URL')
                title = metadata.get('title', '未知标题')
                context_with_sources.append(f"[网络来源: {title}] (URL: {url})\n{doc}")
                source_item['url'] = url
                source_item['title'] = title
            else:
                source = metadata.get('source', '未知来源')
                context_with_sources.append(f"[本地文档: {source}]\n{doc}")
                source_item['source'] = source
            
            sources_for_conflict_detection.append(source_item)
        
        context = "\n\n".join(context_with_sources)
        return context, sources_for_conflict_detection
    
    def generate_prompt(self, question: str, context: str, enable_web_search: bool = False, 
                       knowledge_base_exists: bool = True, conflict_detected: bool = False) -> str:
        """生成提示词"""
        # 添加时间敏感检测
        time_sensitive = any(word in question for word in ["最新", "今年", "当前", "最近", "刚刚"])
        
        prompt_template = """作为一个专业的问答助手，你需要基于以下{context_type}回答用户问题。

提供的参考内容：
{context}

用户问题：{question}

请遵循以下回答原则：
1. 仅基于提供的参考内容回答问题，不要使用你自己的知识
2. 如果参考内容中没有足够信息，请坦诚告知你无法回答
3. 回答应该全面、准确、有条理，并使用适当的段落和结构
4. 请用中文回答
5. 在回答末尾标注信息来源{time_instruction}{conflict_instruction}

请现在开始回答："""
        
        return prompt_template.format(
            context_type="本地文档和网络搜索结果" if enable_web_search and knowledge_base_exists else ("网络搜索结果" if enable_web_search else "本地文档"),
            context=context if context else ("网络搜索结果将用于回答。" if enable_web_search and not knowledge_base_exists else "知识库为空或未找到相关内容。"),
            question=question,
            time_instruction="，优先使用最新的信息" if time_sensitive and enable_web_search else "",
            conflict_instruction="，并明确指出不同来源的差异" if conflict_detected else ""
        )
    
    def process_thinking_content(self, text: str) -> str:
        """处理思维链内容"""
        if text is None:
            return ""
        
        if not isinstance(text, str):
            try:
                processed_text = str(text)
            except:
                return "无法处理的内容格式"
        else:
            processed_text = text
        
        try:
            while "<think>" in processed_text and "</think>" in processed_text:
                start_idx = processed_text.find("<think>")
                end_idx = processed_text.find("</think>")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    thinking_content = processed_text[start_idx + 7:end_idx]
                    before_think = processed_text[:start_idx]
                    after_think = processed_text[end_idx + 8:]
                    
                    processed_text = before_think + "\n\n<details>\n<summary>思考过程（点击展开）</summary>\n\n" + thinking_content + "\n\n</details>\n\n" + after_think
            
            # 处理其他HTML标签
            processed_html = []
            i = 0
            while i < len(processed_text):
                if processed_text[i:i+8] == "<details" or processed_text[i:i+9] == "</details" or \
                   processed_text[i:i+8] == "<summary" or processed_text[i:i+9] == "</summary":
                    tag_end = processed_text.find(">", i)
                    if tag_end != -1:
                        processed_html.append(processed_text[i:tag_end+1])
                        i = tag_end + 1
                        continue
                
                if processed_text[i] == "<":
                    processed_html.append("&lt;")
                elif processed_text[i] == ">":
                    processed_html.append("&gt;")
                else:
                    processed_html.append(processed_text[i])
                i += 1
            
            processed_text = "".join(processed_html)
        except Exception as e:
            logger.error(f"处理思维链内容时出错: {str(e)}")
            try:
                return text.replace("<", "&lt;").replace(">", "&gt;")
            except:
                return "处理内容时出错"
        
        return processed_text
    
    def answer_question(self, question: str, enable_web_search: bool = False) -> str:
        """回答问题"""
        try:
            # 检查知识库
            knowledge_base_exists = not vector_store.is_empty()
            if not knowledge_base_exists and not enable_web_search:
                return "⚠️ 知识库为空，请先上传文档。"
            
            # 递归检索
            all_contexts, all_doc_ids, all_metadata = self.recursive_retrieval(
                question, MAX_RETRIEVAL_ITERATIONS, enable_web_search
            )
            
            # 构建上下文
            context, sources_for_conflict_detection = self.build_context_with_sources(
                all_contexts, all_doc_ids, all_metadata, enable_web_search
            )
            
            # 检测矛盾
            conflict_detected = self.detect_conflicts(sources_for_conflict_detection)
            
            # 生成提示词
            prompt = self.generate_prompt(question, context, enable_web_search, knowledge_base_exists, conflict_detected)
            
            # 生成回答（硅基流动）
            answer = llm_service.generate_answer(prompt)
            
            # 处理思维链
            processed_answer = self.process_thinking_content(answer)
            
            return processed_answer
            
        except Exception as e:
            logger.error(f"问答处理失败: {str(e)}")
            return f"系统错误: {str(e)}"
    
    def stream_answer(self, question: str, enable_web_search: bool = False) -> Generator[Tuple[str, str], None, None]:
        """流式回答问题（硅基流动API，当前以单次返回为主）"""
        try:
            # 检查知识库
            knowledge_base_exists = not vector_store.is_empty()
            if not knowledge_base_exists and not enable_web_search:
                yield "⚠️ 知识库为空，请先上传文档。", "遇到错误"
                return
            
            # 递归检索
            all_contexts, all_doc_ids, all_metadata = self.recursive_retrieval(
                question, MAX_RETRIEVAL_ITERATIONS, enable_web_search
            )
            
            # 构建上下文
            context, sources_for_conflict_detection = self.build_context_with_sources(
                all_contexts, all_doc_ids, all_metadata, enable_web_search
            )
            
            # 检测矛盾
            conflict_detected = self.detect_conflicts(sources_for_conflict_detection)
            
            # 生成提示词
            prompt = self.generate_prompt(question, context, enable_web_search, knowledge_base_exists, conflict_detected)
            
            # 生成回答（非流式，一次性返回）
            answer = llm_service.generate_answer(prompt, temperature=0.7, max_tokens=1536)
            processed_answer = self.process_thinking_content(answer)
            yield processed_answer, "完成!"
            
        except Exception as e:
            yield f"系统错误: {str(e)}", "遇到错误"

# 全局问答引擎实例
qa_engine = QAEngine()
