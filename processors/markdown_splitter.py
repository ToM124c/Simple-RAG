"""
Markdown分割处理模块
"""
import re
import os
import logging
from typing import List, Dict, Any
from config import MIN_SPLIT_LENGTH, MAX_SPLIT_LENGTH

logger = logging.getLogger(__name__)

class MarkdownSplitter:
    """Markdown文档分割器"""
    
    def __init__(self, min_split_length=None, max_split_length=None):
        self.min_split_length = min_split_length or MIN_SPLIT_LENGTH
        self.max_split_length = max_split_length or MAX_SPLIT_LENGTH
        self.outline = []
    
    def parse_markdown(self, md_text: str) -> List[Dict[str, Any]]:
        """解析Markdown文本，将其分割为包含标题和内容的段落"""
        lines = md_text.split('\n')
        sections = []
        current_section = None
        in_code_block = False
        position = 0
        
        for i, line in enumerate(lines):
            # 检测代码块开始/结束
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
            
            # 不在代码块中且检测到标题
            if not in_code_block and line.startswith('#'):
                # 保存当前段落（如果有）
                if current_section is not None:
                    current_section['content'] = current_section['content'].strip()
                    sections.append(current_section)
                
                # 解析新标题
                match = re.match(r'(#+)\s*(.*)', line)
                if match:
                    level = len(match.group(1))
                    heading = match.group(2).strip()
                    self.outline.append({'heading': heading, 'level': level, 'position': position})
                    
                    # 创建新段落
                    current_section = {
                        'heading': heading,
                        'level': level,
                        'content': '',
                        'position': position
                    }
                    position += 1
                continue
            
            # 添加到当前段落内容
            if current_section is not None:
                current_section['content'] += line + '\n'
            else:
                # 文档开头没有标题的情况
                if line.strip():
                    current_section = {
                        'heading': '',
                        'level': 0,
                        'content': line + '\n',
                        'position': position
                    }
                    position += 1
        
        # 添加最后一个段落
        if current_section is not None:
            current_section['content'] = current_section['content'].strip()
            sections.append(current_section)
        
        return sections
    
    def split_long_section(self, section: Dict[str, Any]) -> List[str]:
        """分割超长段落"""
        content = section['content']
        # 按双换行分割段落
        paragraphs = re.split(r'\n\n+', content)
        result = []
        current_chunk = ''
        
        for paragraph in paragraphs:
            # 超长段落需要进一步分割
            if len(paragraph) > self.max_split_length:
                # 保存当前块（如果有）
                if current_chunk:
                    result.append(current_chunk)
                    current_chunk = ''
                
                # 按句子分割
                sentences = re.split(r'(?<=[.!?。！？])\s+', paragraph)
                if not sentences:
                    sentences = [paragraph]
                
                # 处理每个句子
                sentence_chunk = ''
                for sentence in sentences:
                    # 如果句子本身超长
                    if len(sentence) > self.max_split_length:
                        if sentence_chunk:
                            result.append(sentence_chunk)
                            sentence_chunk = ''
                        
                        # 按固定长度分割超长句子
                        start = 0
                        while start < len(sentence):
                            end = start + self.max_split_length
                            result.append(sentence[start:end])
                            start = end
                    # 正常句子
                    else:
                        # 添加到当前句子块
                        if len(sentence_chunk) + len(sentence) + 1 <= self.max_split_length:
                            sentence_chunk = sentence_chunk + ' ' + sentence if sentence_chunk else sentence
                        else:
                            if sentence_chunk:
                                result.append(sentence_chunk)
                            sentence_chunk = sentence
                
                # 添加最后一个句子块
                if sentence_chunk:
                    result.append(sentence_chunk)
            # 正常段落
            else:
                # 添加到当前块
                if len(current_chunk) + len(paragraph) + 2 <= self.max_split_length:
                    current_chunk = current_chunk + '\n\n' + paragraph if current_chunk else paragraph
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    current_chunk = paragraph
        
        # 添加最后一个块
        if current_chunk:
            result.append(current_chunk)
        
        return result
    
    def generate_summary(self, section: Dict[str, Any], part_index: int = None, total_parts: int = None) -> str:
        """为段落生成摘要"""
        if not section.get('heading'):
            # 无标题段落的摘要
            content_preview = section['content'][:50].replace('\n', ' ') + '...' if len(section['content']) > 50 else section['content']
            summary = f"内容段落: {content_preview}"
        else:
            # 有标题段落的摘要
            summary = f"{'#' * section['level']} {section['heading']}"
            
            # 添加分块信息
            if part_index and total_parts:
                summary += f" - 分块 {part_index}/{total_parts}"
        
        return summary
    
    def process_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """处理所有段落，根据最小和最大分割长度进行分割"""
        result = []
        accumulated_section = None
        
        for section in sections:
            content_length = len(section['content'])
            
            # 小段落累积
            if content_length < self.min_split_length:
                if accumulated_section is None:
                    accumulated_section = {
                        'heading': section['heading'],
                        'level': section['level'],
                        'content': section['content'],
                        'headings': [{'heading': section['heading'], 'level': section['level']}]
                    }
                else:
                    # 合并小段落
                    heading_prefix = f"{'#' * section['level']} {section['heading']}\n" if section['heading'] else ""
                    accumulated_section['content'] += f"\n\n{heading_prefix}{section['content']}"
                    if section['heading']:
                        accumulated_section['headings'].append({
                            'heading': section['heading'],
                            'level': section['level']
                        })
                
                # 检查累积段落是否达到最小长度
                if len(accumulated_section['content']) >= self.min_split_length:
                    # 处理累积段落
                    if len(accumulated_section['content']) > self.max_split_length:
                        # 分割超长的累积段落
                        chunks = self.split_long_section({
                            'content': accumulated_section['content']
                        })
                        for i, chunk in enumerate(chunks):
                            summary = self.generate_summary(
                                accumulated_section, 
                                i + 1, 
                                len(chunks)
                            )
                            result.append({
                                'summary': summary,
                                'content': chunk
                            })
                    else:
                        # 添加累积段落
                        summary = self.generate_summary(accumulated_section)
                        result.append({
                            'summary': summary,
                            'content': accumulated_section['content']
                        })
                    accumulated_section = None
                continue
            
            # 处理累积段落（如果有）
            if accumulated_section is not None:
                # 处理累积段落
                if len(accumulated_section['content']) > self.max_split_length:
                    chunks = self.split_long_section({
                        'content': accumulated_section['content']
                    })
                    for i, chunk in enumerate(chunks):
                        summary = self.generate_summary(
                            accumulated_section, 
                            i + 1, 
                            len(chunks)
                        )
                        result.append({
                            'summary': summary,
                            'content': chunk
                        })
                else:
                    summary = self.generate_summary(accumulated_section)
                    result.append({
                        'summary': summary,
                        'content': accumulated_section['content']
                    })
                accumulated_section = None
            
            # 处理当前段落
            if content_length > self.max_split_length:
                # 分割超长段落
                chunks = self.split_long_section(section)
                for i, chunk in enumerate(chunks):
                    summary = self.generate_summary(section, i + 1, len(chunks))
                    result.append({
                        'summary': summary,
                        'content': chunk
                    })
            else:
                # 正常长度段落
                summary = self.generate_summary(section)
                # 添加标题前缀
                if section['heading']:
                    content = f"{'#' * section['level']} {section['heading']}\n{section['content']}"
                else:
                    content = section['content']
                
                result.append({
                    'summary': summary,
                    'content': content
                })
        
        # 处理最后的累积段落
        if accumulated_section is not None:
            if result and len(result[-1]['content']) + len(accumulated_section['content']) < self.max_split_length:
                # 合并到最后一块
                result[-1]['content'] += f"\n\n{accumulated_section['content']}"
                result[-1]['summary'] += f" + 合并段落"
            else:
                # 添加为新的块
                summary = self.generate_summary(accumulated_section)
                result.append({
                    'summary': summary,
                    'content': accumulated_section['content']
                })
        
        return result
    
    def split_markdown(self, md_text: str) -> List[Dict[str, str]]:
        """分割Markdown文档的主函数"""
        self.outline = []  # 重置大纲
        sections = self.parse_markdown(md_text)
        return self.process_sections(sections)
    
    def save_chunks(self, chunks: List[Dict[str, str]], output_dir: str):
        """将分块保存到文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, chunk in enumerate(chunks):
            with open(os.path.join(output_dir, f"chunk_{i+1}.md"), 'w', encoding='utf-8') as f:
                f.write(f"# 摘要: {chunk['summary']}\n\n")
                f.write(chunk['content'])
        
        # 保存大纲
        with open(os.path.join(output_dir, "outline.txt"), 'w', encoding='utf-8') as f:
            for item in self.outline:
                f.write(f"{'#' * item['level']} {item['heading']}\n")

# 全局Markdown分割器实例
markdown_splitter = MarkdownSplitter()
