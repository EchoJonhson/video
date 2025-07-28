#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import List

class ParagraphSegmenter:
    """中文自然段分段器 - 最简规则版本"""
    
    def __init__(self, min_length: int = 50, max_length: int = 500):
        self.min_length = min_length
        self.max_length = max_length
        
        # 话题转换标志词
        self.transition_words = {
            "另外", "然后", "接下来", "首先", "其次", "最后",
            "同时", "此外", "不过", "但是", "然而", "总之"
        }
        
    def segment_paragraphs(self, text: str) -> List[str]:
        """
        将文本分割成自然段
        
        Args:
            text: 带标点的文本
            
        Returns:
            段落列表
        """
        # 按句号、问号、感叹号分句
        sentences = re.split(r'([。！？])', text)
        
        # 重组句子（保留标点）
        full_sentences = []
        for i in range(0, len(sentences)-1, 2):
            if sentences[i].strip():
                full_sentences.append(sentences[i] + sentences[i+1])
        
        # 分段逻辑
        paragraphs = []
        current_para = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查是否应该分段
            should_split = False
            
            # 1. 当前段落已经足够长
            if len(current_para) > self.min_length:
                # 2. 遇到话题转换词
                if any(word in sentence for word in self.transition_words):
                    should_split = True
                # 3. 段落过长
                elif len(current_para) > self.max_length:
                    should_split = True
            
            if should_split and current_para:
                paragraphs.append(current_para.strip())
                current_para = sentence
            else:
                current_para += sentence
        
        # 添加最后一段
        if current_para.strip():
            paragraphs.append(current_para.strip())
        
        return paragraphs


# 测试函数
def test_segmenter():
    text = """首先，让我们来讨论一下今天的主题。这个问题非常重要，需要我们认真对待。
    在过去的几年里，我们看到了很多变化。技术的发展速度越来越快。
    另外，市场环境也在不断变化。我们需要适应这些变化才能生存下去。
    然后，我们来看看具体的解决方案。第一步是要明确目标。第二步是制定计划。
    最后，执行力是关键。没有执行，再好的计划也是空谈。"""
    
    segmenter = ParagraphSegmenter()
    paragraphs = segmenter.segment_paragraphs(text)
    
    print("=== 分段结果 ===")
    for i, para in enumerate(paragraphs, 1):
        print(f"\n段落 {i}:")
        print(para)
        print(f"（长度：{len(para)} 字）")


if __name__ == "__main__":
    test_segmenter()