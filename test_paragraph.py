#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试分段功能的独立脚本"""

from fireredasr.utils.paragraph_segmentation import ParagraphSegmenter

# 模拟一段转写文本（带标点）
test_text = """
大家好，欢迎来到今天的会议。今天我们要讨论三个重要议题。首先是关于公司的战略调整。
在过去的一年里，我们取得了很多成就。销售额增长了百分之三十。客户满意度也有了显著提升。
然而，市场环境正在发生变化。竞争对手的动作越来越快。我们必须要加快创新的步伐。
接下来，让我们谈谈产品开发的问题。目前我们有五个新产品正在研发中。其中两个已经进入测试阶段。
预计在下个季度就能推向市场。这些产品将帮助我们扩大市场份额。同时也能提升品牌影响力。
最后，我想强调一下团队建设的重要性。人才是公司最宝贵的资产。我们需要持续投资员工培训。
只有这样，才能保持竞争优势。谢谢大家的聆听。现在开始自由讨论时间。
"""

# 创建分段器
segmenter = ParagraphSegmenter(min_length=50, max_length=300)

# 执行分段
paragraphs = segmenter.segment_paragraphs(test_text.strip())

# 输出结果
print("=== 分段测试结果 ===")
print(f"原文长度: {len(test_text.strip())} 字")
print(f"分段数量: {len(paragraphs)} 段\n")

for i, para in enumerate(paragraphs, 1):
    print(f"【第{i}段】({len(para)}字)")
    print(para)
    print()

# 测试不同参数
print("\n=== 测试不同参数 ===")
segmenter2 = ParagraphSegmenter(min_length=100, max_length=200)
paragraphs2 = segmenter2.segment_paragraphs(test_text.strip())
print(f"参数(100-200字): 分为{len(paragraphs2)}段")

segmenter3 = ParagraphSegmenter(min_length=30, max_length=100)
paragraphs3 = segmenter3.segment_paragraphs(test_text.strip())
print(f"参数(30-100字): 分为{len(paragraphs3)}段")