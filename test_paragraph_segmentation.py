#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试自然段分段功能
"""

import os
import subprocess
import tempfile
from pathlib import Path

def test_basic_segmentation():
    """测试基础分段功能"""
    print("=== 测试 1: 基础分段功能 ===")
    
    # 使用 examples 目录中的音频文件
    test_audio = "examples/wav/BAC009S0764W0121.wav"
    
    if not os.path.exists(test_audio):
        print(f"❌ 测试音频文件不存在: {test_audio}")
        return False
    
    # 测试命令
    cmd = [
        "python", "long_audio_transcribe.py",
        "--input_audio", test_audio,
        "--model_dir", "examples/pretrained_models/FireRedASR-AED-L",  # 使用AED模型
        "--enable-paragraph",
        "--min-paragraph-length", "30",
        "--max-paragraph-length", "200"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 转写成功")
            print(f"输出目录中应包含 _paragraphs.txt 文件")
            
            # 检查输出文件
            output_dir = Path("examples/wav/BAC009S0764W0121_output")
            if output_dir.exists():
                paragraph_files = list(output_dir.glob("*_paragraphs.txt"))
                if paragraph_files:
                    print(f"✅ 找到段落文件: {paragraph_files[0].name}")
                    # 读取并显示内容
                    with open(paragraph_files[0], 'r', encoding='utf-8') as f:
                        content = f.read()
                        print("\n段落内容预览:")
                        print("-" * 50)
                        print(content[:500] + "..." if len(content) > 500 else content)
                        print("-" * 50)
                    return True
                else:
                    print("❌ 未找到段落文件")
            else:
                print(f"❌ 输出目录不存在: {output_dir}")
        else:
            print(f"❌ 转写失败")
            print(f"错误信息: {result.stderr}")
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
    
    return False


def test_segmentation_module():
    """测试分段模块本身"""
    print("\n=== 测试 2: 分段模块功能 ===")
    
    from fireredasr.utils.paragraph_segmentation import ParagraphSegmenter
    
    # 测试文本
    test_text = """首先，让我们来讨论一下今天的会议主题。这个问题非常重要，需要我们认真对待。在过去的几年里，我们看到了很多变化。技术的发展速度越来越快，市场环境也在不断变化。
    另外，我们还需要考虑客户的需求。客户的需求是多样化的，我们必须提供个性化的解决方案。这就要求我们不断创新，不断改进我们的产品和服务。
    然后，关于具体的实施方案，我建议分三个阶段进行。第一阶段是调研和规划，第二阶段是开发和测试，第三阶段是部署和优化。每个阶段都有明确的目标和时间节点。
    最后，我想强调的是团队合作的重要性。只有大家齐心协力，才能确保项目的成功。让我们一起努力，创造更好的未来。"""
    
    segmenter = ParagraphSegmenter(min_length=50, max_length=300)
    paragraphs = segmenter.segment_paragraphs(test_text)
    
    print(f"原文长度: {len(test_text)} 字")
    print(f"分段数量: {len(paragraphs)} 段")
    print("\n分段结果:")
    for i, para in enumerate(paragraphs, 1):
        print(f"\n【段落 {i}】({len(para)} 字)")
        print(para)
    
    return True


def test_different_parameters():
    """测试不同参数的效果"""
    print("\n=== 测试 3: 不同参数效果 ===")
    
    from fireredasr.utils.paragraph_segmentation import ParagraphSegmenter
    
    test_text = """今天的天气真好。阳光明媚，微风轻拂。另外，空气也很清新。然后，我决定出去散步。首先，我去了公园。公园里有很多人在运动。其次，我去了湖边。湖水波光粼粼，非常美丽。最后，我在咖啡店休息了一会儿。总之，这是美好的一天。"""
    
    # 测试不同的最小长度
    for min_len in [20, 50, 80]:
        print(f"\n--- 最小长度: {min_len} 字 ---")
        segmenter = ParagraphSegmenter(min_length=min_len, max_length=200)
        paragraphs = segmenter.segment_paragraphs(test_text)
        print(f"分段数: {len(paragraphs)}")
        for i, para in enumerate(paragraphs, 1):
            print(f"段落{i}: {len(para)}字")
    
    return True


def main():
    """主测试函数"""
    print("🚀 开始测试自然段分段功能\n")
    
    # 运行各项测试
    tests = [
        ("分段模块功能", test_segmentation_module),
        ("不同参数效果", test_different_parameters),
        ("基础分段功能", test_basic_segmentation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ 测试 {test_name} 出错: {e}")
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    # 使用说明
    print("\n" + "=" * 60)
    print("📝 使用说明")
    print("=" * 60)
    print("1. 基本用法:")
    print("   python long_audio_transcribe.py <音频文件> --enable-paragraph")
    print("\n2. 自定义参数:")
    print("   --min-paragraph-length 50  # 最小段落长度")
    print("   --max-paragraph-length 500 # 最大段落长度")
    print("\n3. 批量处理:")
    print("   python batch_transcribe.py <输入目录> --enable-paragraph")
    print("\n4. 视频处理:")
    print("   python long_video_transcribe.py <视频文件> --enable-paragraph")


if __name__ == "__main__":
    main()