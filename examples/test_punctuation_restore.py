#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文标点符号恢复功能测试示例

功能：
- 测试标点恢复模块的基本功能
- 演示如何在实际应用中使用标点恢复
- 提供不同场景的使用示例
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fireredasr.utils.punctuation_restore import PunctuationRestorer, restore_punctuation_from_file


def test_basic_punctuation():
    """测试基本的标点恢复功能"""
    print("=" * 60)
    print("🔤 测试基本标点恢复功能")
    print("=" * 60)
    
    # 测试文本（无标点）
    test_texts = [
        "今天天气真好我们一起去公园玩吧",
        "你吃饭了吗还没有的话我们一起去吃饭",
        "这个产品真的很不错价格也很合理你要不要试试看",
        "哇这个太厉害了怎么做到的",
        "请问现在几点了我的手表停了"
    ]
    
    # 创建标点恢复器
    print("\n初始化标点恢复器...")
    restorer = PunctuationRestorer()
    
    # 处理每个测试文本
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i}:")
        print(f"原始文本: {text}")
        
        # 恢复标点
        punctuated = restorer.restore_punctuation(text)
        print(f"带标点文本: {punctuated}")
    
    # 清理资源
    restorer.cleanup()
    print("\n✅ 基本功能测试完成")


def test_long_text_punctuation():
    """测试长文本的标点恢复"""
    print("\n" + "=" * 60)
    print("📜 测试长文本标点恢复")
    print("=" * 60)
    
    # 长文本示例（会议记录）
    long_text = """
    各位同事大家好今天我们开会讨论一下新产品的发布计划首先让我们看一下市场调研的结果
    根据最新的数据显示我们的目标用户群体主要集中在25到35岁之间他们对产品的功能性和设计感都有较高要求
    产品部门已经完成了初步设计方案包括三个不同的版本基础版专业版和企业版每个版本都有不同的功能配置
    市场部建议我们先在一线城市进行试点然后逐步推广到二三线城市这样可以更好地控制风险
    财务部门预计整个项目需要投入500万的预算其中产品开发占60%市场推广占30%其他费用占10%
    时间安排上我们计划在下个季度完成产品开发第三季度开始市场推广预计年底前可以实现盈亏平衡
    大家对这个计划有什么意见或建议吗如果没有问题的话我们就按照这个方案执行
    """
    
    # 创建标点恢复器
    restorer = PunctuationRestorer()
    
    print("\n原始文本（前200字）:")
    print(long_text.strip()[:200] + "...")
    
    # 恢复标点
    print("\n正在处理长文本...")
    punctuated = restorer.restore_punctuation(long_text.strip())
    
    print("\n带标点文本（前300字）:")
    print(punctuated[:300] + "...")
    
    # 统计标点符号
    punctuation_counts = {
        '，': punctuated.count('，'),
        '。': punctuated.count('。'),
        '？': punctuated.count('？'),
        '！': punctuated.count('！'),
        '、': punctuated.count('、'),
        '；': punctuated.count('；')
    }
    
    print("\n标点统计:")
    for punct, count in punctuation_counts.items():
        if count > 0:
            print(f"  {punct} : {count} 个")
    
    # 清理资源
    restorer.cleanup()
    print("\n✅ 长文本测试完成")


def test_file_processing():
    """测试文件处理功能"""
    print("\n" + "=" * 60)
    print("📁 测试文件处理功能")
    print("=" * 60)
    
    # 创建测试文件
    test_input_file = Path("test_input.txt")
    test_output_file = Path("test_output_with_punctuation.txt")
    
    # 写入测试文本
    test_content = """
    这是一个测试文件用来演示标点恢复功能
    我们可以处理多行文本每一行都会被正确地添加标点符号
    无论是陈述句疑问句还是感叹句都可以被正确识别
    这个功能对于语音识别后的文本处理特别有用
    让我们看看效果如何
    """
    
    with open(test_input_file, 'w', encoding='utf-8') as f:
        f.write(test_content.strip())
    
    print(f"\n创建测试文件: {test_input_file}")
    
    # 处理文件
    print("处理文件...")
    success = restore_punctuation_from_file(
        str(test_input_file),
        str(test_output_file)
    )
    
    if success:
        print(f"✅ 处理成功，输出文件: {test_output_file}")
        
        # 读取并显示结果
        with open(test_output_file, 'r', encoding='utf-8') as f:
            result = f.read()
        
        print("\n处理结果:")
        print(result)
    else:
        print("❌ 处理失败")
    
    # 清理测试文件
    if test_input_file.exists():
        test_input_file.unlink()
    if test_output_file.exists():
        test_output_file.unlink()
    
    print("\n✅ 文件处理测试完成")


def test_error_handling():
    """测试错误处理机制"""
    print("\n" + "=" * 60)
    print("🛡️ 测试错误处理机制")
    print("=" * 60)
    
    restorer = PunctuationRestorer()
    
    # 测试空文本
    print("\n测试空文本:")
    result = restorer.restore_punctuation("")
    print(f"空文本结果: '{result}' (应该返回空字符串)")
    
    # 测试纯空格
    print("\n测试纯空格:")
    result = restorer.restore_punctuation("   ")
    print(f"纯空格结果: '{result}'")
    
    # 测试特殊字符
    print("\n测试包含特殊字符的文本:")
    text_with_special = "这是一个测试123abc@#$%"
    result = restorer.restore_punctuation(text_with_special)
    print(f"原始: {text_with_special}")
    print(f"结果: {result}")
    
    # 清理资源
    restorer.cleanup()
    print("\n✅ 错误处理测试完成")


def main():
    """主函数"""
    print("🔥 FireRedASR 中文标点符号恢复功能测试")
    print("=" * 60)
    
    try:
        # 运行各项测试
        test_basic_punctuation()
        test_long_text_punctuation()
        test_file_processing()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()