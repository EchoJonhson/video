#!/usr/bin/env python3
"""长视频处理演示脚本"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from long_video_transcribe import LongVideoTranscriber

def demo_long_video():
    """演示长视频处理功能"""
    print("🎬 FireRedASR 长视频处理功能演示")
    print("="*60)
    
    # 创建转写器
    transcriber = LongVideoTranscriber()
    
    # 设置默认参数
    transcriber.model_type = "aed"  # 使用AED模型
    transcriber.max_speech_duration_s = 30
    transcriber.min_silence_duration_ms = 500
    transcriber.min_speech_duration_ms = 1000
    
    # 检查依赖
    print("\n步骤1: 检查系统依赖")
    if not transcriber.check_dependencies():
        print("❌ 依赖检查失败，请安装所需依赖")
        return False
    
    # 扫描文件
    print("\n步骤2: 扫描媒体文件")
    files = transcriber.scan_long_media_files()
    
    if not files:
        print("❌ 没有找到大型媒体文件")
        print("提示: 将音视频文件放入 Use/Input/ 文件夹")
        return False
    
    print(f"✅ 找到 {len(files)} 个媒体文件")
    for i, f in enumerate(files, 1):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {i}. {f.name} ({size_mb:.2f} MB)")
    
    # 选择第一个文件进行处理
    input_file = files[0]
    print(f"\n将处理文件: {input_file.name}")
    
    # 模拟处理流程
    print("\n步骤3: 处理流程预览")
    print("1️⃣ 音频准备: 将视频/音频转换为16kHz WAV格式")
    print("2️⃣ VAD切片: 使用Silero VAD检测语音段并切分")
    print("3️⃣ 批量转写: 使用FireRedASR-AED模型转写每个片段")
    print("4️⃣ 结果拼接: 合并所有片段生成完整文本")
    
    print("\n输出格式:")
    print("- 纯文本 (.txt)")
    print("- SRT字幕 (.srt)")
    print("- 带时间戳文本 (_with_timestamps.txt)")
    print("- 统计信息 (_stats.json)")
    
    print("\n" + "="*60)
    print("✅ 长视频处理功能已准备就绪！")
    print("\n运行完整处理:")
    print("python long_video_transcribe.py")
    print("\n或使用自定义参数:")
    print("python long_video_transcribe.py --model_type llm --max_duration 45")
    
    return True

if __name__ == "__main__":
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(Path.cwd()) + ":" + os.environ.get('PYTHONPATH', '')
    
    success = demo_long_video()
    sys.exit(0 if success else 1)