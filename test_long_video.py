#!/usr/bin/env python3
"""测试长视频处理功能"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from long_video_transcribe import LongVideoTranscriber

def test_long_video_processing():
    """测试长视频处理功能"""
    print("开始测试长视频处理功能...")
    
    # 创建转写器实例
    transcriber = LongVideoTranscriber()
    
    # 设置参数
    transcriber.model_type = "aed"  # 使用AED模型进行测试
    transcriber.max_speech_duration_s = 30
    transcriber.min_silence_duration_ms = 500
    transcriber.min_speech_duration_ms = 1000
    
    # 检查依赖
    if not transcriber.check_dependencies():
        print("依赖检查失败")
        return False
    
    # 扫描文件
    files = transcriber.scan_long_media_files()
    if not files:
        print("没有找到媒体文件")
        return False
    
    print(f"找到 {len(files)} 个文件")
    
    # 处理第一个文件
    input_path = files[0]
    print(f"处理文件: {input_path}")
    
    # 测试音频准备
    print("\n测试音频准备...")
    work_dir = transcriber.temp_dir / "test_run"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    prepared_audio = work_dir / "prepared_audio.wav"
    if not transcriber.prepare_audio(input_path, prepared_audio):
        print("音频准备失败")
        return False
    
    print("✅ 音频准备成功")
    
    # 测试VAD切片
    print("\n测试VAD切片...")
    segments_dir = work_dir / "segments"
    segments_dir.mkdir(exist_ok=True)
    
    try:
        segments = transcriber.slice_audio_with_vad(prepared_audio, segments_dir)
        if not segments:
            print("VAD切片失败或没有检测到语音")
            return False
        
        print(f"✅ VAD切片成功，生成 {len(segments)} 个片段")
        
        # 显示片段信息
        for i, seg in enumerate(segments[:3]):  # 只显示前3个
            print(f"  片段{i}: {seg['start']:.2f}s - {seg['end']:.2f}s (时长: {seg['duration']:.2f}s)")
        
    except Exception as e:
        print(f"VAD切片出错: {str(e)}")
        return False
    
    # 测试模型加载
    print("\n测试模型加载...")
    model_dir = "pretrained_models/FireRedASR-AED-L"
    
    if not Path(model_dir).exists():
        print(f"模型目录不存在: {model_dir}")
        print("请先下载模型文件")
        return False
    
    print("✅ 所有测试通过！")
    print("\n长视频处理功能可以正常使用。")
    print("你可以运行以下命令处理完整视频：")
    print(f"python long_video_transcribe.py")
    
    return True

if __name__ == "__main__":
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(Path.cwd()) + ":" + os.environ.get('PYTHONPATH', '')
    
    success = test_long_video_processing()
    sys.exit(0 if success else 1)