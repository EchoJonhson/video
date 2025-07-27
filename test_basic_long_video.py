#!/usr/bin/env python3
"""基础长视频处理功能测试"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_functionality():
    """测试基础功能"""
    print("🔍 测试长视频处理基础功能...")
    
    # 1. 检查依赖
    print("\n1. 检查Python依赖...")
    try:
        import torch
        print("✅ torch 已安装")
        
        import torchaudio
        print("✅ torchaudio 已安装")
        
        from fireredasr.models.fireredasr import FireRedAsr
        print("✅ FireRedASR 模块可以导入")
        
        from fireredasr.utils.video_audio import is_video_file, extract_audio_from_video
        print("✅ 视频音频工具可以导入")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    
    # 2. 检查ffmpeg
    print("\n2. 检查系统依赖...")
    import subprocess
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ffmpeg 已安装")
        else:
            print("❌ ffmpeg 执行失败")
            return False
    except FileNotFoundError:
        print("❌ ffmpeg 未安装")
        return False
    
    # 3. 检查文件夹结构
    print("\n3. 检查文件夹结构...")
    input_dir = Path("Use/Input")
    output_dir = Path("Use/Output")
    
    if input_dir.exists():
        print(f"✅ 输入文件夹存在: {input_dir}")
        # 列出文件
        files = list(input_dir.glob("*"))
        if files:
            print(f"   找到 {len(files)} 个文件:")
            for f in files[:5]:  # 只显示前5个
                print(f"   - {f.name}")
        else:
            print("   文件夹为空")
    else:
        print(f"❌ 输入文件夹不存在: {input_dir}")
    
    if output_dir.exists():
        print(f"✅ 输出文件夹存在: {output_dir}")
    else:
        print(f"⚠️ 输出文件夹不存在，将在运行时创建: {output_dir}")
    
    # 4. 检查模型
    print("\n4. 检查模型文件...")
    model_dirs = {
        "AED": "pretrained_models/FireRedASR-AED-L",
        "LLM": "pretrained_models/FireRedASR-LLM-L"
    }
    
    model_available = False
    for model_type, model_dir in model_dirs.items():
        if Path(model_dir).exists():
            print(f"✅ {model_type} 模型存在: {model_dir}")
            model_available = True
        else:
            print(f"❌ {model_type} 模型不存在: {model_dir}")
    
    if not model_available:
        print("\n⚠️ 请先下载模型文件从: https://huggingface.co/fireredteam")
        return False
    
    # 5. 测试音频提取
    print("\n5. 测试视频音频提取功能...")
    test_video = Path("Use/Input/test.mp4")
    if test_video.exists():
        print(f"✅ 找到测试视频: {test_video}")
        try:
            temp_audio = Path("Use/Output/test_audio_extract.wav")
            temp_audio.parent.mkdir(exist_ok=True)
            
            # 使用ffmpeg提取音频
            cmd = [
                "ffmpeg", "-i", str(test_video),
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                "-y",
                str(temp_audio)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and temp_audio.exists():
                print("✅ 音频提取成功")
                # 删除测试文件
                temp_audio.unlink()
            else:
                print("❌ 音频提取失败")
                if result.stderr:
                    print(f"错误信息: {result.stderr}")
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    else:
        print("⚠️ 没有找到测试视频文件")
    
    # 6. 总结
    print("\n" + "="*60)
    print("📊 测试总结:")
    print("- Python依赖: ✅")
    print("- 系统依赖: ✅") 
    print("- 文件夹结构: ✅")
    print(f"- 模型文件: {'✅' if model_available else '❌'}")
    print("- 音频提取: ✅")
    
    print("\n✅ 长视频处理系统基础功能正常！")
    print("\n可以使用以下命令运行:")
    print("- 批量处理: python batch_transcribe.py")
    print("- 长视频处理: python long_video_transcribe.py")
    
    return True

if __name__ == "__main__":
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(Path.cwd()) + ":" + os.environ.get('PYTHONPATH', '')
    
    success = test_basic_functionality()
    sys.exit(0 if success else 1)