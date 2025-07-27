#!/usr/bin/env python3
"""
FireRedASR 视频处理 Python 示例

本脚本展示如何在Python代码中使用FireRedASR处理视频文件
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.utils.video_audio import is_video_file, is_audio_file


def process_media_file(file_path, model_type='aed', use_gpu=True):
    """
    处理单个媒体文件（音频或视频）
    
    Args:
        file_path (str): 文件路径
        model_type (str): 模型类型 ('aed' 或 'llm')
        use_gpu (bool): 是否使用GPU
        
    Returns:
        dict: 识别结果
    """
    print(f"\n{'='*60}")
    print(f"🎬 处理文件: {file_path}")
    print(f"🤖 模型类型: {model_type.upper()}")
    print(f"💻 计算设备: {'GPU' if use_gpu else 'CPU'}")
    
    # 检查文件类型
    if is_video_file(file_path):
        print(f"📹 文件类型: 视频文件")
    elif is_audio_file(file_path) or file_path.endswith('.wav'):
        print(f"🎵 文件类型: 音频文件")
    else:
        print(f"❓ 文件类型: 未知，尝试作为音频处理")
    
    # 模型配置
    model_paths = {
        'aed': 'pretrained_models/FireRedASR-AED-L',
        'llm': 'pretrained_models/FireRedASR-LLM-L'
    }
    
    # 解码配置
    decode_configs = {
        'aed': {
            "use_gpu": 1 if use_gpu else 0,
            "beam_size": 3,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.25,
            "aed_length_penalty": 0.6,
            "eos_penalty": 1.0
        },
        'llm': {
            "use_gpu": 1 if use_gpu else 0,
            "beam_size": 3,
            "decode_max_len": 0,
            "decode_min_len": 0,
            "repetition_penalty": 3.0,
            "llm_length_penalty": 1.0,
            "temperature": 1.0
        }
    }
    
    try:
        # 检查模型路径
        model_dir = model_paths[model_type]
        if not os.path.exists(model_dir):
            print(f"❌ 错误: 模型目录不存在: {model_dir}")
            print("请从 https://huggingface.co/fireredteam 下载模型文件")
            return None
        
        # 加载模型
        print(f"🔄 正在加载模型...")
        start_time = time.time()
        model = FireRedAsr.from_pretrained(model_type, model_dir)
        load_time = time.time() - start_time
        print(f"✅ 模型加载完成 (耗时: {load_time:.2f}s)")
        
        # 进行推理
        print(f"🔄 正在进行语音识别...")
        uttid = Path(file_path).stem
        
        start_time = time.time()
        results = model.transcribe([uttid], [file_path], decode_configs[model_type])
        inference_time = time.time() - start_time
        
        # 输出结果
        if results and len(results) > 0:
            result = results[0]
            print(f"✅ 识别完成 (耗时: {inference_time:.2f}s)")
            print(f"📝 识别结果: {result['text']}")
            return result
        else:
            print(f"❌ 识别失败: 没有返回结果")
            return None
            
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        return None
    finally:
        # 清理临时文件
        if 'model' in locals():
            model.feat_extractor.cleanup_temp_files()


def batch_process_directory(directory, model_type='aed', use_gpu=True):
    """
    批量处理目录中的媒体文件
    
    Args:
        directory (str): 目录路径
        model_type (str): 模型类型
        use_gpu (bool): 是否使用GPU
        
    Returns:
        list: 处理结果列表
    """
    print(f"\n{'='*60}")
    print(f"📁 批量处理目录: {directory}")
    
    # 支持的媒体格式
    media_extensions = {
        '.wav', '.mp3', '.flac', '.m4a', '.aac',  # 音频
        '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'  # 视频
    }
    
    # 查找媒体文件
    media_files = []
    for ext in media_extensions:
        media_files.extend(Path(directory).rglob(f'*{ext}'))
    
    print(f"🔍 找到 {len(media_files)} 个媒体文件")
    
    if not media_files:
        print("❌ 目录中没有找到媒体文件")
        return []
    
    results = []
    for i, file_path in enumerate(media_files, 1):
        print(f"\n📋 处理进度: {i}/{len(media_files)}")
        result = process_media_file(str(file_path), model_type, use_gpu)
        if result:
            results.append(result)
    
    return results


def main():
    """主函数"""
    print("🔥 FireRedASR 视频处理 Python 示例")
    print("=" * 60)
    
    # 示例文件路径
    examples = [
        "examples/wav/BAC009S0764W0121.wav",  # 音频示例
        "examples/video/sample.mp4",           # 视频示例（如果存在）
        "examples/video/demo.avi",             # 视频示例（如果存在）
    ]
    
    # 查找实际存在的文件
    available_files = []
    for file_path in examples:
        if os.path.exists(file_path):
            available_files.append(file_path)
    
    if not available_files:
        print("❌ 没有找到示例文件")
        print("请确保以下文件之一存在:")
        for file_path in examples:
            print(f"  - {file_path}")
        return
    
    print("📝 可用的示例文件:")
    for i, file_path in enumerate(available_files, 1):
        file_type = "视频" if is_video_file(file_path) else "音频"
        print(f"  {i}. {file_path} ({file_type})")
    
    # 处理第一个可用文件
    test_file = available_files[0]
    print(f"\n🎯 使用示例文件: {test_file}")
    
    # 测试 AED 模型
    print("\n" + "="*60)
    print("🧪 测试 FireRedASR-AED 模型")
    result_aed = process_media_file(test_file, 'aed', use_gpu=True)
    
    # 测试 LLM 模型（如果AED成功的话）
    if result_aed:
        print("\n" + "="*60)
        print("🧪 测试 FireRedASR-LLM 模型")
        result_llm = process_media_file(test_file, 'llm', use_gpu=True)
    
    # 批量处理示例（如果有视频目录的话）
    video_dir = "examples/video"
    if os.path.exists(video_dir):
        print("\n" + "="*60)
        print("🗂️  批量处理示例")
        batch_results = batch_process_directory(video_dir, 'aed', use_gpu=True)
        print(f"✅ 批量处理完成，共处理 {len(batch_results)} 个文件")
    
    print("\n" + "="*60)
    print("✅ 所有示例执行完成！")
    print("\n💡 在您的代码中使用 FireRedASR:")
    print("""
from fireredasr.models.fireredasr import FireRedAsr

# 加载模型
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")

# 处理单个文件（音频或视频）
results = model.transcribe(
    ["my_video"],
    ["path/to/my_video.mp4"],
    {
        "use_gpu": 1,
        "beam_size": 3,
        "nbest": 1,
        "decode_max_len": 0,
        "softmax_smoothing": 1.25,
        "aed_length_penalty": 0.6,
        "eos_penalty": 1.0
    }
)

print(results[0]['text'])
""")


if __name__ == "__main__":
    main()