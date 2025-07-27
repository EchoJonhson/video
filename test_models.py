#!/usr/bin/env python3
"""
FireRedASR 模型快速测试脚本

支持音频和视频文件测试

用法：
  # 测试音频文件
  python test_models.py --model aed --input examples/wav/BAC009S0764W0121.wav
  python test_models.py --model llm --input examples/wav/BAC009S0764W0121.wav
  
  # 测试视频文件
  python test_models.py --model aed --input examples/video/sample.mp4
  python test_models.py --model both --input examples/video/sample.mp4
  
  # 向后兼容
  python test_models.py --model both --wav examples/wav/BAC009S0764W0121.wav
"""

import argparse
import time
from pathlib import Path

from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.utils.video_audio import is_video_file, is_audio_file

def test_model(model_type, input_path, use_gpu=True):
    """测试指定模型"""
    print(f"\n{'='*50}")
    print(f"测试 {model_type.upper()} 模型")
    print(f"{'='*50}")
    
    # 检查输入文件类型
    if is_video_file(input_path):
        print(f"📹 输入文件类型: 视频文件")
    elif is_audio_file(input_path) or input_path.endswith('.wav'):
        print(f"🎵 输入文件类型: 音频文件")
    else:
        print(f"❓ 输入文件类型: 未知格式，尝试作为音频处理")
    
    print(f"📁 输入文件: {input_path}")
    
    # 模型路径配置
    model_paths = {
        'aed': 'pretrained_models/FireRedASR-AED-L',
        'llm': 'pretrained_models/FireRedASR-LLM-L'
    }
    
    # 解码参数配置
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
        # 加载模型
        print(f"正在加载 {model_type.upper()} 模型...")
        start_time = time.time()
        model = FireRedAsr.from_pretrained(model_type, model_paths[model_type])
        load_time = time.time() - start_time
        print(f"✅ 模型加载成功 (耗时: {load_time:.2f}s)")
        
        # 进行推理
        print(f"正在处理输入文件: {input_path}")
        uttid = Path(input_path).stem
        
        start_time = time.time()
        results = model.transcribe([uttid], [input_path], decode_configs[model_type])
        inference_time = time.time() - start_time
        
        # 输出结果
        if results and len(results) > 0:
            result = results[0]
            print(f"🎯 识别结果: {result['text']}")
            print(f"📊 推理时间: {inference_time:.2f}s")
            print(f"📊 RTF: {result['rtf']}")
        else:
            print("❌ 未获得识别结果")
        
        return True
        
    except Exception as e:
        print(f"❌ {model_type.upper()} 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='FireRedASR 模型测试脚本')
    parser.add_argument('--model', type=str, choices=['aed', 'llm', 'both'], 
                       default='both', help='选择测试模型 (默认: both)')
    parser.add_argument('--input', type=str, 
                       help='测试输入文件路径（音频或视频）')
    parser.add_argument('--wav', type=str, 
                       default='examples/wav/BAC009S0764W0121.wav',
                       help='测试音频文件路径（向后兼容）')
    parser.add_argument('--cpu', action='store_true', 
                       help='使用CPU模式（默认使用GPU）')
    
    args = parser.parse_args()
    
    # 确定输入文件
    input_file = args.input if args.input else args.wav
    
    if not input_file:
        print("❌ 错误: 请指定输入文件 (--input 或 --wav)")
        return
    
    if not Path(input_file).exists():
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return
    
    use_gpu = not args.cpu
    
    print("🔥 FireRedASR 模型测试")
    print(f"输入文件: {input_file}")
    print(f"计算设备: {'GPU' if use_gpu else 'CPU'}")
    
    success_count = 0
    total_count = 0
    
    try:
        if args.model in ['aed', 'both']:
            total_count += 1
            if test_model('aed', input_file, use_gpu):
                success_count += 1
        
        if args.model in ['llm', 'both']:
            total_count += 1
            if test_model('llm', input_file, use_gpu):
                success_count += 1
    finally:
        # 清理临时文件
        print("\n🧹 正在清理临时文件...")
        # 注意：在实际应用中，这里应该访问模型的清理方法
        # 但由于模型可能在不同作用域，我们依赖ASRFeatExtractor的自动清理
    
    # 总结
    print(f"\n{'='*50}")
    print(f"测试完成: {success_count}/{total_count} 个模型测试成功")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()