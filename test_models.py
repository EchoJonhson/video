#!/usr/bin/env python3
"""
FireRedASR 模型快速测试脚本
用法：
  python test_models.py --model aed --wav examples/wav/BAC009S0764W0121.wav
  python test_models.py --model llm --wav examples/wav/BAC009S0764W0121.wav
  python test_models.py --model both --wav examples/wav/BAC009S0764W0121.wav
"""

import argparse
import time
from fireredasr.models.fireredasr import FireRedAsr

def test_model(model_type, wav_path, use_gpu=True):
    """测试指定模型"""
    print(f"\n{'='*50}")
    print(f"测试 {model_type.upper()} 模型")
    print(f"{'='*50}")
    
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
        print(f"正在处理音频文件: {wav_path}")
        uttid = wav_path.split('/')[-1].replace('.wav', '')
        
        start_time = time.time()
        results = model.transcribe([uttid], [wav_path], decode_configs[model_type])
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
    parser.add_argument('--wav', type=str, 
                       default='examples/wav/BAC009S0764W0121.wav',
                       help='测试音频文件路径')
    parser.add_argument('--cpu', action='store_true', 
                       help='使用CPU模式（默认使用GPU）')
    
    args = parser.parse_args()
    
    use_gpu = not args.cpu
    
    print("🔥 FireRedASR 模型测试")
    print(f"音频文件: {args.wav}")
    print(f"计算设备: {'GPU' if use_gpu else 'CPU'}")
    
    success_count = 0
    total_count = 0
    
    if args.model in ['aed', 'both']:
        total_count += 1
        if test_model('aed', args.wav, use_gpu):
            success_count += 1
    
    if args.model in ['llm', 'both']:
        total_count += 1
        if test_model('llm', args.wav, use_gpu):
            success_count += 1
    
    # 总结
    print(f"\n{'='*50}")
    print(f"测试完成: {success_count}/{total_count} 个模型测试成功")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()