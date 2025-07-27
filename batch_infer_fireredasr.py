#!/usr/bin/env python3
"""
批量 FireRedASR 转写脚本

批量处理音频分段文件，使用 FireRedASR 模型进行语音识别
支持 AED 和 LLM 两种模型类型

使用方法：
    python batch_infer_fireredasr.py --input_dir segments/ --model_type aed --model_dir pretrained_models/FireRedASR-AED-L --output_dir transcripts/
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from fireredasr.models.fireredasr import FireRedAsr


class BatchFireRedASRInference:
    def __init__(self, model_type="aed", model_dir=None, output_dir="transcripts"):
        self.model_type = model_type.lower()
        self.model_dir = model_dir
        self.output_dir = Path(output_dir)
        self.model = None
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的音频格式
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    
    def load_model(self):
        """加载 FireRedASR 模型"""
        if self.model is None:
            print(f"🔄 加载 FireRedASR-{self.model_type.upper()} 模型...")
            print(f"📁 模型路径: {self.model_dir}")
            
            start_time = time.time()
            
            try:
                self.model = FireRedAsr.from_pretrained(self.model_type, self.model_dir)
                load_time = time.time() - start_time
                print(f"✅ 模型加载成功 (耗时: {load_time:.2f}s)")
                return True
            except Exception as e:
                print(f"❌ 模型加载失败: {str(e)}")
                return False
        return True
    
    def get_decode_config(self):
        """获取解码配置"""
        if self.model_type == "aed":
            return {
                "use_gpu": 1,
                "beam_size": 3,
                "nbest": 1,
                "decode_max_len": 0,
                "softmax_smoothing": 1.25,
                "aed_length_penalty": 0.6,
                "eos_penalty": 1.0
            }
        else:  # llm
            return {
                "use_gpu": 1,
                "beam_size": 3,
                "decode_max_len": 0,
                "decode_min_len": 0,
                "repetition_penalty": 3.0,
                "llm_length_penalty": 1.0,
                "temperature": 1.0
            }
    
    def scan_audio_files(self, input_dir):
        """扫描输入目录中的音频文件"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"❌ 错误: 输入目录不存在: {input_dir}")
            return []
        
        audio_files = []
        
        # 扫描音频文件
        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                audio_files.append(file_path)
        
        # 按文件名排序
        audio_files.sort()
        
        print(f"📂 扫描目录: {input_dir}")
        print(f"🔍 找到 {len(audio_files)} 个音频文件")
        
        if audio_files:
            print("📋 文件列表:")
            for i, file_path in enumerate(audio_files[:10]):  # 只显示前10个
                print(f"  {i+1:3d}. {file_path.name}")
            if len(audio_files) > 10:
                print(f"  ... 还有 {len(audio_files) - 10} 个文件")
        
        return audio_files
    
    def load_segments_info(self, input_dir):
        """加载分段信息（如果存在）"""
        segments_json_path = Path(input_dir) / "segments.json"
        
        if segments_json_path.exists():
            try:
                with open(segments_json_path, 'r', encoding='utf-8') as f:
                    segments_data = json.load(f)
                print(f"📄 加载分段信息: {segments_json_path}")
                return segments_data.get('segments', [])
            except Exception as e:
                print(f"⚠️ 警告: 无法加载分段信息: {e}")
        
        return None
    
    def transcribe_single_file(self, file_path, uttid=None):
        """转写单个音频文件"""
        if uttid is None:
            uttid = file_path.stem
        
        try:
            start_time = time.time()
            
            # 调用 FireRedASR 转写
            results = self.model.transcribe(
                [uttid], [str(file_path)], self.get_decode_config()
            )
            
            process_time = time.time() - start_time
            
            if results and len(results) > 0:
                result = results[0]
                text = result['text']
                rtf = float(result.get('rtf', 0))
                
                return {
                    "success": True,
                    "text": text,
                    "process_time": process_time,
                    "rtf": rtf,
                    "uttid": uttid
                }
            else:
                return {
                    "success": False,
                    "error": "没有返回结果",
                    "process_time": process_time,
                    "uttid": uttid
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "process_time": 0,
                "uttid": uttid
            }
    
    def batch_transcribe(self, input_dir, segments_info=None):
        """批量转写音频文件"""
        print(f"🎯 开始批量转写")
        print(f"📂 输入目录: {input_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"🤖 使用模型: FireRedASR-{self.model_type.upper()}")
        
        # 加载模型
        if not self.load_model():
            return None
        
        # 扫描音频文件
        audio_files = self.scan_audio_files(input_dir)
        if not audio_files:
            print("❌ 没有找到音频文件")
            return None
        
        # 加载分段信息
        if segments_info is None:
            segments_info = self.load_segments_info(input_dir)
        
        # 创建文件名到分段信息的映射
        segment_map = {}
        if segments_info:
            for segment in segments_info:
                segment_map[segment['filename']] = segment
        
        # 批量转写
        transcription_results = []
        successful_count = 0
        total_process_time = 0
        total_rtf = 0
        
        print(f"\n🔄 开始转写 {len(audio_files)} 个文件...")
        print("=" * 80)
        
        for i, file_path in enumerate(audio_files):
            print(f"\n[{i+1}/{len(audio_files)}] 转写: {file_path.name}")
            
            # 获取分段信息
            segment_info = segment_map.get(file_path.name, {})
            if segment_info:
                start_time = segment_info.get('start_time', 0)
                end_time = segment_info.get('end_time', 0)
                duration = segment_info.get('duration', 0)
                print(f"  📊 时间: {self.format_time(start_time)} - {self.format_time(end_time)} ({self.format_time(duration)})")
            
            # 转写文件
            result = self.transcribe_single_file(file_path)
            
            if result['success']:
                text = result['text']
                process_time = result['process_time']
                rtf = result['rtf']
                
                print(f"  ✅ 转写成功 (耗时: {process_time:.2f}s, RTF: {rtf:.4f})")
                print(f"  📝 结果: {text}")
                
                # 保存单个转写结果
                output_filename = file_path.stem + ".txt"
                output_path = self.output_dir / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # 记录结果
                transcription_result = {
                    "id": i,
                    "filename": file_path.name,
                    "output_file": output_filename,
                    "text": text,
                    "process_time": process_time,
                    "rtf": rtf,
                    "success": True
                }
                
                # 添加分段信息
                if segment_info:
                    transcription_result.update({
                        "start_time": segment_info.get('start_time', 0),
                        "end_time": segment_info.get('end_time', 0),
                        "duration": segment_info.get('duration', 0)
                    })
                
                transcription_results.append(transcription_result)
                successful_count += 1
                total_process_time += process_time
                total_rtf += rtf
                
            else:
                error = result['error']
                print(f"  ❌ 转写失败: {error}")
                
                transcription_result = {
                    "id": i,
                    "filename": file_path.name,
                    "error": error,
                    "success": False
                }
                
                if segment_info:
                    transcription_result.update({
                        "start_time": segment_info.get('start_time', 0),
                        "end_time": segment_info.get('end_time', 0),
                        "duration": segment_info.get('duration', 0)
                    })
                
                transcription_results.append(transcription_result)
        
        # 保存批量转写结果
        results_json_path = self.output_dir / "batch_transcription_results.json"
        
        batch_results = {
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "model_dir": str(self.model_dir),
            "input_dir": str(input_dir),
            "output_dir": str(self.output_dir),
            "total_files": len(audio_files),
            "successful_files": successful_count,
            "failed_files": len(audio_files) - successful_count,
            "total_process_time": total_process_time,
            "average_rtf": total_rtf / successful_count if successful_count > 0 else 0,
            "results": transcription_results
        }
        
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        # 打印统计信息
        print("\n" + "=" * 80)
        print("📊 批量转写完成!")
        print(f"  总文件数: {len(audio_files)}")
        print(f"  成功转写: {successful_count}")
        print(f"  失败转写: {len(audio_files) - successful_count}")
        print(f"  成功率: {(successful_count / len(audio_files) * 100):.1f}%")
        if successful_count > 0:
            print(f"  总处理时间: {total_process_time:.2f}s")
            print(f"  平均 RTF: {(total_rtf / successful_count):.4f}")
        print(f"📄 详细结果: {results_json_path}")
        print(f"📁 转写文件: {self.output_dir}")
        
        # 清理临时文件
        if self.model:
            self.model.feat_extractor.cleanup_temp_files()
        
        return transcription_results
    
    def format_time(self, seconds):
        """格式化时间为 HH:MM:SS 格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    parser = argparse.ArgumentParser(description="批量 FireRedASR 转写脚本")
    parser.add_argument('--input_dir', type=str, required=True, help="输入音频文件目录")
    parser.add_argument('--model_type', type=str, choices=['aed', 'llm'], default='aed', help="模型类型")
    parser.add_argument('--model_dir', type=str, required=True, help="模型目录路径")
    parser.add_argument('--output_dir', type=str, default='transcripts', help="输出目录")
    parser.add_argument('--segments_json', type=str, help="分段信息 JSON 文件路径（可选）")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not Path(args.input_dir).exists():
        print(f"❌ 错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 检查模型目录
    if not Path(args.model_dir).exists():
        print(f"❌ 错误: 模型目录不存在: {args.model_dir}")
        return
    
    # 加载分段信息（如果提供）
    segments_info = None
    if args.segments_json:
        if Path(args.segments_json).exists():
            try:
                with open(args.segments_json, 'r', encoding='utf-8') as f:
                    segments_data = json.load(f)
                segments_info = segments_data.get('segments', [])
                print(f"📄 加载分段信息: {args.segments_json}")
            except Exception as e:
                print(f"⚠️ 警告: 无法加载分段信息: {e}")
        else:
            print(f"⚠️ 警告: 分段信息文件不存在: {args.segments_json}")
    
    # 创建批量转写器
    transcriber = BatchFireRedASRInference(
        model_type=args.model_type,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # 执行批量转写
    results = transcriber.batch_transcribe(args.input_dir, segments_info)
    
    if results is not None:
        print("\n🎉 批量转写成功完成!")
    else:
        print("\n❌ 批量转写失败")
        sys.exit(1)


if __name__ == "__main__":
    main()