#!/usr/bin/env python3
"""
音频切片工具

使用 VAD (Voice Activity Detection) 将长音频文件切分为短片段
支持多种音频格式输入，输出为 16kHz 单声道 WAV 格式

使用方法：
    python audio_slicer.py --input_audio long_audio.wav --output_dir segments/
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import time


class AudioSlicer:
    def __init__(self, output_dir="segments", min_speech_duration_ms=1000, 
                 max_speech_duration_s=30, min_silence_duration_ms=500):
        self.output_dir = Path(output_dir)
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.min_silence_duration_ms = min_silence_duration_ms
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def check_dependencies(self):
        """检查依赖是否安装"""
        print("🔍 检查依赖...")
        
        # 检查 ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            print("✅ ffmpeg 已安装")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ ffmpeg 未安装，请先安装 ffmpeg")
            return False
        
        # 检查 torch 和 torchaudio
        try:
            import torch
            import torchaudio
            print("✅ torch 和 torchaudio 已安装")
        except ImportError:
            print("❌ torch 或 torchaudio 未安装，请运行: pip install torch torchaudio")
            return False
        
        # 检查 silero-vad
        try:
            import silero_vad
            print("✅ silero-vad 已安装")
        except ImportError:
            print("❌ silero-vad 未安装，请运行: pip install silero-vad")
            return False
        
        return True
    
    def prepare_audio(self, input_path, target_sample_rate=16000):
        """准备音频：转换为 16kHz 单声道 WAV 格式"""
        print(f"🎵 准备音频: {input_path}")
        
        input_path = Path(input_path)
        output_path = self.output_dir / "prepared_audio.wav"
        
        # 使用 ffmpeg 转换音频
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-ar", str(target_sample_rate),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-f", "wav",
            "-y",  # 覆盖输出文件
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ 音频准备完成: {output_path}")
            
            # 获取音频信息
            info_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
                str(output_path)
            ]
            info_result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
            audio_info = json.loads(info_result.stdout)
            
            duration = float(audio_info['format']['duration'])
            print(f"📊 音频时长: {self.format_time(duration)}")
            
            return output_path, duration
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 音频转换失败: {e.stderr}")
            return None, None
    
    def segment_audio_with_vad(self, audio_path, duration):
        """使用 VAD 切片音频"""
        print("✂️ 使用 VAD 切片音频...")
        print(f"📋 VAD 参数:")
        print(f"  - 最小语音段长度: {self.min_speech_duration_ms}ms")
        print(f"  - 最大语音段长度: {self.max_speech_duration_s}s")
        print(f"  - 最小静音间隔: {self.min_silence_duration_ms}ms")
        
        try:
            # 导入必要的库
            import torch
            import torchaudio
            from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
            
            # 加载 VAD 模型
            print("🔄 加载 VAD 模型...")
            model = load_silero_vad()
            print("✅ VAD 模型加载完成")
            
            # 读取音频
            print("📖 读取音频文件...")
            wav = read_audio(str(audio_path))
            print(f"✅ 音频读取完成，采样点数: {len(wav)}")
            
            # 获取语音时间戳
            print("🔍 检测语音活动...")
            start_time = time.time()
            
            speech_timestamps = get_speech_timestamps(
                wav, model, 
                return_seconds=True,
                min_speech_duration_ms=self.min_speech_duration_ms,
                max_speech_duration_s=self.max_speech_duration_s,
                min_silence_duration_ms=self.min_silence_duration_ms
            )
            
            vad_time = time.time() - start_time
            print(f"✅ VAD 检测完成 (耗时: {vad_time:.2f}s)")
            print(f"📊 检测到 {len(speech_timestamps)} 个语音段")
            
            if not speech_timestamps:
                print("⚠️ 警告: 未检测到任何语音段")
                return []
            
            # 计算覆盖率
            total_speech_duration = sum(seg['end'] - seg['start'] for seg in speech_timestamps)
            coverage = (total_speech_duration / duration) * 100
            print(f"📈 语音覆盖率: {coverage:.1f}% ({self.format_time(total_speech_duration)}/{self.format_time(duration)})")
            
            # 保存分段信息
            segments_info = []
            
            # 切片音频
            print("\n✂️ 开始切片音频...")
            for i, segment in enumerate(speech_timestamps):
                start_time_seg = segment['start']
                end_time_seg = segment['end']
                duration_seg = end_time_seg - start_time_seg
                
                # 生成分段文件名
                segment_filename = f"segment_{i:03d}.wav"
                segment_path = self.output_dir / segment_filename
                
                # 使用 ffmpeg 切片
                cmd = [
                    "ffmpeg", "-i", str(audio_path),
                    "-ss", str(start_time_seg),
                    "-t", str(duration_seg),
                    "-acodec", "copy",
                    "-y",
                    str(segment_path)
                ]
                
                try:
                    subprocess.run(cmd, capture_output=True, check=True)
                    
                    # 记录分段信息
                    segment_info = {
                        "id": i,
                        "filename": segment_filename,
                        "start_time": start_time_seg,
                        "end_time": end_time_seg,
                        "duration": duration_seg,
                        "file_path": str(segment_path)
                    }
                    segments_info.append(segment_info)
                    
                    print(f"  ✅ 分段 {i:03d}: {self.format_time(start_time_seg)} - {self.format_time(end_time_seg)} ({self.format_time(duration_seg)})")
                    
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ 分段 {i:03d} 切片失败: {e}")
                    continue
            
            # 保存分段信息到 JSON 文件
            segments_json_path = self.output_dir / "segments.json"
            with open(segments_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_segments": len(segments_info),
                    "total_duration": duration,
                    "speech_duration": total_speech_duration,
                    "coverage_percent": coverage,
                    "vad_parameters": {
                        "min_speech_duration_ms": self.min_speech_duration_ms,
                        "max_speech_duration_s": self.max_speech_duration_s,
                        "min_silence_duration_ms": self.min_silence_duration_ms
                    },
                    "segments": segments_info
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 音频切片完成!")
            print(f"📊 统计信息:")
            print(f"  - 总分段数: {len(segments_info)}")
            print(f"  - 原始时长: {self.format_time(duration)}")
            print(f"  - 语音时长: {self.format_time(total_speech_duration)}")
            print(f"  - 覆盖率: {coverage:.1f}%")
            print(f"📄 分段信息保存至: {segments_json_path}")
            print(f"📁 分段文件保存至: {self.output_dir}")
            
            return segments_info
            
        except Exception as e:
            print(f"❌ 音频切片失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def format_time(self, seconds):
        """格式化时间为 HH:MM:SS 格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def slice_audio(self, input_audio):
        """完整的音频切片流程"""
        print("✂️ 音频切片工具")
        print("=" * 50)
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 准备音频
        print("\n🔹 准备音频")
        prepared_audio, duration = self.prepare_audio(input_audio)
        if not prepared_audio:
            return False
        
        # 切片音频
        print("\n🔹 VAD 切片")
        segments_info = self.segment_audio_with_vad(prepared_audio, duration)
        if segments_info is None:
            return False
        
        print("\n" + "=" * 50)
        print("✅ 音频切片完成!")
        print(f"📁 输出目录: {self.output_dir}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="音频切片工具 - 使用 VAD 将长音频切分为短片段")
    parser.add_argument('--input_audio', type=str, required=True, help="输入音频/视频文件路径")
    parser.add_argument('--output_dir', type=str, default='segments', help="输出目录")
    parser.add_argument('--min_speech_duration_ms', type=int, default=1000, help="最小语音段长度(毫秒)")
    parser.add_argument('--max_speech_duration_s', type=int, default=30, help="最大语音段长度(秒)")
    parser.add_argument('--min_silence_duration_ms', type=int, default=500, help="最小静音间隔(毫秒)")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input_audio).exists():
        print(f"❌ 错误: 输入文件不存在: {args.input_audio}")
        return
    
    # 创建切片器
    slicer = AudioSlicer(
        output_dir=args.output_dir,
        min_speech_duration_ms=args.min_speech_duration_ms,
        max_speech_duration_s=args.max_speech_duration_s,
        min_silence_duration_ms=args.min_silence_duration_ms
    )
    
    # 执行切片
    success = slicer.slice_audio(args.input_audio)
    
    if success:
        print("\n🎉 切片成功完成!")
    else:
        print("\n❌ 切片失败")
        sys.exit(1)


if __name__ == "__main__":
    main()