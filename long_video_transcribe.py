#!/usr/bin/env python3
"""
FireRedASR 长视频转文字批量处理系统

功能：
- 自动扫描 Use/Input/ 中的长视频文件
- 智能切片处理（使用 Silero VAD）
- 批量使用 FireRedASR 模型转写
- 拼接成完整文字稿和字幕文件
- 结果保存到 Use/Output/ 文件夹

使用方法：
    python long_video_transcribe.py
    python long_video_transcribe.py --model_type llm
    python long_video_transcribe.py --max_duration 45 --min_silence 300
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import time
import torch
import torchaudio
import requests
import zipfile

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from fireredasr.models.fireredasr import FireRedAsr
from utils.hardware_manager import get_hardware_manager
from utils.smart_model_loader import create_smart_loader
from utils.parallel_processor import AudioBatchProcessor
from fireredasr.utils.video_audio import is_video_file, is_audio_file
from fireredasr.utils.punctuation_restore import PunctuationRestorer
from fireredasr.utils.paragraph_segmentation import ParagraphSegmenter


class LongVideoTranscriber:
    def __init__(self):
        self.input_dir = Path("Use/Input")
        self.output_dir = Path("Use/Output")
        self.temp_dir = Path("Use/Output/temp_long_video")
        
        # VAD 参数
        self.min_speech_duration_ms = 1000
        self.max_speech_duration_s = 30
        self.min_silence_duration_ms = 500
        
        # 模型相关
        self.model_type = None
        self.model = None
        
        # 支持的格式
        self.supported_video = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        self.supported_audio = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        
        # 初始化智能系统
        print("🔧 初始化智能处理系统...")
        self.hardware_manager = get_hardware_manager()
        self.smart_loader = create_smart_loader(self.hardware_manager)
        self.parallel_processor = None
        
        # 检查环境变量配置
        force_cpu = os.environ.get('FIREREDASR_FORCE_CPU', '').lower() in ['1', 'true', 'yes']
        if force_cpu:
            print("⚠️ 强制使用 CPU 模式 (FIREREDASR_FORCE_CPU=1)")
            self.hardware_manager.strategy['name'] = 'cpu_primary'
            self.hardware_manager.strategy['use_gpu'] = False
        
        # 打印硬件配置
        self.hardware_manager.print_hardware_info()
        
        # 标点恢复相关
        self.enable_punctuation = True  # 默认启用标点恢复
        self.punctuation_restorer = None
        self.punctuation_model_dir = None
        self.punctuation_chunk_size = 256
        self.punctuation_stride = 128
        
        # 分段相关
        self.enable_paragraph = False  # 默认不启用分段
        self.paragraph_segmenter = None
        self.paragraph_method = "rule"  # rule/semantic/hybrid
        self.min_paragraph_length = 50
        self.max_paragraph_length = 500
        
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
        
        # 检查 torchaudio
        try:
            import torchaudio
            print("✅ torchaudio 已安装")
        except ImportError:
            print("❌ torchaudio 未安装，请运行: pip install torchaudio")
            return False
        
        return True
    
    def scan_long_media_files(self):
        """扫描输入文件夹中的长媒体文件"""
        if not self.input_dir.exists():
            print(f"❌ 错误: 输入文件夹不存在: {self.input_dir}")
            return []
        
        media_files = []
        all_extensions = self.supported_audio | self.supported_video
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in all_extensions:
                # 获取文件时长（粗略估计）
                file_size_mb = file_path.stat().st_size / 1024 / 1024
                # 为了测试，暂时降低文件大小限制
                if file_size_mb > 0.1:  # 大于0.1MB的文件（测试用）
                    media_files.append(file_path)
        
        return sorted(media_files)
    
    def display_files(self, files):
        """显示找到的文件"""
        if not files:
            print("❌ 在 Use/Input/ 文件夹中没有找到大型媒体文件")
            print("提示：长视频处理适用于大于10MB的音视频文件")
            return False
        
        print(f"\n📁 在 Use/Input/ 中找到 {len(files)} 个大型媒体文件:")
        print("-" * 60)
        
        for i, file_path in enumerate(files, 1):
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            if is_video_file(str(file_path)):
                file_type = "📹 视频"
            else:
                file_type = "🎵 音频"
            
            print(f"{i:2d}. {file_type} | {file_path.name} ({file_size:.2f} MB)")
        
        print("-" * 60)
        return True
    
    def get_model_dir(self):
        """根据命令行参数获取模型目录"""
        if not self.model_type:
            print("❌ 未指定模型类型，请使用 --model_type 参数")
            return None
        
        if self.model_type == "aed":
            model_dir = "pretrained_models/FireRedASR-AED-L"
            print("✅ 使用 FireRedASR-AED 模型 (快速, 适合长音频)")
        elif self.model_type == "llm":
            model_dir = "pretrained_models/FireRedASR-LLM-L"
            print("✅ 使用 FireRedASR-LLM 模型 (高精度, 处理较慢)")
        else:
            print(f"❌ 未知模型类型: {self.model_type}")
            return None
        
        # 检查模型路径
        if not Path(model_dir).exists():
            print(f"❌ 模型目录不存在: {model_dir}")
            print("请先下载模型文件，参考 step.md 文档")
            return None
        
        return model_dir
    
    def select_model(self):
        """让用户选择模型"""
        print("\n🤖 请选择要使用的模型:")
        print("1. FireRedASR-AED (快速, 适合长音频)")
        print("2. FireRedASR-LLM (高精度, 处理较慢)")
        
        while True:
            try:
                choice = input("\n请输入选择 (1 或 2): ").strip()
                if choice == "1":
                    self.model_type = "aed"
                    model_dir = "pretrained_models/FireRedASR-AED-L"
                    print("✅ 选择了 FireRedASR-AED 模型")
                    break
                elif choice == "2":
                    self.model_type = "llm"
                    model_dir = "pretrained_models/FireRedASR-LLM-L"
                    print("✅ 选择了 FireRedASR-LLM 模型")
                    break
                else:
                    print("❌ 无效输入，请输入 1 或 2")
            except KeyboardInterrupt:
                print("\n\n👋 用户取消操作")
                return None
        
        # 检查模型路径
        if not Path(model_dir).exists():
            print(f"❌ 错误: 模型目录不存在: {model_dir}")
            print("请从 https://huggingface.co/fireredteam 下载模型文件")
            return None
        
        return model_dir
    
    def prepare_audio(self, input_path, output_path):
        """准备音频：转换为 16kHz 单声道 WAV 格式"""
        print(f"🎵 准备音频: {input_path.name}")
        
        # 使用 ffmpeg 转换音频
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-f", "wav",
            "-y",  # 覆盖输出文件
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ 音频准备完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 音频转换失败: {e.stderr}")
            return False
    
    def load_silero_vad(self):
        """加载 Silero VAD 模型"""
        print("🔄 加载 VAD 模型...")
        
        # 方法1: 使用 pip 安装的 silero-vad 包（推荐）
        try:
            print("📦 尝试使用 silero-vad 包...")
            from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
            
            model = load_silero_vad()
            
            # 创建兼容的 save_audio 函数
            def save_audio(path, tensor, sampling_rate):
                torchaudio.save(path, tensor, sampling_rate)
            
            print("✅ VAD 模型加载成功 (silero-vad 包)")
            return model, get_speech_timestamps, read_audio, save_audio
            
        except ImportError as e:
            print(f"❌ silero-vad 包未安装: {e}")
        except Exception as e:
            print(f"❌ silero-vad 包加载失败: {str(e)}")
        
        # 方法2: 尝试从 torch.hub 加载
        for attempt in range(2):
            try:
                print(f"📁 尝试从 torch.hub 加载 (尝试 {attempt + 1}/2)...")
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=attempt > 0,
                    trust_repo=True
                )
                (get_speech_timestamps, save_audio, read_audio, 
                 VADIterator, collect_chunks) = utils
                
                print("✅ VAD 模型加载成功 (torch.hub)")
                return model, get_speech_timestamps, read_audio, save_audio
                
            except Exception as e:
                print(f"❌ torch.hub 加载失败 (尝试 {attempt + 1}/2): {str(e)}")
                if attempt == 0:
                    time.sleep(3)
        
        # 如果所有方法都失败
        raise Exception("❌ VAD模型加载失败！请确保已安装 silero-vad: pip install silero-vad")
    
    def slice_audio_with_vad(self, audio_path, output_dir):
        """使用 VAD 切分音频"""
        print(f"✂️ 开始切分音频...")
        
        # 加载 VAD 模型
        vad_model, get_speech_timestamps, read_audio, save_audio = self.load_silero_vad()
        
        # 读取音频
        wav = read_audio(str(audio_path))
        
        # 获取语音时间戳  
        speech_timestamps = get_speech_timestamps(
            wav, 
            vad_model,
            threshold=0.5,
            sampling_rate=16000,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms
        )
        
        # 转换为秒（silero-vad默认返回采样点，需要转换）
        for ts in speech_timestamps:
            ts['start'] = ts['start'] / 16000.0
            ts['end'] = ts['end'] / 16000.0
        
        if not speech_timestamps:
            print("❌ 没有检测到语音段")
            return []
        
        print(f"✅ 检测到 {len(speech_timestamps)} 个初始语音段")
        
        # 合并和切分语音段
        segments = []
        current_segment = None
        
        for timestamp in speech_timestamps:
            if current_segment is None:
                current_segment = {
                    'start': timestamp['start'],
                    'end': timestamp['end']
                }
            elif (timestamp['start'] - current_segment['end'] < 1.0 and 
                  timestamp['end'] - current_segment['start'] < self.max_speech_duration_s):
                # 合并相近的段
                current_segment['end'] = timestamp['end']
            else:
                # 保存当前段，开始新段
                segments.append(current_segment)
                current_segment = {
                    'start': timestamp['start'],
                    'end': timestamp['end']
                }
        
        if current_segment:
            segments.append(current_segment)
        
        print(f"✅ 合并后得到 {len(segments)} 个语音段")
        
        # 保存音频段
        segment_files = []
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        for i, segment in enumerate(segments):
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            
            segment_waveform = waveform[:, start_sample:end_sample]
            segment_path = output_dir / f"segment_{i:03d}.wav"
            
            torchaudio.save(str(segment_path), segment_waveform, sample_rate)
            
            segment_files.append({
                'index': i,
                'file': segment_path.name,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['end'] - segment['start']
            })
        
        # 保存分段信息
        segments_info_path = output_dir / "segments.json"
        with open(segments_info_path, 'w', encoding='utf-8') as f:
            json.dump(segment_files, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 音频切分完成，共 {len(segment_files)} 个片段")
        return segment_files
    
    def batch_transcribe(self, segments_dir, model_dir):
        """智能批量转写音频片段"""
        print("\n🎤 开始智能批量转写...")
        
        # 验证分段目录和文件
        if not segments_dir.exists():
            print(f"❌ 分段目录不存在: {segments_dir}")
            return None
        
        segments_info_path = segments_dir / "segments.json"
        if not segments_info_path.exists():
            print(f"❌ 分段信息文件不存在: {segments_info_path}")
            return None
        
        # 读取分段信息
        try:
            with open(segments_info_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            print(f"📋 加载分段信息: {len(segments)} 个片段")
        except Exception as e:
            print(f"❌ 读取分段信息失败: {e}")
            return None
        
        # 验证分段文件是否存在
        valid_segments = []
        missing_files = []
        
        for segment in segments:
            segment_path = segments_dir / segment['file']
            if segment_path.exists():
                valid_segments.append(segment)
            else:
                missing_files.append(segment['file'])
        
        if missing_files:
            print(f"⚠️ 警告: {len(missing_files)} 个分段文件不存在")
            if len(missing_files) <= 5:
                for f in missing_files:
                    print(f"  - {f}")
            else:
                for f in missing_files[:3]:
                    print(f"  - {f}")
                print(f"  ... 还有 {len(missing_files) - 3} 个文件")
        
        if not valid_segments:
            print("❌ 没有有效的分段文件")
            return None
        
        print(f"✅ 找到 {len(valid_segments)} 个有效分段文件")
        segments = valid_segments
        
        # 使用智能模型加载器
        self.model = self.smart_loader.load_model(self.model_type, model_dir)
        if not self.model:
            print("❌ 模型加载失败")
            return None
        
        # 优化模型以进行推理
        self.smart_loader.optimize_for_inference()
        
        # 获取智能解码配置
        decode_config = self.smart_loader.get_transcribe_config()
        print(f"🎯 解码配置: {decode_config}")
        
        # 获取并行处理配置
        strategy = self.hardware_manager.get_optimal_config()['strategy']
        
        # 智能选择并行处理策略
        segment_count = len(segments)
        
        if self.model_type == "llm":
            # LLM 模型智能处理策略
            # 检查是否使用了GPU辅助（编码器在GPU）
            gpu_assisted = self.smart_loader.strategy.get('gpu_role') in ['encoder_only', 'feature_extraction']
            
            if gpu_assisted and segment_count > 10:
                # GPU辅助模式下，可以使用有限的并行
                max_workers = min(2, max(1, strategy['cpu_threads'] // 8))  # 保守的并行度
                batch_size = 1
                print(f"🚀 LLM GPU辅助模式: {segment_count} 个分段，使用 {max_workers} 线程并行")
                print("📌 提示: 编码器在GPU上，LLM主体在CPU上，采用保守并行策略")
            else:
                # 纯CPU模式或少量分段，使用串行处理
                max_workers = 1
                batch_size = 1
                if segment_count <= 10:
                    print(f"⚠️ LLM 串行处理: 分段数较少({segment_count}个)，使用串行处理")
                else:
                    print("⚠️ LLM 纯CPU模式，使用串行处理以确保稳定性")
        else:
            # AED 模型可以安全地并行处理
            # 根据分段数量和硬件能力智能调整并行度
            if segment_count <= 10:
                max_workers = min(2, strategy['cpu_threads'])  # 少量分段用少线程
            elif segment_count <= 50:
                max_workers = min(4, strategy['cpu_threads'])  # 中等数量分段
            else:
                max_workers = min(8, strategy['cpu_threads'])  # 大量分段用更多线程
            
            batch_size = min(strategy.get('batch_size', 2), 2)  # 限制批次大小避免内存问题
            print(f"🔧 AED 智能并行: {segment_count} 个分段，使用 {max_workers} 线程")
        
        print(f"🔧 处理配置: {max_workers} 线程, 批次大小: {batch_size}")
        
        # 准备音频片段路径
        segment_paths = [segments_dir / segment['file'] for segment in segments]
        
        # 创建输出目录
        transcripts_dir = segments_dir.parent / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建线程锁以保护模型访问
        import threading
        model_lock = threading.Lock()
        
        # 创建转录函数
        def transcribe_single_segment(segment_path):
            """转录单个音频片段"""
            try:
                # 检查文件是否存在
                if not segment_path.exists():
                    print(f"⚠️ 跳过不存在的文件: {segment_path.name}")
                    return None
                
                # 找到对应的 segment 信息
                segment_info = None
                for seg in segments:
                    if segments_dir / seg['file'] == segment_path:
                        segment_info = seg
                        break
                
                if not segment_info:
                    print(f"⚠️ 找不到分段信息: {segment_path.name}")
                    return None
                
                uttid = f"segment_{segment_info['index']:03d}"
                
                # 使用锁保护模型调用
                start_time = time.time()
                with model_lock:
                    # 清理缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 调用模型
                    result = self.model.transcribe([uttid], [str(segment_path)], decode_config)
                
                process_time = time.time() - start_time
                
                if result and len(result) > 0:
                    text = result[0]['text']
                    rtf = float(result[0].get('rtf', 0))
                    
                    # 保存单个结果
                    transcript_path = transcripts_dir / f"{uttid}.txt"
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    return {
                        'index': segment_info['index'],
                        'file': segment_info['file'],
                        'start': segment_info['start'],
                        'end': segment_info['end'],
                        'duration': segment_info['duration'],
                        'text': text,
                        'rtf': rtf,
                        'process_time': process_time
                    }
                else:
                    print(f"⚠️ 模型转录无结果: {segment_path.name}")
                    return None
                
            except Exception as e:
                print(f"❌ 转录片段失败 {segment_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        
        # 根据模型类型选择处理方式
        if max_workers == 1:
            # 串行处理
            print(f"🚀 串行转写 {len(segment_paths)} 个片段...")
            results = []
            for i, segment_path in enumerate(segment_paths):
                print(f"处理片段 {i+1}/{len(segment_paths)}: {segment_path.name}")
                result = transcribe_single_segment(segment_path)
                if result:
                    results.append(result)
                # 定期清理内存
                if (i + 1) % 10 == 0:
                    import gc
                    gc.collect()
        else:
            # 并行处理
            print(f"🚀 使用 {max_workers} 线程并行转写 {len(segment_paths)} 个片段...")
            processor = AudioBatchProcessor(max_workers=max_workers)
            results = processor.process_audio_segments(
                segment_paths, 
                transcribe_single_segment,
                batch_size=batch_size,
                model_type=self.model_type
            )
        
        # 保存转写结果汇总
        if results:
            try:
                results_path = segments_dir.parent / "transcripts.json"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                total = len(segments)
                print(f"\n✅ 智能批量转写完成: {len(results)}/{total} 成功")
                return results
            except Exception as e:
                print(f"❌ 保存转写结果失败: {e}")
                return results  # 返回结果但记录保存失败
        else:
            print("\n❌ 没有成功转写的片段")
            return None
    
    def generate_unique_filename(self, base_path, extension):
        """生成唯一的文件名，避免覆盖"""
        counter = 1
        final_path = Path(f"{base_path}{extension}")
        
        while final_path.exists():
            counter += 1
            final_path = Path(f"{base_path}_{counter}{extension}")
        
        return final_path
    
    def concatenate_results(self, results, input_filename):
        """拼接转写结果"""
        print("\n📝 拼接转写结果...")
        
        # 按时间排序
        results.sort(key=lambda x: x['start'])
        
        # 获取输入文件的基本名称（不含扩展名）
        base_name = Path(input_filename).stem
        output_base = self.output_dir / base_name
        
        # 生成纯文本
        full_text = []
        for result in results:
            full_text.append(result['text'])
        
        txt_path = self.generate_unique_filename(output_base, ".txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_text))
        print(f"✅ 生成纯文本: {txt_path.name}")
        
        # 生成带时间戳的文本
        timestamp_text = []
        for result in results:
            start_time = str(timedelta(seconds=result['start'])).split('.')[0]
            end_time = str(timedelta(seconds=result['end'])).split('.')[0]
            timestamp_text.append(f"[{start_time} --> {end_time}]")
            timestamp_text.append(result['text'])
            timestamp_text.append("")
        
        timestamp_path = self.generate_unique_filename(output_base, "_时间戳.txt")
        with open(timestamp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(timestamp_text))
        print(f"✅ 生成时间戳文本: {timestamp_path.name}")
        
        # 生成 SRT 字幕
        srt_lines = []
        for i, result in enumerate(results, 1):
            start_time = self.seconds_to_srt_time(result['start'])
            end_time = self.seconds_to_srt_time(result['end'])
            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(result['text'])
            srt_lines.append("")
        
        srt_path = self.generate_unique_filename(output_base, ".srt")
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_lines))
        print(f"✅ 生成字幕文件: {srt_path.name}")
        
        # 生成统计信息
        total_duration = results[-1]['end'] if results else 0
        total_process_time = sum(r.get('process_time', 0) for r in results)
        avg_rtf = sum(r.get('rtf', 0) for r in results) / len(results) if results else 0
        
        stats = {
            'total_segments': len(results),
            'total_duration': total_duration,
            'total_duration_formatted': str(timedelta(seconds=total_duration)),
            'total_process_time': total_process_time,
            'average_rtf': avg_rtf,
            'total_characters': sum(len(r['text']) for r in results),
            'model_type': self.model_type
        }
        
        stats_path = self.generate_unique_filename(output_base, "_统计.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 统计信息:")
        print(f"   总时长: {stats['total_duration_formatted']}")
        print(f"   处理时间: {total_process_time:.2f}s")
        print(f"   平均 RTF: {avg_rtf:.4f}")
        print(f"   总字符数: {stats['total_characters']}")
        
        # 标点恢复处理
        if self.enable_punctuation:
            try:
                print(f"\n🔤 开始标点恢复处理...")
                
                # 初始化标点恢复器（延迟加载）
                if self.punctuation_restorer is None:
                    self.punctuation_restorer = PunctuationRestorer(
                        cache_dir=self.punctuation_model_dir,
                        chunk_size=self.punctuation_chunk_size,
                        stride=self.punctuation_stride
                    )
                    
                # 对纯文本进行标点恢复
                full_text_content = '\n'.join(full_text)
                punctuated_text = self.punctuation_restorer.restore_punctuation(full_text_content)
                
                # 保存带标点的纯文本
                punctuated_txt_path = self.generate_unique_filename(output_base, "_标点.txt")
                with open(punctuated_txt_path, 'w', encoding='utf-8') as f:
                    f.write(punctuated_text)
                print(f"✅ 生成带标点文本: {punctuated_txt_path.name}")
                
                # 生成带标点的 SRT 字幕
                # 将带标点的文本按原始分段重新分配
                punctuated_lines = punctuated_text.split('\n')
                if len(punctuated_lines) == len(results):
                    # 如果行数匹配，直接使用
                    for i, result in enumerate(results):
                        if i < len(punctuated_lines):
                            result['punctuated_text'] = punctuated_lines[i]
                        else:
                            result['punctuated_text'] = result['text']
                else:
                    # 如果行数不匹配，尝试按字符长度分配
                    punctuated_full = punctuated_text.replace('\n', ' ')
                    char_offset = 0
                    for result in results:
                        orig_len = len(result['text'])
                        result['punctuated_text'] = punctuated_full[char_offset:char_offset + orig_len].strip()
                        char_offset += orig_len
                
                # 生成带标点的 SRT
                punctuated_srt_lines = []
                for i, result in enumerate(results, 1):
                    start_time = self.seconds_to_srt_time(result['start'])
                    end_time = self.seconds_to_srt_time(result['end'])
                    punctuated_srt_lines.append(str(i))
                    punctuated_srt_lines.append(f"{start_time} --> {end_time}")
                    punctuated_srt_lines.append(result.get('punctuated_text', result['text']))
                    punctuated_srt_lines.append("")
                
                punctuated_srt_path = self.generate_unique_filename(output_base, "_标点.srt")
                with open(punctuated_srt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(punctuated_srt_lines))
                print(f"✅ 生成带标点字幕: {punctuated_srt_path.name}")
                
                # 如果启用了分段功能
                if self.enable_paragraph and punctuated_text:
                    try:
                        print(f"\n📑 开始自然段分段处理...")
                        
                        # 初始化分段器
                        if self.paragraph_segmenter is None:
                            self.paragraph_segmenter = ParagraphSegmenter(
                                min_length=self.min_paragraph_length,
                                max_length=self.max_paragraph_length
                            )
                        
                        # 执行分段
                        paragraphs = self.paragraph_segmenter.segment_paragraphs(punctuated_text)
                        
                        # 保存分段结果
                        paragraph_txt_path = self.generate_unique_filename(output_base, "_段落.txt")
                        with open(paragraph_txt_path, 'w', encoding='utf-8') as f:
                            f.write(f"FireRedASR 视频转写结果（自然段格式）\n")
                            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"段落数: {len(paragraphs)}\n")
                            f.write("=" * 60 + "\n\n")
                            
                            for i, para in enumerate(paragraphs, 1):
                                f.write(f"【第{i}段】\n{para}\n\n")
                        
                        print(f"✅ 生成自然段文件: {paragraph_txt_path.name}")
                        print(f"   共分为 {len(paragraphs)} 个自然段")
                        
                    except Exception as e:
                        print(f"⚠️ 分段处理失败: {str(e)}")
                        print("   将保留带标点版本")
                
            except Exception as e:
                print(f"⚠️ 标点恢复失败: {str(e)}")
                print("   将保留无标点版本")
    
    def seconds_to_srt_time(self, seconds):
        """将秒数转换为 SRT 时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def process_long_video(self, input_path):
        """处理单个长视频文件的完整流程"""
        print(f"\n{'='*60}")
        print(f"🎬 处理文件: {input_path.name}")
        print(f"{'='*60}")
        
        # 创建临时工作目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = self.temp_dir / f"{input_path.stem}_{timestamp}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        segments_dir = work_dir / "segments"
        segments_dir.mkdir(exist_ok=True)
        
        try:
            # 步骤1：准备音频
            print("\n[步骤 1/4] 准备音频...")
            prepared_audio = work_dir / "prepared_audio.wav"
            if not self.prepare_audio(input_path, prepared_audio):
                return False
            
            # 步骤2：VAD 切片
            print("\n[步骤 2/4] VAD 语音检测和切片...")
            segments = self.slice_audio_with_vad(prepared_audio, segments_dir)
            if not segments:
                return False
            
            # 步骤3：批量转写
            print("\n[步骤 3/4] 批量转写...")
            model_dir = self.get_model_dir()
            if not model_dir:
                return False
            
            results = self.batch_transcribe(segments_dir, model_dir)
            if not results:
                return False
            
            # 步骤4：拼接结果
            print("\n[步骤 4/4] 拼接结果...")
            self.concatenate_results(results, input_path.name)
            
            print(f"\n✅ 处理完成！")
            
            # 清理临时文件（可选）
            # shutil.rmtree(work_dir)
            
            return True
            
        except Exception as e:
            print(f"\n❌ 处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理模型
            if self.model:
                self.model.feat_extractor.cleanup_temp_files()
    
    def run(self):
        """运行长视频批量处理"""
        print("🔥 FireRedASR 长视频转文字批量处理系统")
        print("=" * 60)
        
        # 检查依赖
        if not self.check_dependencies():
            return
        
        # 扫描文件
        files = self.scan_long_media_files()
        if not self.display_files(files):
            return
        
        # 用户确认
        try:
            confirm = input(f"\n是否处理这 {len(files)} 个长视频文件? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes', '是']:
                print("👋 用户取消操作")
                return
        except KeyboardInterrupt:
            print("\n\n👋 用户取消操作")
            return
        
        # 询问 VAD 参数
        try:
            custom = input("\n是否使用自定义 VAD 参数? (y/n) [默认: n]: ").strip().lower()
            if custom in ['y', 'yes', '是']:
                self.max_speech_duration_s = int(input("最大语音段长度（秒）[默认: 30]: ") or "30")
                self.min_silence_duration_ms = int(input("最小静音间隔（毫秒）[默认: 500]: ") or "500")
                self.min_speech_duration_ms = int(input("最小语音段长度（毫秒）[默认: 1000]: ") or "1000")
        except KeyboardInterrupt:
            print("\n\n👋 用户取消操作")
            return
        
        # 批量处理
        print(f"\n🚀 开始批量处理...")
        success_count = 0
        
        for i, file_path in enumerate(files, 1):
            print(f"\n\n[{i}/{len(files)}] 处理进度")
            if self.process_long_video(file_path):
                success_count += 1
            
            # 询问是否继续
            if i < len(files):
                try:
                    cont = input("\n继续处理下一个文件? (y/n) [默认: y]: ").strip().lower()
                    if cont in ['n', 'no', '否']:
                        print("👋 用户停止处理")
                        break
                except KeyboardInterrupt:
                    print("\n\n👋 用户中断处理")
                    break
        
        # 总结
        print("\n" + "=" * 60)
        print(f"✅ 批量处理完成!")
        print(f"📊 总计: {len(files)} 个文件, 成功: {success_count} 个")
        print(f"📁 结果保存在: {self.output_dir}")
        
        # 清理临时目录
        if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
            self.temp_dir.rmdir()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='FireRedASR 长视频转文字批量处理系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                # 交互式处理
  %(prog)s --model_type llm               # 使用 LLM 模型
  %(prog)s --max_duration 45              # 设置最大段长为 45 秒
  %(prog)s --min_silence 300              # 设置最小静音为 300 毫秒
        """
    )
    
    parser.add_argument('--model_type', type=str, choices=['aed', 'llm'],
                        help='模型类型（如不指定则交互式选择）')
    parser.add_argument('--max_duration', type=int, default=30,
                        help='最大语音段长度（秒）')
    parser.add_argument('--min_silence', type=int, default=500,
                        help='最小静音间隔（毫秒）')
    parser.add_argument('--min_speech', type=int, default=1000,
                        help='最小语音段长度（毫秒）')
    
    # 标点恢复相关参数
    parser.add_argument('--enable-punctuation', action='store_true', default=True,
                        help='启用标点恢复（默认启用）')
    parser.add_argument('--disable-punctuation', action='store_true',
                        help='禁用标点恢复')
    parser.add_argument('--punctuation-model-dir', type=str,
                        help='自定义标点恢复模型路径')
    parser.add_argument('--punctuation-chunk-size', type=int, default=256,
                        help='标点恢复文本块大小（默认: 256）')
    parser.add_argument('--punctuation-stride', type=int, default=128,
                        help='标点恢复滑动窗口步长（默认: 128）')
    
    # 分段相关参数
    parser.add_argument('--enable-paragraph', action='store_true',
                        help='启用自然段分段功能')
    parser.add_argument('--paragraph-method', type=str, default='rule',
                        choices=['rule', 'semantic', 'hybrid'],
                        help='分段方法：rule（规则）、semantic（语义）、hybrid（混合）')
    parser.add_argument('--min-paragraph-length', type=int, default=50,
                        help='最小段落长度（默认: 50字）')
    parser.add_argument('--max-paragraph-length', type=int, default=500,
                        help='最大段落长度（默认: 500字）')
    
    args = parser.parse_args()
    
    # 检查是否在正确的目录
    if not Path("fireredasr").exists():
        print("❌ 错误: 请在 FireRedASR 项目根目录下运行此脚本")
        return
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(Path.cwd()) + ":" + os.environ.get('PYTHONPATH', '')
    
    try:
        transcriber = LongVideoTranscriber()
        
        # 设置参数
        if args.model_type:
            transcriber.model_type = args.model_type
        transcriber.max_speech_duration_s = args.max_duration
        transcriber.min_silence_duration_ms = args.min_silence
        transcriber.min_speech_duration_ms = args.min_speech
        
        # 设置标点恢复参数
        if args.disable_punctuation:
            transcriber.enable_punctuation = False
        else:
            transcriber.enable_punctuation = True
        
        if args.punctuation_model_dir:
            transcriber.punctuation_model_dir = args.punctuation_model_dir
        transcriber.punctuation_chunk_size = args.punctuation_chunk_size
        transcriber.punctuation_stride = args.punctuation_stride
        
        # 设置分段参数
        transcriber.enable_paragraph = args.enable_paragraph
        transcriber.paragraph_method = args.paragraph_method
        transcriber.min_paragraph_length = args.min_paragraph_length
        transcriber.max_paragraph_length = args.max_paragraph_length
        
        transcriber.run()
        
    except Exception as e:
        print(f"❌ 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()