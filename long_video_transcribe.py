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
from fireredasr.utils.cpu_optimization_config import CPUOptimizationConfig
from utils.terminal_beautifier import TerminalBeautifier, create_progress_bar


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
        self.beautifier = TerminalBeautifier()
        self.beautifier.print_header("FireRedASR 长视频转写系统", "智能语音识别处理引擎 v2.0")
        
        # 使用动画显示初始化过程
        spinner = self.beautifier.create_spinner("正在初始化智能处理系统...")
        spinner.start()
        
        self.hardware_manager = get_hardware_manager()
        self.smart_loader = create_smart_loader(self.hardware_manager)
        self.parallel_processor = None
        
        spinner.stop(success_msg="智能处理系统初始化完成！")
        
        # 检查环境变量配置
        force_cpu = os.environ.get('FIREREDASR_FORCE_CPU', '').lower() in ['1', 'true', 'yes']
        if force_cpu:
            self.beautifier.print_warning("强制使用 CPU 模式 (FIREREDASR_FORCE_CPU=1)")
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
        self.beautifier.print_section("检查依赖", "🔍")
        
        # 检查 ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            self.beautifier.print_success("ffmpeg 已安装")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.beautifier.print_error("ffmpeg 未安装，请先安装 ffmpeg")
            return False
        
        # 检查 torchaudio
        try:
            import torchaudio
            self.beautifier.print_success("torchaudio 已安装")
        except ImportError:
            self.beautifier.print_error("torchaudio 未安装，请运行: pip install torchaudio")
            return False
        
        return True
    
    def scan_long_media_files(self):
        """扫描输入文件夹中的长媒体文件"""
        if not self.input_dir.exists():
            self.beautifier.print_error(f"输入文件夹不存在: {self.input_dir}")
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
            self.beautifier.print_error("在 Use/Input/ 文件夹中没有找到大型媒体文件")
            self.beautifier.print_info("提示：长视频处理适用于大于10MB的音视频文件")
            return False
        
        self.beautifier.print_section(f"扫描结果：发现 {len(files)} 个待处理文件", "📁")
        
        # 创建文件列表表格
        headers = ["序号", "文件名", "大小", "类型", "预估时长"]
        rows = []
        
        total_size = 0
        for i, file_path in enumerate(files, 1):
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            total_size += file_size
            
            if is_video_file(str(file_path)):
                file_type = "视频"
                icon = "📹"
            else:
                file_type = "音频"
                icon = "🎵"
            
            # 估算时长（基于文件大小的粗略估计）
            estimated_duration = file_size * 0.5  # 假设2MB/分钟
            duration_str = f"~{int(estimated_duration)}分钟"
            
            rows.append([
                f"{icon} {i}",
                file_path.name[:40] + ("..." if len(file_path.name) > 40 else ""),
                f"{file_size:.1f} MB",
                file_type,
                duration_str
            ])
        
        self.beautifier.print_table(headers, rows)
        self.beautifier.print_info(f"\n总文件大小: {total_size:.1f} MB", "💾")
        return True
    
    def get_model_dir(self):
        """根据命令行参数获取模型目录"""
        if not self.model_type:
            self.beautifier.print_error("未指定模型类型，请使用 --model_type 参数")
            return None
        
        if self.model_type == "aed":
            model_dir = "pretrained_models/FireRedASR-AED-L"
            self.beautifier.print_success("使用 FireRedASR-AED 模型 (快速, 适合长音频)")
        elif self.model_type == "llm":
            model_dir = "pretrained_models/FireRedASR-LLM-L"
            self.beautifier.print_success("使用 FireRedASR-LLM 模型 (高精度, 处理较慢)")
        else:
            self.beautifier.print_error(f"未知模型类型: {self.model_type}")
            return None
        
        # 检查模型路径
        if not Path(model_dir).exists():
            self.beautifier.print_error(f"模型目录不存在: {model_dir}")
            self.beautifier.print_info("请先下载模型文件，参考 step.md 文档")
            return None
        
        return model_dir
    
    def select_model(self):
        """让用户选择模型"""
        self.beautifier.print_section("请选择要使用的模型", "🤖")
        model_options = [
            ["1", "FireRedASR-AED", "快速, 适合长音频"],
            ["2", "FireRedASR-LLM", "高精度, 处理较慢"]
        ]
        self.beautifier.print_table(["选项", "模型", "特点"], model_options)
        
        while True:
            try:
                choice = input("\n请输入选择 (1 或 2): ").strip()
                if choice == "1":
                    self.model_type = "aed"
                    model_dir = "pretrained_models/FireRedASR-AED-L"
                    self.beautifier.print_success("选择了 FireRedASR-AED 模型")
                    break
                elif choice == "2":
                    self.model_type = "llm"
                    model_dir = "pretrained_models/FireRedASR-LLM-L"
                    self.beautifier.print_success("选择了 FireRedASR-LLM 模型")
                    break
                else:
                    self.beautifier.print_error("无效输入，请输入 1 或 2")
            except KeyboardInterrupt:
                self.beautifier.print_warning("用户取消操作", "👋")
                return None
        
        # 检查模型路径
        if not Path(model_dir).exists():
            self.beautifier.print_error(f"模型目录不存在: {model_dir}")
            self.beautifier.print_info("请从 https://huggingface.co/fireredteam 下载模型文件")
            return None
        
        return model_dir
    
    def prepare_audio(self, input_path, output_path):
        """准备音频：转换为 16kHz 单声道 WAV 格式"""
        self.beautifier.print_info(f"准备音频: {input_path.name}", "🎵")
        
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
            self.beautifier.print_success("音频准备完成")
            return True
        except subprocess.CalledProcessError as e:
            self.beautifier.print_error(f"音频转换失败: {e.stderr}")
            return False
    
    def load_silero_vad(self):
        """加载 Silero VAD 模型"""
        self.beautifier.print_info("加载 VAD 模型...", "🔄")
        
        # 方法1: 使用 pip 安装的 silero-vad 包（推荐）
        try:
            self.beautifier.print_info("尝试使用 silero-vad 包...", "📦")
            from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
            
            model = load_silero_vad()
            
            # 创建兼容的 save_audio 函数
            def save_audio(path, tensor, sampling_rate):
                torchaudio.save(path, tensor, sampling_rate)
            
            self.beautifier.print_success("VAD 模型加载成功 (silero-vad 包)")
            return model, get_speech_timestamps, read_audio, save_audio
            
        except ImportError as e:
            self.beautifier.print_error(f"silero-vad 包未安装: {e}")
        except Exception as e:
            self.beautifier.print_error(f"silero-vad 包加载失败: {str(e)}")
        
        # 方法2: 尝试从 torch.hub 加载
        for attempt in range(2):
            try:
                self.beautifier.print_info(f"尝试从 torch.hub 加载 (尝试 {attempt + 1}/2)...", "📁")
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=attempt > 0,
                    trust_repo=True
                )
                (get_speech_timestamps, save_audio, read_audio, 
                 VADIterator, collect_chunks) = utils
                
                self.beautifier.print_success("VAD 模型加载成功 (torch.hub)")
                return model, get_speech_timestamps, read_audio, save_audio
                
            except Exception as e:
                self.beautifier.print_error(f"torch.hub 加载失败 (尝试 {attempt + 1}/2): {str(e)}")
                if attempt == 0:
                    time.sleep(3)
        
        # 如果所有方法都失败
        raise Exception("❌ VAD模型加载失败！请确保已安装 silero-vad: pip install silero-vad")
    
    def slice_audio_with_vad(self, audio_path, output_dir):
        """使用 VAD 切分音频"""
        self.beautifier.print_section("语音活动检测 (VAD)", "✂️")
        
        # 使用动画显示加载过程
        spinner = self.beautifier.create_spinner("正在加载 VAD 模型...")
        spinner.start()
        
        # 加载 VAD 模型
        vad_model, get_speech_timestamps, read_audio, save_audio = self.load_silero_vad()
        spinner.stop(success_msg="VAD 模型加载成功！")
        
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
            self.beautifier.print_error("没有检测到语音段")
            return []
        
        self.beautifier.print_success(f"检测到 {len(speech_timestamps)} 个初始语音段")
        
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
        
        self.beautifier.print_success(f"合并后得到 {len(segments)} 个语音段")
        
        # 保存音频段
        segment_files = []
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # 创建进度条
        progress_bar = self.beautifier.create_progress_bar(len(segments), "保存音频片段")
        
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
            
            # 更新进度条
            progress_bar.update(i + 1, f"片段 {i+1}/{len(segments)}")
        
        # 保存分段信息
        segments_info_path = output_dir / "segments.json"
        with open(segments_info_path, 'w', encoding='utf-8') as f:
            json.dump(segment_files, f, ensure_ascii=False, indent=2)
        
        self.beautifier.print_success(f"音频切分完成，共 {len(segment_files)} 个片段")
        return segment_files
    
    def batch_transcribe(self, segments_dir, model_dir):
        """智能批量转写音频片段"""
        self.beautifier.print_section("开始智能批量转写", "🎤")
        
        # 验证分段目录和文件
        if not segments_dir.exists():
            self.beautifier.print_error(f"分段目录不存在: {segments_dir}")
            return None
        
        segments_info_path = segments_dir / "segments.json"
        if not segments_info_path.exists():
            self.beautifier.print_error(f"分段信息文件不存在: {segments_info_path}")
            return None
        
        # 读取分段信息
        try:
            with open(segments_info_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            self.beautifier.print_info(f"加载分段信息: {len(segments)} 个片段", "📋")
        except Exception as e:
            self.beautifier.print_error(f"读取分段信息失败: {e}")
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
            self.beautifier.print_warning(f"{len(missing_files)} 个分段文件不存在")
            if len(missing_files) <= 5:
                for f in missing_files:
                    self.beautifier.print_info(f"  - {f}", "")
            else:
                for f in missing_files[:3]:
                    self.beautifier.print_info(f"  - {f}", "")
                self.beautifier.print_info(f"  ... 还有 {len(missing_files) - 3} 个文件", "")
        
        if not valid_segments:
            self.beautifier.print_error("没有有效的分段文件")
            return None
        
        self.beautifier.print_success(f"找到 {len(valid_segments)} 个有效分段文件")
        segments = valid_segments
        
        # 使用智能模型加载器
        self.model = self.smart_loader.load_model(self.model_type, model_dir)
        if not self.model:
            self.beautifier.print_error("模型加载失败")
            return None
        
        # 优化模型以进行推理
        self.smart_loader.optimize_for_inference()
        
        # 获取智能解码配置
        decode_config = self.smart_loader.get_transcribe_config()
        self.beautifier.print_info(f"解码配置: {decode_config}", "🎯")
        
        # 获取并行处理配置
        strategy = self.hardware_manager.get_optimal_config()['strategy']
        
        # 智能选择并行处理策略
        segment_count = len(segments)
        
        if self.model_type == "llm":
            # 导入CPU优化配置
            cpu_optimizer = CPUOptimizationConfig()
            
            # 获取动态优化配置
            opt_config = cpu_optimizer.get_dynamic_config(segment_count, "llm")
            
            # LLM 模型智能处理策略
            gpu_assisted = self.smart_loader.strategy.get('gpu_role') in ['encoder_only', 'feature_extraction']
            
            if gpu_assisted:
                # GPU辅助模式下的优化配置
                max_workers = opt_config["max_workers"]
                batch_size = opt_config["batch_size"]
                
                # 内存使用估算
                memory_est = cpu_optimizer.estimate_memory_usage("llm", max_workers)
                
                self.beautifier.print_model_config("llm", {
                    "max_workers": max_workers,
                    "batch_size": batch_size,
                    "memory_usage": memory_est['total_gb']
                })
                self.beautifier.print_info(f"🚀 LLM GPU辅助模式优化:")
                self.beautifier.print_info(f"   - 分段数: {segment_count}")
                self.beautifier.print_info(f"   - 并行线程: {max_workers} (原2个，现优化为{max_workers}个)")
                self.beautifier.print_info(f"   - 预估内存: {memory_est['total_gb']:.1f}GB / {memory_est['available_gb']:.1f}GB ({memory_est['usage_percent']:.1f}%)")
                self.beautifier.print_info(f"   - CPU配置: i9-14900KF (24栃32线程)")
                self.beautifier.print_info("📌 优化策略: 编码器在GPU，LLM主体在CPU，使用动态并行度调整")
                
                # 启用预读取优化
                self.prefetch_segments = opt_config["memory_config"]["prefetch_segments"]
                
            else:
                # 纯CPU模式保持原有策略
                max_workers = 1
                batch_size = 1
                if segment_count <= 10:
                    self.beautifier.print_warning(f"LLM 串行处理: 分段数较少({segment_count}个)，使用串行处理")
                else:
                    self.beautifier.print_warning("LLM 纯CPU模式，使用串行处理以确保稳定性")
        else:
            # AED 模型优化
            cpu_optimizer = CPUOptimizationConfig()
            opt_config = cpu_optimizer.get_dynamic_config(segment_count, "aed")
            
            max_workers = opt_config["max_workers"]
            batch_size = opt_config["batch_size"]
            
            self.beautifier.print_info("AED 智能并行优化:", "🔧")
            self.beautifier.print_info(f"   - 分段数: {segment_count}", "")
            self.beautifier.print_info(f"   - 并行线程: {max_workers}", "")
            self.beautifier.print_info(f"   - 批处理大小: {batch_size}", "")
        
        self.beautifier.print_info(f"处理配置: {max_workers} 线程, 批次大小: {batch_size}", "🔧")
        
        # 准备音频片段路径
        segment_paths = [segments_dir / segment['file'] for segment in segments]
        
        # 创建输出目录
        transcripts_dir = segments_dir.parent / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建线程锁以保护模型访问
        import threading
        model_lock = threading.Lock()
        
        # 预读取优化（如果启用）
        audio_cache = {}
        if hasattr(self, 'prefetch_segments') and self.prefetch_segments > 0:
            self.beautifier.print_info(f"启用预读取优化，预加载 {self.prefetch_segments} 个音频段...", "📥")
            from concurrent.futures import ThreadPoolExecutor
            
            def prefetch_audio(idx):
                if idx < len(segments):
                    segment_path = segments_dir / segments[idx]['file']
                    if segment_path.exists():
                        with open(segment_path, 'rb') as f:
                            audio_cache[idx] = segment_path
                            
            # 预加载前几个音频段
            with ThreadPoolExecutor(max_workers=2) as prefetch_executor:
                for i in range(min(self.prefetch_segments, len(segments))):
                    prefetch_executor.submit(prefetch_audio, i)
        
        # 创建转录函数
        def transcribe_single_segment(segment_path):
            """转录单个音频片段"""
            try:
                # 检查文件是否存在
                if not segment_path.exists():
                    self.beautifier.print_warning(f"跳过不存在的文件: {segment_path.name}")
                    return None
                
                # 找到对应的 segment 信息
                segment_info = None
                for seg in segments:
                    if segments_dir / seg['file'] == segment_path:
                        segment_info = seg
                        break
                
                if not segment_info:
                    self.beautifier.print_warning(f"找不到分段信息: {segment_path.name}")
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
                    self.beautifier.print_warning(f"模型转录无结果: {segment_path.name}")
                    return None
                
            except Exception as e:
                self.beautifier.print_error(f"转录片段失败 {segment_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        
        # 根据模型类型选择处理方式
        if max_workers == 1:
            # 串行处理
            self.beautifier.print_info(f"串行转写 {len(segment_paths)} 个片段...", "🚀")
            results = []
            
            # 创建进度条
            progress_bar = self.beautifier.create_progress_bar(len(segment_paths), "转写进度")
            
            for i, segment_path in enumerate(segment_paths):
                # 更新进度条
                progress_bar.update(i, f"处理: {segment_path.name}")
                
                result = transcribe_single_segment(segment_path)
                if result:
                    results.append(result)
                    
                # 定期清理内存
                if (i + 1) % 10 == 0:
                    import gc
                    gc.collect()
            
            # 完成进度条
            progress_bar.update(len(segment_paths), "转写完成！")
        else:
            # 并行处理
            self.beautifier.print_info(f"使用 {max_workers} 线程并行转写 {len(segment_paths)} 个片段...", "🚀")
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
                self.beautifier.print_success(f"智能批量转写完成: {len(results)}/{total} 成功")
                return results
            except Exception as e:
                self.beautifier.print_error(f"保存转写结果失败: {e}")
                return results  # 返回结果但记录保存失败
        else:
            self.beautifier.print_error("没有成功转写的片段")
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
        self.beautifier.print_section("拼接转写结果", "📝")
        
        # 按时间排序
        results.sort(key=lambda x: x['start'])
        
        # 获取输入文件的基本名称（不含扩展名）
        base_name = Path(input_filename).stem
        output_base = self.output_dir / base_name
        
        # 生成纯文本（连续段落格式，更适合阅读）
        full_text = []
        for result in results:
            text = result['text'].strip()
            if text:  # 只添加非空文本
                full_text.append(text)
        
        # 将文本拼接成连续段落，而不是逐行显示
        continuous_text = ' '.join(full_text)
        
        txt_path = self.generate_unique_filename(output_base, ".txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"FireRedASR 视频转写结果\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(continuous_text)
        self.beautifier.print_success(f"生成纯文本: {txt_path.name}")
        
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
        self.beautifier.print_success(f"生成时间戳文本: {timestamp_path.name}")
        
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
        self.beautifier.print_success(f"生成字幕文件: {srt_path.name}")
        
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
        
        self.beautifier.print_stats({
            "总时长": stats['total_duration_formatted'],
            "处理时间": f"{total_process_time:.2f}s",
            "平均 RTF": avg_rtf,
            "总字符数": stats['total_characters']
        })
        
        # 标点恢复处理
        if self.enable_punctuation:
            try:
                self.beautifier.print_section("开始标点恢复处理", "🔤")
                
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
                
                # 保存带标点的纯文本（书籍格式）
                punctuated_txt_path = self.generate_unique_filename(output_base, "_标点.txt")
                with open(punctuated_txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"FireRedASR 视频转写结果（带标点符号）\n")
                    f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(punctuated_text)
                self.beautifier.print_success(f"生成带标点文本: {punctuated_txt_path.name}")
                
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
                self.beautifier.print_success(f"生成带标点字幕: {punctuated_srt_path.name}")
                
                # 如果启用了分段功能
                if self.enable_paragraph and punctuated_text:
                    try:
                        self.beautifier.print_section("开始自然段分段处理", "📑")
                        
                        # 初始化分段器
                        if self.paragraph_segmenter is None:
                            self.paragraph_segmenter = ParagraphSegmenter(
                                min_length=self.min_paragraph_length,
                                max_length=self.max_paragraph_length
                            )
                        
                        # 执行分段
                        paragraphs = self.paragraph_segmenter.segment_paragraphs(punctuated_text)
                        
                        # 保存分段结果（优化的书籍排版格式）
                        paragraph_txt_path = self.generate_unique_filename(output_base, "_段落.txt")
                        with open(paragraph_txt_path, 'w', encoding='utf-8') as f:
                            # 文档头部信息
                            f.write(f"FireRedASR 视频转写结果\n")
                            f.write(f"\n文件: {base_name}\n")
                            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"总时长: {stats['total_duration_formatted']}\n")
                            f.write(f"段落数: {len(paragraphs)}\n")
                            f.write("\n" + "=" * 60 + "\n\n")
                            
                            # 正文内容 - 自然的书籍排版
                            for i, para in enumerate(paragraphs, 1):
                                # 使用缩进表示段落开始，而不是标号
                                f.write(f"    {para}\n\n")  # 段首缩进4个空格
                        
                        self.beautifier.print_success(f"生成自然段文件: {paragraph_txt_path.name}")
                        self.beautifier.print_info(f"   共分为 {len(paragraphs)} 个自然段")
                        
                        # 同时生成一个更精美的 Markdown 格式版本
                        markdown_path = self.generate_unique_filename(output_base, "_段落.md")
                        with open(markdown_path, 'w', encoding='utf-8') as f:
                            # Markdown 头部
                            f.write(f"# {base_name} - 转写文稿\n\n")
                            f.write(f"**处理时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
                            f.write(f"**视频时长:** {stats['total_duration_formatted']}  \n")
                            f.write(f"**段落数量:** {len(paragraphs)}  \n\n")
                            f.write("---\n\n")
                            
                            # 正文内容
                            for i, para in enumerate(paragraphs, 1):
                                f.write(f"{para}\n\n")
                        
                        self.beautifier.print_success(f"生成 Markdown 文件: {markdown_path.name}")
                        
                    except Exception as e:
                        self.beautifier.print_warning(f"分段处理失败: {str(e)}")
                        self.beautifier.print_info("   将保留带标点版本")
                
            except Exception as e:
                self.beautifier.print_warning(f"标点恢复失败: {str(e)}")
                self.beautifier.print_info("   将保留无标点版本")
    
    def seconds_to_srt_time(self, seconds):
        """将秒数转换为 SRT 时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def process_long_video(self, input_path):
        """处理单个长视频文件的完整流程"""
        self.beautifier.print_header(f"处理文件: {input_path.name}", "")
        
        # 创建临时工作目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = self.temp_dir / f"{input_path.stem}_{timestamp}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        segments_dir = work_dir / "segments"
        segments_dir.mkdir(exist_ok=True)
        
        try:
            # 步骤1：准备音频
            self.beautifier.print_step(1, 4, "准备音频")
            prepared_audio = work_dir / "prepared_audio.wav"
            if not self.prepare_audio(input_path, prepared_audio):
                return False
            
            # 步骤2：VAD 切片
            self.beautifier.print_step(2, 4, "VAD 语音检测和切片")
            segments = self.slice_audio_with_vad(prepared_audio, segments_dir)
            if not segments:
                return False
            
            # 步骤3：批量转写
            self.beautifier.print_step(3, 4, "批量转写")
            model_dir = self.get_model_dir()
            if not model_dir:
                return False
            
            results = self.batch_transcribe(segments_dir, model_dir)
            if not results:
                return False
            
            # 步骤4：拼接结果
            self.beautifier.print_step(4, 4, "拼接结果")
            self.concatenate_results(results, input_path.name)
            
            self.beautifier.print_success("处理完成！")
            
            # 清理临时文件（可选）
            # shutil.rmtree(work_dir)
            
            return True
            
        except Exception as e:
            self.beautifier.print_error(f"处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理模型
            if self.model:
                self.model.feat_extractor.cleanup_temp_files()
    
    def run(self):
        """运行长视频批量处理"""
        self.beautifier.print_header("FireRedASR 长视频转文字批量处理系统", "")
        
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
                self.beautifier.print_warning("用户取消操作", "👋")
                return
        except KeyboardInterrupt:
            self.beautifier.print_warning("\n用户取消操作", "👋")
            return
        
        # 询问 VAD 参数
        try:
            custom = input("\n是否使用自定义 VAD 参数? (y/n) [默认: n]: ").strip().lower()
            if custom in ['y', 'yes', '是']:
                self.max_speech_duration_s = int(input("最大语音段长度（秒）[默认: 30]: ") or "30")
                self.min_silence_duration_ms = int(input("最小静音间隔（毫秒）[默认: 500]: ") or "500")
                self.min_speech_duration_ms = int(input("最小语音段长度（毫秒）[默认: 1000]: ") or "1000")
        except KeyboardInterrupt:
            self.beautifier.print_warning("\n用户取消操作", "👋")
            return
        
        # 批量处理
        self.beautifier.print_section("开始批量处理", "🚀")
        success_count = 0
        
        for i, file_path in enumerate(files, 1):
            self.beautifier.print_info(f"\n[{i}/{len(files)}] 处理进度", "📄")
            if self.process_long_video(file_path):
                success_count += 1
            
            # 询问是否继续
            if i < len(files):
                try:
                    cont = input("\n继续处理下一个文件? (y/n) [默认: y]: ").strip().lower()
                    if cont in ['n', 'no', '否']:
                        self.beautifier.print_warning("用户停止处理", "👋")
                        break
                except KeyboardInterrupt:
                    self.beautifier.print_warning("\n用户中断处理", "👋")
                    break
        
        # 总结
        self.beautifier.print_summary(
            "批量处理完成",
            {
                "处理文件数": len(files),
                "成功转写": success_count,
                "失败文件": len(files) - success_count,
                "输出目录": str(self.output_dir),
                "处理时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            style="double"
        )
        
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