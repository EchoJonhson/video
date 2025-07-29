#!/usr/bin/env python3
"""
FireRedASR 长音频转文字完整流程

功能：
- 自动将长音频文件切片（使用 WhisperX VAD）
- 批量使用 FireRedASR 模型转写
- 拼接成完整文字稿
- 支持时间戳和字幕格式输出

使用方法：
    python long_audio_transcribe.py --input_audio your_video.mp4 --model_type aed --model_dir pretrained_models/FireRedASR-AED-L
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.utils.punctuation_restore import PunctuationRestorer
from fireredasr.utils.paragraph_segmentation import ParagraphSegmenter


class LongAudioTranscriber:
    def __init__(self, model_type="aed", model_dir=None, output_dir="long_audio_output"):
        self.model_type = model_type
        self.model_dir = model_dir
        self.output_dir = Path(output_dir)
        self.segments_dir = self.output_dir / "segments"
        self.transcripts_dir = self.output_dir / "transcripts"
        self.model = None
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # 检查 whisperx
        try:
            import whisperx
            print("✅ whisperx 已安装")
        except ImportError:
            print("❌ whisperx 未安装，请运行: pip install whisperx")
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
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ 音频转换失败: {e.stderr}")
            return None
    
    def segment_audio_with_vad(self, audio_path):
        """使用 VAD 切片音频"""
        print("✂️ 使用 VAD 切片音频...")
        
        try:
            # 使用 silero-vad 进行语音活动检测
            import torch
            import torchaudio
            from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
            
            # 加载 VAD 模型
            model = load_silero_vad()
            
            # 读取音频
            wav = read_audio(str(audio_path))
            
            # 获取语音时间戳
            speech_timestamps = get_speech_timestamps(
                wav, model, 
                return_seconds=True,
                min_speech_duration_ms=1000,  # 最小语音段长度 1秒
                max_speech_duration_s=30,     # 最大语音段长度 30秒
                min_silence_duration_ms=500   # 最小静音间隔 0.5秒
            )
            
            print(f"📊 检测到 {len(speech_timestamps)} 个语音段")
            
            # 保存分段信息
            segments_info = []
            
            # 切片音频
            for i, segment in enumerate(speech_timestamps):
                start_time = segment['start']
                end_time = segment['end']
                duration = end_time - start_time
                
                # 生成分段文件名
                segment_filename = f"segment_{i:03d}.wav"
                segment_path = self.segments_dir / segment_filename
                
                # 使用 ffmpeg 切片
                cmd = [
                    "ffmpeg", "-i", str(audio_path),
                    "-ss", str(start_time),
                    "-t", str(duration),
                    "-acodec", "copy",
                    "-y",
                    str(segment_path)
                ]
                
                subprocess.run(cmd, capture_output=True, check=True)
                
                # 记录分段信息
                segment_info = {
                    "id": i,
                    "filename": segment_filename,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration
                }
                segments_info.append(segment_info)
                
                print(f"  ✅ 分段 {i:03d}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
            
            # 保存分段信息到 JSON 文件
            segments_json_path = self.output_dir / "segments.json"
            with open(segments_json_path, 'w', encoding='utf-8') as f:
                json.dump(segments_info, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 音频切片完成，共 {len(segments_info)} 个分段")
            print(f"📄 分段信息保存至: {segments_json_path}")
            
            return segments_info
            
        except Exception as e:
            print(f"❌ 音频切片失败: {str(e)}")
            return None
    
    def load_fireredasr_model(self):
        """加载 FireRedASR 模型"""
        if self.model is None:
            print(f"🔄 加载 FireRedASR-{self.model_type.upper()} 模型...")
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
    
    def transcribe_segments(self, segments_info):
        """批量转写音频分段"""
        print(f"🎯 开始批量转写 {len(segments_info)} 个音频分段...")
        
        if not self.load_fireredasr_model():
            return None
        
        decode_config = self.get_decode_config()
        transcription_results = []
        
        try:
            for i, segment_info in enumerate(segments_info):
                segment_filename = segment_info['filename']
                segment_path = self.segments_dir / segment_filename
                
                print(f"\n[{i+1}/{len(segments_info)}] 转写: {segment_filename}")
                
                # 转写单个分段
                uttid = f"segment_{i:03d}"
                start_time = time.time()
                
                results = self.model.transcribe(
                    [uttid], [str(segment_path)], decode_config
                )
                
                process_time = time.time() - start_time
                
                if results and len(results) > 0:
                    result = results[0]
                    text = result['text']
                    rtf = float(result.get('rtf', 0))
                    
                    print(f"  ✅ 转写完成 (耗时: {process_time:.2f}s, RTF: {rtf:.4f})")
                    print(f"  📝 结果: {text}")
                    
                    # 保存单个转写结果
                    transcript_filename = f"segment_{i:03d}.txt"
                    transcript_path = self.transcripts_dir / transcript_filename
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    # 记录转写结果
                    transcription_result = {
                        "id": i,
                        "filename": segment_filename,
                        "transcript_file": transcript_filename,
                        "text": text,
                        "start_time": segment_info['start_time'],
                        "end_time": segment_info['end_time'],
                        "duration": segment_info['duration'],
                        "process_time": process_time,
                        "rtf": rtf
                    }
                    transcription_results.append(transcription_result)
                    
                else:
                    print(f"  ❌ 转写失败: 没有返回结果")
                    transcription_results.append(None)
            
            # 保存转写结果到 JSON 文件
            transcripts_json_path = self.output_dir / "transcripts.json"
            with open(transcripts_json_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_results, f, ensure_ascii=False, indent=2)
            
            successful = len([r for r in transcription_results if r is not None])
            print(f"\n✅ 批量转写完成! 成功: {successful}/{len(segments_info)}")
            print(f"📄 转写结果保存至: {transcripts_json_path}")
            
            return transcription_results
            
        except Exception as e:
            print(f"❌ 批量转写失败: {str(e)}")
            return None
        
        finally:
            # 清理临时文件
            if self.model:
                self.model.feat_extractor.cleanup_temp_files()
    
    def merge_transcripts(self, transcription_results, output_formats=['txt', 'srt']):
        """拼接转写结果为完整文字稿"""
        print("📝 拼接转写结果...")
        
        if not transcription_results:
            print("❌ 没有转写结果可拼接")
            return None
        
        # 过滤掉失败的转写结果
        valid_results = [r for r in transcription_results if r is not None]
        
        if not valid_results:
            print("❌ 没有有效的转写结果")
            return None
        
        # 按时间顺序排序
        valid_results.sort(key=lambda x: x['start_time'])
        
        output_files = []
        
        # 生成纯文本格式
        if 'txt' in output_formats:
            txt_path = self.output_dir / "full_transcript.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"FireRedASR 长音频转写结果\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"使用模型: FireRedASR-{self.model_type.upper()}\n")
                f.write(f"总时长: {self.format_time(sum(r['duration'] for r in valid_results))}\n")
                f.write("=" * 60 + "\n\n")
                
                # 将所有文本连接成连续段落
                all_text = []
                for result in valid_results:
                    text = result['text'].strip()
                    if text:  # 只添加非空文本
                        all_text.append(text)
                
                # 使用空格连接，形成连续文本
                continuous_text = ' '.join(all_text)
                f.write(continuous_text)
            
            # 同时保存一个带时间戳的版本（可选）
            timestamp_txt_path = self.output_dir / "full_transcript_with_timestamps.txt"
            with open(timestamp_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"FireRedASR 长音频转写结果（带时间戳）\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"使用模型: FireRedASR-{self.model_type.upper()}\n")
                f.write(f"总分段数: {len(valid_results)}\n")
                f.write("=" * 60 + "\n\n")
                
                for result in valid_results:
                    start_time = result['start_time']
                    end_time = result['end_time']
                    text = result['text']
                    
                    # 格式化时间
                    start_str = self.format_time(start_time)
                    end_str = self.format_time(end_time)
                    
                    f.write(f"[{start_str} - {end_str}] {text}\n\n")
            
            output_files.append(timestamp_txt_path)
            print(f"📄 带时间戳文本文件: {timestamp_txt_path}")
            
            output_files.append(txt_path)
            print(f"📄 纯文本文件: {txt_path}")
        
        # 生成 SRT 字幕格式
        if 'srt' in output_formats:
            srt_path = self.output_dir / "full_transcript.srt"
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, result in enumerate(valid_results, 1):
                    start_time = result['start_time']
                    end_time = result['end_time']
                    text = result['text']
                    
                    # SRT 时间格式
                    start_srt = self.format_time_srt(start_time)
                    end_srt = self.format_time_srt(end_time)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_srt} --> {end_srt}\n")
                    f.write(f"{text}\n\n")
            
            output_files.append(srt_path)
            print(f"📄 SRT 字幕文件: {srt_path}")
        
        # 生成统计信息
        total_duration = sum(r['duration'] for r in valid_results)
        total_process_time = sum(r['process_time'] for r in valid_results)
        avg_rtf = sum(r['rtf'] for r in valid_results) / len(valid_results)
        
        stats = {
            "total_segments": len(valid_results),
            "total_duration": total_duration,
            "total_process_time": total_process_time,
            "average_rtf": avg_rtf,
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat()
        }
        
        stats_path = self.output_dir / "transcription_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 转写统计:")
        print(f"  总分段数: {len(valid_results)}")
        print(f"  总时长: {self.format_time(total_duration)}")
        print(f"  处理时间: {total_process_time:.2f}s")
        print(f"  平均 RTF: {avg_rtf:.4f}")
        print(f"📄 统计信息: {stats_path}")
        
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
                # 提取所有文本内容
                full_text = '\n'.join([r['text'] for r in valid_results])
                punctuated_text = self.punctuation_restorer.restore_punctuation(full_text)
                
                # 保存带标点的纯文本
                if 'txt' in output_formats:
                    punctuated_txt_path = self.output_dir / "full_transcript_with_punctuation.txt"
                    with open(punctuated_txt_path, 'w', encoding='utf-8') as f:
                        f.write(f"FireRedASR 长音频转写结果（带标点符号）\n")
                        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"使用模型: FireRedASR-{self.model_type.upper()}\n")
                        f.write(f"总时长: {self.format_time(sum(r['duration'] for r in valid_results))}\n")
                        f.write("=" * 60 + "\n\n")
                        
                        # 直接写入带标点的连续文本
                        f.write(punctuated_text)
                    
                    output_files.append(punctuated_txt_path)
                    print(f"📄 带标点文本文件: {punctuated_txt_path}")
                
                # 生成带标点的 SRT 字幕
                if 'srt' in output_formats:
                    punctuated_srt_path = self.output_dir / "full_transcript_with_punctuation.srt"
                    with open(punctuated_srt_path, 'w', encoding='utf-8') as f:
                        punctuated_lines = punctuated_text.split('\n')
                        for i, result in enumerate(valid_results, 1):
                            start_time = result['start_time']
                            end_time = result['end_time']
                            
                            # 尝试使用对应的带标点文本
                            if i-1 < len(punctuated_lines):
                                text = punctuated_lines[i-1]
                            else:
                                text = result['text']
                            
                            # SRT 时间格式
                            start_srt = self.format_time_srt(start_time)
                            end_srt = self.format_time_srt(end_time)
                            
                            f.write(f"{i}\n")
                            f.write(f"{start_srt} --> {end_srt}\n")
                            f.write(f"{text}\n\n")
                    
                    output_files.append(punctuated_srt_path)
                    print(f"📄 带标点字幕文件: {punctuated_srt_path}")
                
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
                        if 'txt' in output_formats:
                            paragraph_txt_path = self.output_dir / "full_transcript_paragraphs.txt"
                            with open(paragraph_txt_path, 'w', encoding='utf-8') as f:
                                f.write(f"FireRedASR 长音频转写结果\n")
                                f.write(f"\n处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"使用模型: FireRedASR-{self.model_type.upper()}\n")
                                f.write(f"总时长: {self.format_time(sum(r['duration'] for r in valid_results))}\n")
                                f.write(f"段落数: {len(paragraphs)}\n")
                                f.write("\n" + "=" * 60 + "\n\n")
                                
                                # 使用书籍排版格式
                                for i, para in enumerate(paragraphs, 1):
                                    # 段首缩进4个空格
                                    f.write(f"    {para}\n\n")
                            
                            # 同时生成 Markdown 格式
                            markdown_path = self.output_dir / "full_transcript_paragraphs.md"
                            with open(markdown_path, 'w', encoding='utf-8') as f:
                                # Markdown 头部
                                f.write(f"# 音频转写文稿\n\n")
                                f.write(f"**处理时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
                                f.write(f"**音频时长:** {self.format_time(sum(r['duration'] for r in valid_results))}  \n")
                                f.write(f"**段落数量:** {len(paragraphs)}  \n\n")
                                f.write("---\n\n")
                                
                                # 正文内容
                                for i, para in enumerate(paragraphs, 1):
                                    f.write(f"{para}\n\n")
                            
                            output_files.append(markdown_path)
                            print(f"📄 Markdown 文件: {markdown_path}")
                            
                            output_files.append(paragraph_txt_path)
                            print(f"📄 自然段格式文件: {paragraph_txt_path}")
                            print(f"   共分为 {len(paragraphs)} 个自然段")
                        
                    except Exception as e:
                        print(f"⚠️ 分段处理失败: {str(e)}")
                        print("   将保留带标点版本")
                    
            except Exception as e:
                print(f"⚠️ 标点恢复失败: {str(e)}")
                print("   将保留无标点版本")
        
        return output_files
    
    def format_time(self, seconds):
        """格式化时间为 HH:MM:SS 格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def format_time_srt(self, seconds):
        """格式化时间为 SRT 格式 HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def process_long_audio(self, input_audio, output_formats=['txt', 'srt']):
        """完整的长音频处理流程"""
        print("🔥 FireRedASR 长音频转文字完整流程")
        print("=" * 60)
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 第一步：准备音频
        print("\n🔹 第一步：准备音频")
        prepared_audio = self.prepare_audio(input_audio)
        if not prepared_audio:
            return False
        
        # 第二步：切片音频
        print("\n🔹 第二步：VAD 切片音频")
        segments_info = self.segment_audio_with_vad(prepared_audio)
        if not segments_info:
            return False
        
        # 第三步：批量转写
        print("\n🔹 第三步：批量转写音频分段")
        transcription_results = self.transcribe_segments(segments_info)
        if not transcription_results:
            return False
        
        # 第四步：拼接结果
        print("\n🔹 第四步：拼接转写结果")
        output_files = self.merge_transcripts(transcription_results, output_formats)
        if not output_files:
            return False
        
        print("\n" + "=" * 60)
        print("✅ 长音频转文字流程完成!")
        print(f"📁 输出目录: {self.output_dir}")
        print("📄 输出文件:")
        for file_path in output_files:
            print(f"  - {file_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="FireRedASR 长音频转文字完整流程")
    parser.add_argument('--input_audio', type=str, required=True, help="输入音频/视频文件路径")
    parser.add_argument('--model_type', type=str, choices=['aed', 'llm'], default='aed', help="模型类型")
    parser.add_argument('--model_dir', type=str, required=True, help="模型目录路径")
    parser.add_argument('--output_dir', type=str, default='long_audio_output', help="输出目录")
    parser.add_argument('--output_formats', type=str, nargs='+', choices=['txt', 'srt'], default=['txt', 'srt'], help="输出格式")
    
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
    
    # 检查输入文件
    if not Path(args.input_audio).exists():
        print(f"❌ 错误: 输入文件不存在: {args.input_audio}")
        return
    
    # 检查模型目录
    if not Path(args.model_dir).exists():
        print(f"❌ 错误: 模型目录不存在: {args.model_dir}")
        return
    
    # 创建转写器
    transcriber = LongAudioTranscriber(
        model_type=args.model_type,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
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
    
    # 执行完整流程
    success = transcriber.process_long_audio(args.input_audio, args.output_formats)
    
    if success:
        print("\n🎉 处理成功完成!")
    else:
        print("\n❌ 处理失败")
        sys.exit(1)


if __name__ == "__main__":
    main()