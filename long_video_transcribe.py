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

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.utils.video_audio import is_video_file, is_audio_file


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
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (get_speech_timestamps, save_audio, read_audio, 
         VADIterator, collect_chunks) = utils
        
        print("✅ VAD 模型加载成功")
        return model, get_speech_timestamps, read_audio, save_audio
    
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
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True
        )
        
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
        """批量转写音频片段"""
        print("\n🎤 开始批量转写...")
        
        # 加载模型
        print(f"🔄 加载 {self.model_type.upper()} 模型...")
        start_time = time.time()
        
        try:
            self.model = FireRedAsr.from_pretrained(self.model_type, model_dir)
            load_time = time.time() - start_time
            print(f"✅ 模型加载成功 (耗时: {load_time:.2f}s)")
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return None
        
        # 获取解码配置
        if self.model_type == "aed":
            decode_config = {
                "use_gpu": 1,
                "beam_size": 3,
                "nbest": 1,
                "decode_max_len": 0,
                "softmax_smoothing": 1.25,
                "aed_length_penalty": 0.6,
                "eos_penalty": 1.0
            }
        else:  # llm
            decode_config = {
                "use_gpu": 1,
                "beam_size": 3,
                "decode_max_len": 0,
                "decode_min_len": 0,
                "repetition_penalty": 3.0,
                "llm_length_penalty": 1.0,
                "temperature": 1.0
            }
        
        # 读取分段信息
        segments_info_path = segments_dir / "segments.json"
        with open(segments_info_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        # 批量转写
        results = []
        total = len(segments)
        
        for i, segment in enumerate(segments):
            segment_path = segments_dir / segment['file']
            print(f"\n[{i+1}/{total}] 转写: {segment['file']}")
            
            try:
                uttid = f"segment_{segment['index']:03d}"
                start_time = time.time()
                
                result = self.model.transcribe([uttid], [str(segment_path)], decode_config)
                
                process_time = time.time() - start_time
                
                if result and len(result) > 0:
                    text = result[0]['text']
                    rtf = float(result[0].get('rtf', 0))
                    
                    print(f"✅ 完成 (耗时: {process_time:.2f}s, RTF: {rtf:.4f})")
                    print(f"📝 文本: {text}")
                    
                    # 保存单个结果
                    transcript_path = segments_dir.parent / "transcripts" / f"{uttid}.txt"
                    transcript_path.parent.mkdir(exist_ok=True)
                    
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    results.append({
                        'index': segment['index'],
                        'file': segment['file'],
                        'start': segment['start'],
                        'end': segment['end'],
                        'duration': segment['duration'],
                        'text': text,
                        'process_time': process_time,
                        'rtf': rtf
                    })
                else:
                    print(f"❌ 转写失败")
                    
            except Exception as e:
                print(f"❌ 处理出错: {str(e)}")
        
        # 保存转写结果汇总
        if results:
            results_path = segments_dir.parent / "transcripts.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 批量转写完成: {len(results)}/{total} 成功")
            return results
        else:
            print("\n❌ 没有成功转写的片段")
            return None
    
    def concatenate_results(self, results, output_base_path):
        """拼接转写结果"""
        print("\n📝 拼接转写结果...")
        
        # 按时间排序
        results.sort(key=lambda x: x['start'])
        
        # 生成纯文本
        full_text = []
        for result in results:
            full_text.append(result['text'])
        
        txt_path = Path(str(output_base_path) + ".txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_text))
        print(f"✅ 生成纯文本: {txt_path}")
        
        # 生成带时间戳的文本
        timestamp_text = []
        for result in results:
            start_time = str(timedelta(seconds=result['start'])).split('.')[0]
            end_time = str(timedelta(seconds=result['end'])).split('.')[0]
            timestamp_text.append(f"[{start_time} --> {end_time}]")
            timestamp_text.append(result['text'])
            timestamp_text.append("")
        
        timestamp_path = Path(str(output_base_path) + "_with_timestamps.txt")
        with open(timestamp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(timestamp_text))
        print(f"✅ 生成时间戳文本: {timestamp_path}")
        
        # 生成 SRT 字幕
        srt_lines = []
        for i, result in enumerate(results, 1):
            start_time = self.seconds_to_srt_time(result['start'])
            end_time = self.seconds_to_srt_time(result['end'])
            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(result['text'])
            srt_lines.append("")
        
        srt_path = Path(str(output_base_path) + ".srt")
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_lines))
        print(f"✅ 生成 SRT 字幕: {srt_path}")
        
        # 生成统计信息
        total_duration = results[-1]['end'] if results else 0
        total_process_time = sum(r['process_time'] for r in results)
        avg_rtf = sum(r['rtf'] for r in results) / len(results) if results else 0
        
        stats = {
            'total_segments': len(results),
            'total_duration': total_duration,
            'total_duration_formatted': str(timedelta(seconds=total_duration)),
            'total_process_time': total_process_time,
            'average_rtf': avg_rtf,
            'total_characters': sum(len(r['text']) for r in results),
            'model_type': self.model_type
        }
        
        stats_path = Path(str(output_base_path) + "_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 统计信息:")
        print(f"   总时长: {stats['total_duration_formatted']}")
        print(f"   处理时间: {total_process_time:.2f}s")
        print(f"   平均 RTF: {avg_rtf:.4f}")
        print(f"   总字符数: {stats['total_characters']}")
    
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
            model_dir = self.select_model()
            if not model_dir:
                return False
            
            results = self.batch_transcribe(segments_dir, model_dir)
            if not results:
                return False
            
            # 步骤4：拼接结果
            print("\n[步骤 4/4] 拼接结果...")
            output_base = self.output_dir / f"{input_path.stem}_transcription_{timestamp}"
            self.concatenate_results(results, output_base)
            
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
        
        transcriber.run()
        
    except Exception as e:
        print(f"❌ 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()