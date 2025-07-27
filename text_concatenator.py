#!/usr/bin/env python3
"""
文本拼接工具

将批量转写的文本文件按时间顺序拼接成完整的文字稿
支持多种输出格式：纯文本、SRT字幕、VTT字幕等

使用方法：
    python text_concatenator.py --input_dir transcripts/ --output_file full_transcript.txt --format txt srt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import re


class TextConcatenator:
    def __init__(self, output_formats=['txt']):
        self.output_formats = output_formats
        self.supported_formats = {'txt', 'srt', 'vtt', 'json'}
        
        # 验证输出格式
        for fmt in self.output_formats:
            if fmt not in self.supported_formats:
                raise ValueError(f"不支持的输出格式: {fmt}")
    
    def load_transcription_results(self, input_dir):
        """加载转写结果"""
        input_path = Path(input_dir)
        
        # 尝试加载批量转写结果 JSON
        results_json_path = input_path / "batch_transcription_results.json"
        if results_json_path.exists():
            try:
                with open(results_json_path, 'r', encoding='utf-8') as f:
                    batch_results = json.load(f)
                print(f"📄 加载批量转写结果: {results_json_path}")
                return batch_results['results']
            except Exception as e:
                print(f"⚠️ 警告: 无法加载批量转写结果: {e}")
        
        # 尝试加载分段信息和文本文件
        segments_json_path = input_path.parent / "segments.json"
        if segments_json_path.exists():
            try:
                with open(segments_json_path, 'r', encoding='utf-8') as f:
                    segments_data = json.load(f)
                segments_info = segments_data.get('segments', [])
                print(f"📄 加载分段信息: {segments_json_path}")
                
                # 匹配文本文件
                results = []
                for segment in segments_info:
                    segment_id = segment['id']
                    filename = segment['filename']
                    text_filename = Path(filename).stem + '.txt'
                    text_path = input_path / text_filename
                    
                    if text_path.exists():
                        try:
                            with open(text_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            
                            result = {
                                "id": segment_id,
                                "filename": filename,
                                "text": text,
                                "start_time": segment['start_time'],
                                "end_time": segment['end_time'],
                                "duration": segment['duration'],
                                "success": True
                            }
                            results.append(result)
                            
                        except Exception as e:
                            print(f"⚠️ 警告: 无法读取文本文件 {text_path}: {e}")
                    else:
                        print(f"⚠️ 警告: 文本文件不存在: {text_path}")
                
                return results
                
            except Exception as e:
                print(f"⚠️ 警告: 无法加载分段信息: {e}")
        
        # 直接扫描文本文件
        print("📂 直接扫描文本文件...")
        text_files = list(input_path.glob('*.txt'))
        text_files.sort()
        
        results = []
        for i, text_path in enumerate(text_files):
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                # 尝试从文件名提取序号
                match = re.search(r'segment_(\d+)', text_path.stem)
                segment_id = int(match.group(1)) if match else i
                
                result = {
                    "id": segment_id,
                    "filename": text_path.name,
                    "text": text,
                    "start_time": 0,  # 默认值
                    "end_time": 0,    # 默认值
                    "duration": 0,    # 默认值
                    "success": True
                }
                results.append(result)
                
            except Exception as e:
                print(f"⚠️ 警告: 无法读取文本文件 {text_path}: {e}")
        
        return results
    
    def filter_and_sort_results(self, results):
        """过滤和排序转写结果"""
        # 过滤成功的结果
        valid_results = [r for r in results if r.get('success', False) and r.get('text', '').strip()]
        
        if not valid_results:
            print("❌ 没有有效的转写结果")
            return []
        
        # 按 ID 或时间排序
        if all('start_time' in r for r in valid_results):
            valid_results.sort(key=lambda x: x['start_time'])
            print(f"📊 按时间排序 {len(valid_results)} 个有效结果")
        else:
            valid_results.sort(key=lambda x: x['id'])
            print(f"📊 按 ID 排序 {len(valid_results)} 个有效结果")
        
        return valid_results
    
    def generate_txt_format(self, results, output_path):
        """生成纯文本格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入头部信息
            f.write("FireRedASR 长音频转写结果\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总分段数: {len(results)}\n")
            f.write("=" * 60 + "\n\n")
            
            # 写入转写内容
            for result in results:
                start_time = result.get('start_time', 0)
                end_time = result.get('end_time', 0)
                text = result['text']
                
                if start_time > 0 or end_time > 0:
                    # 有时间信息
                    start_str = self.format_time(start_time)
                    end_str = self.format_time(end_time)
                    f.write(f"[{start_str} - {end_str}] {text}\n\n")
                else:
                    # 无时间信息
                    f.write(f"{text}\n\n")
        
        print(f"📄 纯文本文件: {output_path}")
    
    def generate_srt_format(self, results, output_path):
        """生成 SRT 字幕格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results, 1):
                start_time = result.get('start_time', 0)
                end_time = result.get('end_time', 0)
                text = result['text']
                
                # SRT 时间格式
                start_srt = self.format_time_srt(start_time)
                end_srt = self.format_time_srt(end_time)
                
                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{text}\n\n")
        
        print(f"📄 SRT 字幕文件: {output_path}")
    
    def generate_vtt_format(self, results, output_path):
        """生成 VTT 字幕格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for i, result in enumerate(results, 1):
                start_time = result.get('start_time', 0)
                end_time = result.get('end_time', 0)
                text = result['text']
                
                # VTT 时间格式
                start_vtt = self.format_time_vtt(start_time)
                end_vtt = self.format_time_vtt(end_time)
                
                f.write(f"{start_vtt} --> {end_vtt}\n")
                f.write(f"{text}\n\n")
        
        print(f"📄 VTT 字幕文件: {output_path}")
    
    def generate_json_format(self, results, output_path):
        """生成 JSON 格式"""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_segments": len(results),
            "segments": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 JSON 文件: {output_path}")
    
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
    
    def format_time_vtt(self, seconds):
        """格式化时间为 VTT 格式 HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    def concatenate_texts(self, input_dir, output_file):
        """拼接文本的主要流程"""
        print("📝 文本拼接工具")
        print("=" * 50)
        
        # 加载转写结果
        print("\n🔹 加载转写结果")
        results = self.load_transcription_results(input_dir)
        
        if not results:
            print("❌ 没有找到转写结果")
            return False
        
        print(f"📊 找到 {len(results)} 个转写结果")
        
        # 过滤和排序
        print("\n🔹 过滤和排序结果")
        valid_results = self.filter_and_sort_results(results)
        
        if not valid_results:
            return False
        
        # 生成输出文件
        print("\n🔹 生成输出文件")
        output_path = Path(output_file)
        output_dir = output_path.parent
        output_stem = output_path.stem
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        for fmt in self.output_formats:
            if fmt == 'txt':
                txt_path = output_dir / f"{output_stem}.txt"
                self.generate_txt_format(valid_results, txt_path)
                output_files.append(txt_path)
            
            elif fmt == 'srt':
                srt_path = output_dir / f"{output_stem}.srt"
                self.generate_srt_format(valid_results, srt_path)
                output_files.append(srt_path)
            
            elif fmt == 'vtt':
                vtt_path = output_dir / f"{output_stem}.vtt"
                self.generate_vtt_format(valid_results, vtt_path)
                output_files.append(vtt_path)
            
            elif fmt == 'json':
                json_path = output_dir / f"{output_stem}.json"
                self.generate_json_format(valid_results, json_path)
                output_files.append(json_path)
        
        # 生成统计信息
        total_duration = sum(r.get('duration', 0) for r in valid_results)
        total_chars = sum(len(r['text']) for r in valid_results)
        
        print("\n📊 拼接统计:")
        print(f"  有效分段数: {len(valid_results)}")
        if total_duration > 0:
            print(f"  总时长: {self.format_time(total_duration)}")
        print(f"  总字符数: {total_chars}")
        print(f"  输出格式: {', '.join(self.output_formats)}")
        
        print("\n" + "=" * 50)
        print("✅ 文本拼接完成!")
        print("📄 输出文件:")
        for file_path in output_files:
            print(f"  - {file_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="文本拼接工具 - 将转写结果拼接成完整文档")
    parser.add_argument('--input_dir', type=str, required=True, help="转写结果目录")
    parser.add_argument('--output_file', type=str, default='full_transcript.txt', help="输出文件路径（不含扩展名）")
    parser.add_argument('--format', type=str, nargs='+', choices=['txt', 'srt', 'vtt', 'json'], 
                       default=['txt'], help="输出格式")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not Path(args.input_dir).exists():
        print(f"❌ 错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 处理输出文件路径
    output_file = args.output_file
    if output_file.endswith('.txt'):
        output_file = output_file[:-4]  # 移除扩展名
    
    # 创建拼接器
    concatenator = TextConcatenator(output_formats=args.format)
    
    # 执行拼接
    success = concatenator.concatenate_texts(args.input_dir, output_file)
    
    if success:
        print("\n🎉 拼接成功完成!")
    else:
        print("\n❌ 拼接失败")
        sys.exit(1)


if __name__ == "__main__":
    main()