#!/usr/bin/env python3
"""
FireRedASR 批量语音识别脚本

功能：
- 自动扫描 Use/Input/ 文件夹中的音频和视频文件
- 用户选择使用的模型（AED或LLM）
- 批量进行语音识别转换
- 结果保存到 Use/Output/ 文件夹中
- 支持格式：WAV, MP3, FLAC, M4A, AAC, MP4, AVI, MOV, MKV, FLV, WMV

使用方法：
    python batch_transcribe.py
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.utils.video_audio import is_video_file, is_audio_file
from fireredasr.utils.punctuation_restore import PunctuationRestorer
from fireredasr.utils.paragraph_segmentation import ParagraphSegmenter


class BatchTranscriber:
    def __init__(self):
        self.input_dir = Path("Use/Input")
        self.output_dir = Path("Use/Output")
        self.supported_audio = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        self.supported_video = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        self.model = None
        self.model_type = None
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标点恢复相关
        self.enable_punctuation = True  # 默认启用标点恢复
        self.punctuation_restorer = None
        self.punctuation_model_dir = None
        self.punctuation_chunk_size = 256
        self.punctuation_stride = 128
        
        # 分段相关（批量模式下可能用处不大，但保持接口一致）
        self.enable_paragraph = False  # 默认不启用分段
        self.paragraph_segmenter = None
        self.min_paragraph_length = 50
        self.max_paragraph_length = 500
    
    def scan_input_files(self):
        """扫描输入文件夹中的媒体文件"""
        if not self.input_dir.exists():
            print(f"❌ 错误: 输入文件夹不存在: {self.input_dir}")
            print("请创建 Use/Input/ 文件夹并放入音频或视频文件")
            return []
        
        media_files = []
        all_extensions = self.supported_audio | self.supported_video
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in all_extensions:
                media_files.append(file_path)
        
        return sorted(media_files)
    
    def display_files(self, files):
        """显示找到的文件"""
        if not files:
            print("❌ 在 Use/Input/ 文件夹中没有找到支持的媒体文件")
            print(f"支持的音频格式: {', '.join(self.supported_audio)}")
            print(f"支持的视频格式: {', '.join(self.supported_video)}")
            return False
        
        print(f"\n📁 在 Use/Input/ 中找到 {len(files)} 个媒体文件:")
        print("-" * 60)
        
        audio_count = 0
        video_count = 0
        
        for i, file_path in enumerate(files, 1):
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            if is_audio_file(str(file_path)) or file_path.suffix.lower() in self.supported_audio:
                file_type = "🎵 音频"
                audio_count += 1
            else:
                file_type = "📹 视频"
                video_count += 1
            
            print(f"{i:2d}. {file_type} | {file_path.name} ({file_size:.2f} MB)")
        
        print("-" * 60)
        print(f"总计: {audio_count} 个音频文件, {video_count} 个视频文件")
        return True
    
    def select_model(self):
        """让用户选择模型"""
        print("\n🤖 请选择要使用的模型:")
        print("1. FireRedASR-AED (快速, 适合批量处理)")
        print("2. FireRedASR-LLM (高精度, 较慢)")
        
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
                return False
        
        # 检查模型路径
        if not Path(model_dir).exists():
            print(f"❌ 错误: 模型目录不存在: {model_dir}")
            print("请从 https://huggingface.co/fireredteam 下载模型文件")
            return False
        
        return model_dir
    
    def load_model(self, model_dir):
        """加载模型"""
        print(f"\n🔄 正在加载 {self.model_type.upper()} 模型...")
        start_time = time.time()
        
        try:
            self.model = FireRedAsr.from_pretrained(self.model_type, model_dir)
            load_time = time.time() - start_time
            print(f"✅ 模型加载成功 (耗时: {load_time:.2f}s)")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return False
    
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
    
    def transcribe_file(self, file_path):
        """转录单个文件"""
        print(f"\n🔄 处理: {file_path.name}")
        
        try:
            uttid = file_path.stem
            decode_config = self.get_decode_config()
            
            start_time = time.time()
            results = self.model.transcribe([uttid], [str(file_path)], decode_config)
            process_time = time.time() - start_time
            
            if results and len(results) > 0:
                result = results[0]
                text = result['text']
                rtf = float(result.get('rtf', 0))
                
                print(f"✅ 识别完成 (耗时: {process_time:.2f}s, RTF: {rtf:.4f})")
                print(f"📝 结果: {text}")
                
                return {
                    'file': file_path.name,
                    'text': text,
                    'duration': process_time,
                    'rtf': rtf,
                    'model': self.model_type,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                print(f"❌ 识别失败: 没有返回结果")
                return None
                
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            return None
    
    def save_results(self, all_results):
        """保存结果到输出文件夹"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存文本结果
        txt_file = self.output_dir / f"transcription_results_{timestamp}.txt"
        json_file = self.output_dir / f"transcription_results_{timestamp}.json"
        
        # 写入文本文件
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"FireRedASR 批量语音识别结果\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"使用模型: {self.model_type.upper()}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(all_results, 1):
                if result:
                    f.write(f"{i}. 文件: {result['file']}\n")
                    f.write(f"   识别结果: {result['text']}\n")
                    f.write(f"   处理时间: {result['duration']:.2f}s\n")
                    f.write(f"   RTF: {result['rtf']:.4f}\n")
                    f.write("-" * 40 + "\n")
        
        # 写入JSON文件
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model': self.model_type,
                    'total_files': len(all_results),
                    'successful': len([r for r in all_results if r is not None])
                },
                'results': all_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存:")
        print(f"📄 文本文件: {txt_file}")
        print(f"📄 JSON文件: {json_file}")
        
        # 标点恢复处理
        if self.enable_punctuation and all_results:
            try:
                print(f"\n🔤 开始标点恢复处理...")
                
                # 初始化标点恢复器（延迟加载）
                if self.punctuation_restorer is None:
                    self.punctuation_restorer = PunctuationRestorer(
                        cache_dir=self.punctuation_model_dir,
                        chunk_size=self.punctuation_chunk_size,
                        stride=self.punctuation_stride
                    )
                
                # 生成带标点的文本文件
                punctuated_txt_file = self.output_dir / f"transcription_results_{timestamp}_with_punctuation.txt"
                punctuated_json_file = self.output_dir / f"transcription_results_{timestamp}_with_punctuation.json"
                
                # 处理每个结果的标点恢复
                punctuated_results = []
                for result in all_results:
                    if result and result.get('text'):
                        punctuated_text = self.punctuation_restorer.restore_punctuation(result['text'])
                        punctuated_result = result.copy()
                        punctuated_result['text'] = punctuated_text
                        punctuated_result['original_text'] = result['text']
                        punctuated_results.append(punctuated_result)
                    else:
                        punctuated_results.append(result)
                
                # 写入带标点的文本文件
                with open(punctuated_txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"FireRedASR 批量语音识别结果（带标点）\n")
                    f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"使用模型: {self.model_type.upper()}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for i, result in enumerate(punctuated_results, 1):
                        if result:
                            f.write(f"{i}. 文件: {result['file']}\n")
                            f.write(f"   识别结果: {result['text']}\n")
                            f.write(f"   处理时间: {result['duration']:.2f}s\n")
                            f.write(f"   RTF: {result['rtf']:.4f}\n")
                            f.write("-" * 40 + "\n")
                
                # 写入带标点的JSON文件
                with open(punctuated_json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'metadata': {
                            'timestamp': datetime.now().isoformat(),
                            'model': self.model_type,
                            'total_files': len(punctuated_results),
                            'successful': len([r for r in punctuated_results if r is not None]),
                            'punctuation_enabled': True
                        },
                        'results': punctuated_results
                    }, f, ensure_ascii=False, indent=2)
                
                print(f"📄 带标点文本文件: {punctuated_txt_file}")
                print(f"📄 带标点JSON文件: {punctuated_json_file}")
                
                # 如果启用了分段功能，合并所有文本并分段
                if self.enable_paragraph and punctuated_results:
                    try:
                        print(f"\n📑 开始合并文本并进行自然段分段...")
                        
                        # 初始化分段器
                        if self.paragraph_segmenter is None:
                            self.paragraph_segmenter = ParagraphSegmenter(
                                min_length=self.min_paragraph_length,
                                max_length=self.max_paragraph_length
                            )
                        
                        # 合并所有识别结果的文本
                        merged_text = ""
                        for result in punctuated_results:
                            if result and result.get('text'):
                                merged_text += result['text'] + "。"
                        
                        # 执行分段
                        paragraphs = self.paragraph_segmenter.segment_paragraphs(merged_text)
                        
                        # 保存分段结果（优化的书籍排版格式）
                        paragraph_txt_file = self.output_dir / f"transcription_results_{timestamp}_paragraphs.txt"
                        with open(paragraph_txt_file, 'w', encoding='utf-8') as f:
                            f.write(f"FireRedASR 批量识别结果\n")
                            f.write(f"\n处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"使用模型: {self.model_type.upper()}\n")
                            f.write(f"文件数: {len(punctuated_results)}\n")
                            f.write(f"段落数: {len(paragraphs)}\n")
                            f.write("\n" + "=" * 60 + "\n\n")
                            
                            # 使用书籍排版格式
                            for i, para in enumerate(paragraphs, 1):
                                # 段首缩进4个空格
                                f.write(f"    {para}\n\n")
                        
                        # 同时生成 Markdown 格式
                        markdown_file = self.output_dir / f"transcription_results_{timestamp}_paragraphs.md"
                        with open(markdown_file, 'w', encoding='utf-8') as f:
                            # Markdown 头部
                            f.write(f"# 批量识别文稿\n\n")
                            f.write(f"**处理时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
                            f.write(f"**文件数量:** {len(punctuated_results)}  \n")
                            f.write(f"**段落数量:** {len(paragraphs)}  \n\n")
                            f.write("---\n\n")
                            
                            # 正文内容
                            for i, para in enumerate(paragraphs, 1):
                                f.write(f"{para}\n\n")
                        
                        print(f"📄 自然段格式文件: {paragraph_txt_file}")
                        print(f"   共分为 {len(paragraphs)} 个自然段")
                        
                    except Exception as e:
                        print(f"⚠️ 分段处理失败: {str(e)}")
                        print("   将保留带标点版本")
                
            except Exception as e:
                print(f"⚠️ 标点恢复失败: {str(e)}")
                print("   将保留无标点版本")
    
    def run(self):
        """运行批量转录"""
        print("🔥 FireRedASR 批量语音识别工具")
        print("=" * 60)
        
        # 1. 扫描输入文件
        files = self.scan_input_files()
        if not self.display_files(files):
            return
        
        # 2. 用户确认
        try:
            confirm = input(f"\n是否继续处理这 {len(files)} 个文件? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes', '是']:
                print("👋 用户取消操作")
                return
        except KeyboardInterrupt:
            print("\n\n👋 用户取消操作")
            return
        
        # 3. 选择模型
        model_dir = self.select_model()
        if not model_dir:
            return
        
        # 4. 加载模型
        if not self.load_model(model_dir):
            return
        
        # 5. 批量处理
        print(f"\n🚀 开始批量处理 {len(files)} 个文件...")
        print("=" * 60)
        
        all_results = []
        successful = 0
        
        try:
            for i, file_path in enumerate(files, 1):
                print(f"\n[{i}/{len(files)}]", end=" ")
                result = self.transcribe_file(file_path)
                all_results.append(result)
                
                if result:
                    successful += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断处理")
            print(f"已处理 {len(all_results)} 个文件")
        
        finally:
            # 清理临时文件
            if self.model:
                self.model.feat_extractor.cleanup_temp_files()
        
        # 6. 保存结果
        if all_results:
            self.save_results(all_results)
            
            print("\n" + "=" * 60)
            print(f"✅ 批量处理完成!")
            print(f"📊 总计: {len(all_results)} 个文件, 成功: {successful} 个")
            print(f"📁 结果保存在: {self.output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FireRedASR 批量语音识别工具")
    
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
                        help='启用自然段分段功能（将合并所有文本后分段）')
    parser.add_argument('--min-paragraph-length', type=int, default=50,
                        help='最小段落长度（默认: 50字）')
    parser.add_argument('--max-paragraph-length', type=int, default=500,
                        help='最大段落长度（默认: 500字）')
    
    args = parser.parse_args()
    
    # 检查是否在正确的目录
    if not Path("fireredasr").exists():
        print("❌ 错误: 请在 FireRedASR 项目根目录下运行此脚本")
        print("当前目录应该包含 fireredasr/ 文件夹")
        return
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(Path.cwd()) + ":" + os.environ.get('PYTHONPATH', '')
    
    try:
        transcriber = BatchTranscriber()
        
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
        transcriber.min_paragraph_length = args.min_paragraph_length
        transcriber.max_paragraph_length = args.max_paragraph_length
        
        transcriber.run()
    except Exception as e:
        print(f"❌ 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()