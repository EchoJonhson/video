#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from pathlib import Path

from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.utils.video_audio import is_video_file, is_audio_file


parser = argparse.ArgumentParser()
parser.add_argument('--asr_type', type=str, required=True, choices=["aed", "llm"])
parser.add_argument('--model_dir', type=str, required=True)

# Input / Output
parser.add_argument("--wav_path", type=str, help="单个音频文件路径")
parser.add_argument("--wav_paths", type=str, nargs="*", help="多个音频文件路径")
parser.add_argument("--wav_dir", type=str, help="音频文件目录")
parser.add_argument("--wav_scp", type=str, help="音频文件列表文件")
# 视频支持
parser.add_argument("--video_path", type=str, help="单个视频文件路径")
parser.add_argument("--video_paths", type=str, nargs="*", help="多个视频文件路径")
parser.add_argument("--video_dir", type=str, help="视频文件目录")
# 混合输入
parser.add_argument("--input_path", type=str, help="单个音频或视频文件路径")
parser.add_argument("--input_paths", type=str, nargs="*", help="多个音频或视频文件路径")
parser.add_argument("--input_dir", type=str, help="包含音频和视频文件的目录")
parser.add_argument("--output", type=str, help="输出文件路径")

# Decode Options
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--beam_size", type=int, default=1)
parser.add_argument("--decode_max_len", type=int, default=0)
# FireRedASR-AED
parser.add_argument("--nbest", type=int, default=1)
parser.add_argument("--softmax_smoothing", type=float, default=1.0)
parser.add_argument("--aed_length_penalty", type=float, default=0.0)
parser.add_argument("--eos_penalty", type=float, default=1.0)
# FireRedASR-LLM
parser.add_argument("--decode_min_len", type=int, default=0)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--llm_length_penalty", type=float, default=0.0)
parser.add_argument("--temperature", type=float, default=1.0)


def main(args):
    input_files = get_input_info(args)
    fout = open(args.output, "w") if args.output else None

    model = FireRedAsr.from_pretrained(args.asr_type, args.model_dir)
    
    try:
        batch_uttid = []
        batch_input_path = []
        for i, input_file in enumerate(input_files):
            uttid, file_path = input_file
            batch_uttid.append(uttid)
            batch_input_path.append(file_path)
            if len(batch_input_path) < args.batch_size and i != len(input_files) - 1:
                continue

            results = model.transcribe(
                batch_uttid,
                batch_input_path,
                {
                "use_gpu": args.use_gpu,
                "beam_size": args.beam_size,
                "nbest": args.nbest,
                "decode_max_len": args.decode_max_len,
                "softmax_smoothing": args.softmax_smoothing,
                "aed_length_penalty": args.aed_length_penalty,
                "eos_penalty": args.eos_penalty,
                "decode_min_len": args.decode_min_len,
                "repetition_penalty": args.repetition_penalty,
                "llm_length_penalty": args.llm_length_penalty,
                "temperature": args.temperature
                }
            )

            for result in results:
                print(result)
                if fout is not None:
                    fout.write(f"{result['uttid']}\t{result['text']}\n")

            batch_uttid = []
            batch_input_path = []
    
    finally:
        # 清理临时文件
        model.feat_extractor.cleanup_temp_files()
        if fout is not None:
            fout.close()


def get_input_info(args):
    """
    获取输入文件信息，支持音频和视频文件
    
    Returns:
        files: list of (uttid, file_path)
    """
    def get_uttid(path):
        """从文件路径生成utterance ID"""
        stem = Path(path).stem
        return stem
    
    files = []
    
    # 处理单个文件输入
    if args.wav_path:
        files = [(get_uttid(args.wav_path), args.wav_path)]
    elif args.video_path:
        files = [(get_uttid(args.video_path), args.video_path)]
    elif args.input_path:
        files = [(get_uttid(args.input_path), args.input_path)]
    
    # 处理多个文件输入
    elif args.wav_paths and len(args.wav_paths) >= 1:
        files = [(get_uttid(p), p) for p in sorted(args.wav_paths)]
    elif args.video_paths and len(args.video_paths) >= 1:
        files = [(get_uttid(p), p) for p in sorted(args.video_paths)]
    elif args.input_paths and len(args.input_paths) >= 1:
        files = [(get_uttid(p), p) for p in sorted(args.input_paths)]
    
    # 处理脚本文件输入
    elif args.wav_scp:
        files = [line.strip().split() for line in open(args.wav_scp)]
    
    # 处理目录输入
    elif args.wav_dir:
        wav_files = glob.glob(f"{args.wav_dir}/**/*.wav", recursive=True)
        files = [(get_uttid(p), p) for p in sorted(wav_files)]
    elif args.video_dir:
        video_patterns = ['**/*.mp4', '**/*.avi', '**/*.mov', '**/*.mkv', '**/*.flv', '**/*.wmv']
        video_files = []
        for pattern in video_patterns:
            video_files.extend(glob.glob(f"{args.video_dir}/{pattern}", recursive=True))
        files = [(get_uttid(p), p) for p in sorted(video_files)]
    elif args.input_dir:
        # 支持音频和视频混合
        media_patterns = [
            '**/*.wav', '**/*.mp3', '**/*.flac', '**/*.m4a', '**/*.aac',  # 音频
            '**/*.mp4', '**/*.avi', '**/*.mov', '**/*.mkv', '**/*.flv', '**/*.wmv'  # 视频
        ]
        media_files = []
        for pattern in media_patterns:
            media_files.extend(glob.glob(f"{args.input_dir}/{pattern}", recursive=True))
        files = [(get_uttid(p), p) for p in sorted(media_files)]
    
    else:
        raise ValueError("请提供有效的输入文件信息 (音频、视频或混合)")
    
    print(f"#输入文件数量={len(files)}")
    
    # 打印文件类型统计
    audio_count = sum(1 for _, path in files if is_audio_file(path) or path.endswith('.wav'))
    video_count = sum(1 for _, path in files if is_video_file(path))
    print(f"  音频文件: {audio_count}")
    print(f"  视频文件: {video_count}")
    
    return files


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
