#!/bin/bash

# FireRedASR 视频文件处理示例脚本
# 本脚本演示如何使用FireRedASR处理视频文件

echo "🔥 FireRedASR 视频处理示例"
echo "=========================="

# 配置变量
MODEL_AED="pretrained_models/FireRedASR-AED-L"
MODEL_LLM="pretrained_models/FireRedASR-LLM-L"

# 检查模型是否存在
check_model() {
    if [ ! -d "$1" ]; then
        echo "❌ 错误: 模型目录不存在: $1"
        echo "请从 https://huggingface.co/fireredteam 下载模型文件"
        exit 1
    fi
}

echo "检查模型文件..."
check_model "$MODEL_AED"
check_model "$MODEL_LLM"
echo "✅ 模型文件检查完成"

# 示例视频文件（如果存在的话）
VIDEO_DIR="examples/video"
SAMPLE_VIDEOS=(
    "sample.mp4"
    "demo.avi"
    "test.mov"
)

echo ""
echo "🔍 搜索示例视频文件..."

# 查找实际存在的视频文件
FOUND_VIDEO=""
for video in "${SAMPLE_VIDEOS[@]}"; do
    if [ -f "$VIDEO_DIR/$video" ]; then
        FOUND_VIDEO="$VIDEO_DIR/$video"
        echo "✅ 找到视频文件: $FOUND_VIDEO"
        break
    fi
done

# 如果没找到示例视频，创建一个测试用的说明
if [ -z "$FOUND_VIDEO" ]; then
    echo "⚠️  未找到示例视频文件"
    echo "请将您的视频文件放置到以下位置之一："
    for video in "${SAMPLE_VIDEOS[@]}"; do
        echo "  - $VIDEO_DIR/$video"
    done
    echo ""
    echo "支持的视频格式: MP4, AVI, MOV, MKV, FLV, WMV"
    echo ""
    echo "示例用法:"
    echo "--------"
    
    # 使用音频文件作为示例
    AUDIO_FILE="examples/wav/BAC009S0764W0121.wav"
    if [ -f "$AUDIO_FILE" ]; then
        echo "# 处理音频文件（向后兼容）"
        echo "speech2text.py --asr_type aed --model_dir $MODEL_AED --wav_path $AUDIO_FILE"
        echo ""
        echo "# 使用新的通用输入参数"
        echo "speech2text.py --asr_type aed --model_dir $MODEL_AED --input_path $AUDIO_FILE"
        echo ""
        echo "# 处理视频文件（如果有的话）"
        echo "speech2text.py --asr_type aed --model_dir $MODEL_AED --video_path your_video.mp4"
        echo "speech2text.py --asr_type llm --model_dir $MODEL_LLM --input_path your_video.mp4"
        
        echo ""
        echo "现在使用音频文件进行演示..."
        FOUND_VIDEO="$AUDIO_FILE"
    else
        echo "❌ 连音频示例文件都不存在，无法进行演示"
        exit 1
    fi
fi

echo ""
echo "🚀 开始处理: $FOUND_VIDEO"
echo "=========================="

# 测试 FireRedASR-AED
echo ""
echo "📝 测试 FireRedASR-AED 模型"
echo "----------------------------"

if [[ "$FOUND_VIDEO" == *.wav ]] || [[ "$FOUND_VIDEO" == *.mp3 ]]; then
    # 音频文件，使用原有参数
    python fireredasr/speech2text.py \
        --asr_type aed \
        --model_dir "$MODEL_AED" \
        --wav_path "$FOUND_VIDEO" \
        --use_gpu 1 \
        --beam_size 3 \
        --nbest 1 \
        --decode_max_len 0 \
        --softmax_smoothing 1.25 \
        --aed_length_penalty 0.6 \
        --eos_penalty 1.0
else
    # 视频文件，使用新参数
    python fireredasr/speech2text.py \
        --asr_type aed \
        --model_dir "$MODEL_AED" \
        --input_path "$FOUND_VIDEO" \
        --use_gpu 1 \
        --beam_size 3 \
        --nbest 1 \
        --decode_max_len 0 \
        --softmax_smoothing 1.25 \
        --aed_length_penalty 0.6 \
        --eos_penalty 1.0
fi

echo ""
echo "📝 测试 FireRedASR-LLM 模型"
echo "----------------------------"

if [[ "$FOUND_VIDEO" == *.wav ]] || [[ "$FOUND_VIDEO" == *.mp3 ]]; then
    # 音频文件，使用原有参数
    python fireredasr/speech2text.py \
        --asr_type llm \
        --model_dir "$MODEL_LLM" \
        --wav_path "$FOUND_VIDEO" \
        --use_gpu 1 \
        --beam_size 3 \
        --decode_max_len 0 \
        --decode_min_len 0 \
        --repetition_penalty 3.0 \
        --llm_length_penalty 1.0 \
        --temperature 1.0
else
    # 视频文件，使用新参数
    python fireredasr/speech2text.py \
        --asr_type llm \
        --model_dir "$MODEL_LLM" \
        --input_path "$FOUND_VIDEO" \
        --use_gpu 1 \
        --beam_size 3 \
        --decode_max_len 0 \
        --decode_min_len 0 \
        --repetition_penalty 3.0 \
        --llm_length_penalty 1.0 \
        --temperature 1.0
fi

echo ""
echo "✅ 视频处理演示完成！"
echo ""
echo "💡 更多用法:"
echo "------------"
echo "# 批量处理视频目录"
echo "speech2text.py --asr_type aed --model_dir $MODEL_AED --video_dir path/to/videos/"
echo ""
echo "# 混合处理音频和视频"
echo "speech2text.py --asr_type aed --model_dir $MODEL_AED --input_dir path/to/media/"
echo ""
echo "# 处理多个文件"
echo "speech2text.py --asr_type aed --model_dir $MODEL_AED --input_paths video1.mp4 audio1.wav video2.avi"