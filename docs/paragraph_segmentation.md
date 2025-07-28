# FireRedASR 中文自然段分段功能使用指南

## 功能概述

FireRedASR 新增了中文自然段分段功能，可以将长音频/视频转写的结果自动分割成符合阅读习惯的自然段落。

## 使用方法

### 1. 长音频转写中使用

```bash
# 基础使用 - 启用分段功能
python long_audio_transcribe.py \
    --input_audio your_audio.mp3 \
    --model_type aed \
    --enable-paragraph

# 自定义段落长度范围
python long_audio_transcribe.py \
    --input_audio your_audio.mp3 \
    --model_type aed \
    --enable-paragraph \
    --min-paragraph-length 100 \
    --max-paragraph-length 300
```

### 2. 长视频转写中使用

```bash
# 启用分段功能
python long_video_transcribe.py \
    --enable-paragraph

# 使用LLM模型并启用分段
python long_video_transcribe.py \
    --model_type llm \
    --enable-paragraph
```

### 3. 批量转写中使用

```bash
# 批量转写并合并分段
python batch_transcribe.py \
    --enable-paragraph
```

## 参数说明

- `--enable-paragraph`: 启用自然段分段功能
- `--min-paragraph-length`: 最小段落长度（默认50字）
- `--max-paragraph-length`: 最大段落长度（默认500字）

## 输出文件

启用分段功能后，会额外生成以下文件：

- **长音频转写**: `full_transcript_paragraphs.txt`
- **长视频转写**: `[视频名]_段落.txt`
- **批量转写**: `transcription_results_[时间戳]_paragraphs.txt`

## 分段算法说明

当前版本使用基于规则的分段算法：

1. **强分界点**: 句号、问号、感叹号
2. **话题转换词**: "另外"、"然后"、"接下来"、"首先"等
3. **长度控制**: 确保段落不会过短或过长

## 示例效果

原始文本：
```
大家好欢迎来到今天的会议今天我们要讨论三个重要议题首先是关于公司的战略调整...
```

分段后：
```
【第1段】
大家好，欢迎来到今天的会议。今天我们要讨论三个重要议题。首先是关于公司的战略调整。

【第2段】
另外，市场环境也在不断变化。我们需要适应这些变化才能生存下去。然后，我们来看看具体的解决方案。

【第3段】
最后，执行力是关键。没有执行，再好的计划也是空谈。
```

## 注意事项

1. 分段功能依赖于标点恢复，确保标点恢复功能正常工作
2. 分段结果的质量取决于转写和标点恢复的准确性
3. 对于短音频，分段可能效果不明显

## 后续优化计划

- 增加基于语义的分段算法
- 支持自定义话题转换词库
- 提供分段质量评估指标