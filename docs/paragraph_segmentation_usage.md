# FireRedASR 中文自然段分段功能使用文档

## 功能概述

FireRedASR 新增的自然段分段功能可以将长音频/视频转写后的文本自动分割成结构清晰的自然段落，提高文本的可读性。该功能基于规则的分段算法，无需外部依赖，100%离线可用。

## 主要特性

- 🚀 **零外部依赖**：纯规则算法，无需下载额外模型
- 🎯 **智能分段**：基于标点符号、话题转换词和长度控制的智能分段
- 🔧 **灵活配置**：支持自定义最小/最大段落长度
- 📦 **全面集成**：支持音频、视频和批量处理

## 使用方法

### 1. 长音频转写分段

```bash
# 基本用法
python long_audio_transcribe.py --input_audio audio.wav --model_dir path/to/model --enable-paragraph

# 自定义参数
python long_audio_transcribe.py \
    --input_audio audio.wav \
    --model_dir path/to/model \
    --enable-paragraph \
    --min-paragraph-length 50 \
    --max-paragraph-length 500
```

### 2. 长视频转写分段

```bash
# 基本用法
python long_video_transcribe.py --input_video video.mp4 --model_dir path/to/model --enable-paragraph

# 自定义参数
python long_video_transcribe.py \
    --input_video video.mp4 \
    --model_dir path/to/model \
    --enable-paragraph \
    --min-paragraph-length 50 \
    --max-paragraph-length 500
```

### 3. 批量处理分段

```bash
# 批量处理目录中的所有音频文件
python batch_transcribe.py \
    --input_dir /path/to/audio/files \
    --model_dir path/to/model \
    --enable-paragraph \
    --min-paragraph-length 50 \
    --max-paragraph-length 500
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable-paragraph` | 启用自然段分段功能 | 关闭 |
| `--paragraph-method` | 分段方法（目前仅支持rule） | rule |
| `--min-paragraph-length` | 最小段落长度（字符数） | 50 |
| `--max-paragraph-length` | 最大段落长度（字符数） | 500 |

## 分段算法原理

### 规则分段策略

1. **强分界点检测**：句号（。）、问号（？）、感叹号（！）
2. **话题转换词识别**：
   - 顺序词：首先、其次、最后、然后、接下来
   - 转折词：但是、然而、不过
   - 补充词：另外、同时、此外
   - 总结词：总之

3. **长度控制**：
   - 段落长度达到最小值后，遇到话题转换词即分段
   - 段落长度超过最大值时强制分段

## 输出格式

### 标准输出文件

启用分段功能后，会生成多种格式的输出文件：

1. **纯文本文件**（连续段落，无时间戳）
   - 音频：`output_dir/full_transcript.txt`
   - 视频：`output_dir/video_name.txt`
   - 特点：连续文本，适合阅读

2. **带时间戳文本**（保留时间信息）
   - 音频：`output_dir/full_transcript_with_timestamps.txt`
   - 视频：`output_dir/video_name_时间戳.txt`
   - 特点：保留原始时间轴信息

3. **带标点文本**（标点恢复后）
   - 音频：`output_dir/full_transcript_with_punctuation.txt`
   - 视频：`output_dir/video_name_标点.txt`
   - 特点：添加标点符号，提高可读性

4. **自然段格式**（书籍排版）
   - 音频：`output_dir/full_transcript_paragraphs.txt`
   - 视频：`output_dir/video_name_段落.txt`
   - 批量：`output_dir/transcription_results_时间戳_paragraphs.txt`
   - 特点：段首缩进，自然段落划分

5. **Markdown格式**（精美排版）
   - 音频：`output_dir/full_transcript_paragraphs.md`
   - 视频：`output_dir/video_name_段落.md`
   - 批量：`output_dir/transcription_results_时间戳_paragraphs.md`
   - 特点：支持富文本渲染，适合分享

### 输出示例

#### 书籍格式（.txt）
```
FireRedASR 视频转写结果

文件: example_video
处理时间: 2025-07-29 10:30:00
总时长: 00:15:30
段落数: 4

============================================================

    首先，让我们来讨论一下今天的会议主题。这个问题非常重要，需要我们认真对待。在过去的几年里，我们看到了很多变化。技术的发展速度越来越快，市场环境也在不断变化。

    另外，我们还需要考虑客户的需求。客户的需求是多样化的，我们必须提供个性化的解决方案。这就要求我们不断创新，不断改进我们的产品和服务。

    然后，关于具体的实施方案，我建议分三个阶段进行。第一阶段是调研和规划，第二阶段是开发和测试，第三阶段是部署和优化。

    最后，我想强调的是团队合作的重要性。只有大家齐心协力，才能确保项目的成功。
```

#### Markdown格式（.md）
```markdown
# example_video - 转写文稿

**处理时间:** 2025-07-29 10:30:00  
**视频时长:** 00:15:30  
**段落数量:** 4  

---

首先，让我们来讨论一下今天的会议主题。这个问题非常重要，需要我们认真对待。在过去的几年里，我们看到了很多变化。技术的发展速度越来越快，市场环境也在不断变化。

另外，我们还需要考虑客户的需求。客户的需求是多样化的，我们必须提供个性化的解决方案。这就要求我们不断创新，不断改进我们的产品和服务。

然后，关于具体的实施方案，我建议分三个阶段进行。第一阶段是调研和规划，第二阶段是开发和测试，第三阶段是部署和优化。

最后，我想强调的是团队合作的重要性。只有大家齐心协力，才能确保项目的成功。
```

## 最佳实践

1. **参数调优建议**：
   - 会议记录：`--min-paragraph-length 80 --max-paragraph-length 300`
   - 演讲稿：`--min-paragraph-length 100 --max-paragraph-length 400`
   - 对话转写：`--min-paragraph-length 50 --max-paragraph-length 200`

2. **使用场景**：
   - 长会议记录的结构化整理
   - 演讲内容的段落划分
   - 采访记录的主题分段
   - 教学视频的章节整理

3. **注意事项**：
   - 分段功能依赖于标点恢复，建议同时启用 `--enable-punctuation`
   - 分段结果会受到转写准确度的影响
   - 可根据实际需求调整段落长度参数

## 测试功能

运行测试脚本验证功能：

```bash
python test_paragraph_segmentation.py
```

测试脚本会：
- 测试分段模块的基本功能
- 展示不同参数的分段效果
- 验证与转写工具的集成

## 未来计划

- [ ] 添加语义分段算法（基于句子相似度）
- [ ] 支持自定义话题转换词库
- [ ] 添加段落标题自动生成
- [ ] 支持更多输出格式（Markdown、JSON等）

## 常见问题

**Q: 分段功能会影响转写速度吗？**
A: 分段算法非常轻量，对整体转写速度的影响可以忽略不计。

**Q: 可以在没有标点的文本上使用分段吗？**
A: 不建议。分段算法依赖标点符号判断句子边界，建议先启用标点恢复功能。

**Q: 如何处理过短或过长的段落？**
A: 调整 `--min-paragraph-length` 和 `--max-paragraph-length` 参数，找到适合您内容的最佳值。

## 技术支持

如有问题或建议，请通过 GitHub Issues 反馈。