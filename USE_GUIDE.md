# FireRedASR 智能批量处理使用指南

## 🚀 快速开始

### 1. 准备文件
将需要转录的音频或视频文件放入 `Use/Input/` 文件夹中：

```bash
Use/
├── Input/           # 放入您的音频/视频文件
│   ├── audio1.wav
│   ├── video1.mp4
│   ├── lecture.mp4  # 长视频也支持！
│   └── ...
└── Output/          # 结果将保存在这里
    ├── temp_long_video/  # 长视频处理临时文件
    └── ...
```

### 2. 选择处理方式

#### 🎯 方式一：智能批量处理（推荐）
```bash
python batch_transcribe.py
```
**特点**：自动检测文件类型，智能选择处理策略

#### ⚡ 方式二：长视频专用处理
```bash
python long_video_transcribe.py
```
**特点**：专为长音频/视频优化，支持VAD切分和并行处理

### 3. 模型选择智能建议
系统会根据您的文件特点智能推荐：

| 文件特征 | 推荐模型 | 优势 |
|----------|----------|------|
| 短音频 (< 30秒) | **LLM** | 最高准确率 |
| 长音频 (> 60秒) | **AED** | 稳定性好，速度快 |
| 批量小文件 | **AED** | 处理效率高 |
| 高质量要求 | **LLM** | 语言理解优秀 |

### 4. 查看丰富的处理结果
处理完成后，结果保存在 `Use/Output/` 文件夹：

**常规批量处理结果：**
- `transcription_results_YYYYMMDD_HHMMSS.txt` - 易读文本格式
- `transcription_results_YYYYMMDD_HHMMSS.json` - 结构化JSON数据

**长视频处理结果：**
- `filename_transcription_YYYYMMDD_HHMMSS.txt` - 完整文字稿
- `filename_transcription_YYYYMMDD_HHMMSS.srt` - SRT字幕文件
- `filename_transcription_YYYYMMDD_HHMMSS_with_timestamps.txt` - 带时间戳文本
- `filename_transcription_YYYYMMDD_HHMMSS_stats.json` - 处理统计信息

## 📁 支持的文件格式

### 音频格式
- WAV, MP3, FLAC, M4A, AAC, OGG

### 视频格式  
- MP4, AVI, MOV, MKV, FLV, WMV

## 📊 输出格式说明

### 文本文件 (.txt)
```
FireRedASR 批量语音识别结果
处理时间: 2025-07-27 23:03:38
使用模型: AED
============================================================

1. 文件: test.mp4
   识别结果: 你好你好你能听见我的声音吗
   处理时间: 1.65s
   RTF: 0.0964
----------------------------------------
```

### JSON文件 (.json)
```json
{
  "metadata": {
    "timestamp": "2025-07-27T23:03:38.723036",
    "model": "aed",
    "total_files": 1,
    "successful": 1
  },
  "results": [
    {
      "file": "test.mp4",
      "text": "你好你好你能听见我的声音吗",
      "duration": 1.65,
      "rtf": 0.0964,
      "model": "aed",
      "timestamp": "2025-07-27T23:03:38.722826"
    }
  ]
}
```

## 🔧 高级用法与性能优化

### 长视频处理高级参数
```bash
# 自定义VAD参数，适合不同场景
python long_video_transcribe.py \
    --model_type aed \
    --max_duration 45 \          # 最大音频段长度（秒）
    --min_silence 300 \          # 最小静音间隔（毫秒）
    --vad_threshold 0.5          # VAD检测阈值

# 课堂录制优化（长句子）
python long_video_transcribe.py --max_duration 60 --min_silence 800

# 对话录音优化（短对话）
python long_video_transcribe.py --max_duration 20 --min_silence 200
```

### 单个文件处理（兼容老版本）
```bash
# 处理音频文件
python fireredasr/speech2text.py --wav_path audio.wav --asr_type aed --model_dir pretrained_models/FireRedASR-AED-L

# 处理视频文件
python fireredasr/speech2text.py --video_path video.mp4 --asr_type aed --model_dir pretrained_models/FireRedASR-AED-L

# 使用通用输入参数
python fireredasr/speech2text.py --input_path media_file --asr_type llm --model_dir pretrained_models/FireRedASR-LLM-L
```

### 硬件优化建议

#### GPU配置
```bash
# 检查CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 指定GPU设备
export CUDA_VISIBLE_DEVICES=0
python batch_transcribe.py
```

#### 内存优化
```bash
# 低内存模式（减小批处理大小）
python long_video_transcribe.py --batch_size 1

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

## ⚠️ 注意事项与最佳实践

### 基本要求
1. **模型文件**: 确保已下载模型文件到 `pretrained_models/` 目录
2. **依赖安装**: 运行前确保已安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. **硬件要求**: 
   - GPU内存 ≥ 8GB（LLM模型）
   - 系统内存 ≥ 16GB（长视频处理）
   - 存储空间预留原文件3-5倍大小

### 处理建议
4. **文件准备**: 音频预处理为16kHz单声道可提高效果
5. **长视频**: 超过30分钟建议使用 `long_video_transcribe.py`
6. **批量处理**: 文件数量多时建议使用GPU加速
7. **中断恢复**: 支持Ctrl+C中断，已处理结果自动保存

### 临时文件管理
- 视频文件自动提取音频到临时目录
- 长视频切片文件可选择保留或自动清理
- 处理完成后系统会自动清理临时文件

## 🎯 性能指标详解

### RTF (Real Time Factor) 实时因子
- **RTF < 0.5**: 处理极快，适合实时应用
- **RTF = 1**: 处理时间等于音频时长
- **RTF > 2**: 处理较慢，建议优化硬件配置

### 典型性能表现
| 文件类型 | 模型 | RTF | 准确率 |
|----------|------|-----|--------|
| 短音频 | LLM | 0.3-0.5 | 96%+ |
| 短音频 | AED | 0.1-0.2 | 92%+ |
| 长视频 | AED | 0.2-0.4 | 90%+ |

## 🔧 故障排查与解决方案

### 常见错误与解决方案

#### 1. 模型相关错误
```bash
❌ 错误: 模型目录不存在: pretrained_models/FireRedASR-AED-L
```
**解决**：
- 从 [HuggingFace](https://huggingface.co/fireredteam) 下载模型
- 检查目录结构是否正确
- 确认模型文件完整性

#### 2. 输入文件问题
```bash
❌ 在 Use/Input/ 文件夹中没有找到支持的媒体文件
```
**解决**：
- 检查文件扩展名是否支持
- 确认文件未损坏
- 验证文件路径正确

#### 3. 依赖环境问题
```bash
ModuleNotFoundError: No module named 'torch'
```
**解决**：
```bash
# 安装PyTorch（CUDA版本）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

#### 4. 内存不足问题
```bash
CUDA out of memory
```
**解决**：
- 使用CPU模式：`export CUDA_VISIBLE_DEVICES=""`
- 减少批处理大小：`--batch_size 1`
- 清理GPU缓存：`torch.cuda.empty_cache()`

#### 5. 长视频处理超时
**解决**：
- 检查VAD参数设置
- 分段处理：使用更小的 `max_duration`
- 增加处理超时时间

### 性能优化建议
1. **预处理优化**: 使用ffmpeg预处理音频格式
2. **硬件配置**: 优先使用GPU，确保驱动更新
3. **参数调优**: 根据音频特点调整VAD参数
4. **存储优化**: 使用SSD存储提高IO性能

### 获取技术支持
遇到复杂问题时：
1. 查看详细错误日志
2. 检查系统硬件配置
3. 验证音频文件格式和质量
4. 参考项目GitHub Issues页面