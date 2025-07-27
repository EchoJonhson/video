# FireRedASR 批量处理使用指南

## 🚀 快速开始

### 1. 准备文件
将需要转录的音频或视频文件放入 `Use/Input/` 文件夹中：

```bash
Use/
├── Input/           # 放入您的音频/视频文件
│   ├── audio1.wav
│   ├── video1.mp4
│   └── ...
└── Output/          # 结果将保存在这里
```

### 2. 运行批量处理脚本
```bash
python batch_transcribe.py
```

### 3. 选择模型
脚本会提示您选择模型：
- **选项 1**: FireRedASR-AED (快速，适合批量处理)
- **选项 2**: FireRedASR-LLM (高精度，较慢)

### 4. 查看结果
处理完成后，结果将保存在 `Use/Output/` 文件夹中：
- `transcription_results_YYYYMMDD_HHMMSS.txt` - 人类可读的文本格式
- `transcription_results_YYYYMMDD_HHMMSS.json` - 机器可读的JSON格式

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

## 🔧 高级用法

### 单个文件处理
如果只需要处理单个文件，也可以使用命令行工具：

```bash
# 处理音频文件
python fireredasr/speech2text.py --wav_path audio.wav --asr_type aed --model_dir pretrained_models/FireRedASR-AED-L

# 处理视频文件
python fireredasr/speech2text.py --video_path video.mp4 --asr_type aed --model_dir pretrained_models/FireRedASR-AED-L

# 使用通用输入参数
python fireredasr/speech2text.py --input_path media_file --asr_type llm --model_dir pretrained_models/FireRedASR-LLM-L
```

### 测试模型
使用测试脚本验证模型功能：
```bash
# 测试音频文件
python test_models.py --input audio.wav --model aed

# 测试视频文件  
python test_models.py --input video.mp4 --model both
```

## ⚠️ 注意事项

1. **模型文件**: 确保已下载模型文件到 `pretrained_models/` 目录
2. **依赖安装**: 运行前确保已安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. **文件大小**: 大文件处理可能需要更长时间
4. **临时文件**: 视频文件会自动提取音频，处理完成后临时文件会自动清理
5. **中断处理**: 可以使用 Ctrl+C 中断处理，已处理的结果会保存

## 🎯 性能指标说明

- **RTF (Real Time Factor)**: 实时因子，表示处理时间与音频时长的比值
  - RTF < 1: 处理速度快于实时
  - RTF = 1: 处理速度等于实时  
  - RTF > 1: 处理速度慢于实时

## 📞 问题排查

### 常见错误

1. **模型目录不存在**
   ```
   ❌ 错误: 模型目录不存在: pretrained_models/FireRedASR-AED-L
   ```
   解决：从 https://huggingface.co/fireredteam 下载模型文件

2. **没有找到输入文件**
   ```
   ❌ 在 Use/Input/ 文件夹中没有找到支持的媒体文件
   ```
   解决：检查文件格式是否支持，确保文件在正确目录

3. **依赖缺失**
   ```
   ModuleNotFoundError: No module named 'moviepy'
   ```
   解决：安装依赖 `pip install moviepy librosa soundfile`

### 获取帮助
如遇到其他问题，请检查：
1. Python环境是否正确
2. 所有依赖是否已安装
3. 模型文件是否下载完整
4. 输入文件是否损坏