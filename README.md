<div align="center">
<h1>FireRedASR: 开源工业级
<br>
自动语音识别模型</h1>

</div>

[[论文]](https://arxiv.org/pdf/2501.14350)
[[模型]](https://huggingface.co/fireredteam)
[[博客]](https://fireredteam.github.io/demos/firered_asr/)

FireRedASR 是一系列开源工业级自动语音识别（ASR）模型，支持普通话、中国方言和英语，在公开的普通话 ASR 基准测试中达到了新的最佳水平（SOTA），同时还提供了出色的歌词识别能力。


## 🔥 最新消息
- [2025/02/17] 我们发布了 [FireRedASR-LLM-L](https://huggingface.co/fireredteam/FireRedASR-LLM-L/tree/main) 模型权重。
- [2025/01/24] 我们发布了 [技术报告](https://arxiv.org/pdf/2501.14350)、[博客](https://fireredteam.github.io/demos/firered_asr/) 和 [FireRedASR-AED-L](https://huggingface.co/fireredteam/FireRedASR-AED-L/tree/main) 模型权重。


## 方法

FireRedASR 旨在满足各种应用场景中对卓越性能和最优效率的多样化需求。它包含两个变体：
- FireRedASR-LLM：旨在实现最先进（SOTA）的性能，并支持无缝的端到端语音交互。它采用编码器-适配器-LLM框架，充分利用大语言模型（LLM）的能力。
- FireRedASR-AED：旨在平衡高性能和计算效率，并作为基于LLM的语音模型中的有效语音表示模块。它使用基于注意力的编码器-解码器（AED）架构。

![Model](/assets/FireRedASR_model.png)


## 评估
结果以中文的字符错误率（CER%）和英文的词错误率（WER%）报告。

### 公开普通话ASR基准测试评估
| Model            | #Params | aishell1 | aishell2 | ws\_net  | ws\_meeting | Average-4 |
|:----------------:|:-------:|:--------:|:--------:|:--------:|:-----------:|:---------:|
| FireRedASR-LLM   | 8.3B | 0.76 | 2.15 | 4.60 | 4.67 | 3.05 |
| FireRedASR-AED   | 1.1B | 0.55 | 2.52 | 4.88 | 4.76 | 3.18 |
| Seed-ASR         | 12B+ | 0.68 | 2.27 | 4.66 | 5.69 | 3.33 |
| Qwen-Audio       | 8.4B | 1.30 | 3.10 | 9.50 | 10.87 | 6.19 |
| SenseVoice-L     | 1.6B | 2.09 | 3.04 | 6.01 | 6.73 | 4.47 |
| Whisper-Large-v3 | 1.6B | 5.14 | 4.96 | 10.48 | 18.87 | 9.86 |
| Paraformer-Large | 0.2B | 1.68 | 2.85 | 6.74 | 6.97 | 4.56 |

`ws` means WenetSpeech.

### 公开中国方言和英语ASR基准测试评估
|Test Set       | KeSpeech | LibriSpeech test-clean | LibriSpeech test-other  |
| :------------:| :------: | :--------------------: | :----------------------:|
|FireRedASR-LLM | 3.56 | 1.73 | 3.67 |
|FireRedASR-AED | 4.48 | 1.93 | 4.44 |
|Previous SOTA Results | 6.70 | 1.82 | 3.50 |


## 使用方法

**注意：由于模型文件过大，本仓库不包含预训练模型。请按照以下步骤下载模型文件：**

从 [huggingface](https://huggingface.co/fireredteam) 下载模型文件并将其放置在 `pretrained_models` 文件夹中。

如果您想使用 `FireRedASR-LLM-L`，还需要下载 [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) 并将其放置在 `pretrained_models` 文件夹中。然后进入 `FireRedASR-LLM-L` 文件夹并运行 `$ ln -s ../Qwen2-7B-Instruct`


### 环境设置
创建 Python 环境并安装依赖
```bash
$ git clone https://github.com/EchoJonhson/video.git
$ cd video
$ conda create --name fireredasr python=3.10
$ conda activate fireredasr
$ pip install -r requirements.txt
```

设置 Linux PATH 和 PYTHONPATH
```bash
$ export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
$ export PYTHONPATH=$PWD/:$PYTHONPATH
```

### 格式转换

**音频格式转换** - 将音频转换为 16kHz 16-bit PCM 格式
```bash
ffmpeg -i input_audio -ar 16000 -ac 1 -acodec pcm_s16le -f wav output.wav
```

**从视频提取音频** - 从视频文件提取音频（可选，FireRedASR可自动处理）
```bash
ffmpeg -i input_video.mp4 -ar 16000 -ac 1 -acodec pcm_s16le -f wav output.wav
```

### 快速开始

**处理音频文件**
```bash
$ cd examples
$ bash inference_fireredasr_aed.sh
$ bash inference_fireredasr_llm.sh
```

**处理视频文件**
```bash
$ cd examples
$ bash inference_video.sh                    # Shell脚本示例
$ python video_processing_example.py         # Python示例
```

### 命令行使用

**音频文件处理（向后兼容）**
```bash
$ speech2text.py --help
$ speech2text.py --wav_path examples/wav/BAC009S0764W0121.wav --asr_type "aed" --model_dir pretrained_models/FireRedASR-AED-L
$ speech2text.py --wav_path examples/wav/BAC009S0764W0121.wav --asr_type "llm" --model_dir pretrained_models/FireRedASR-LLM-L
```

**视频文件处理**
```bash
# 处理单个视频文件
$ speech2text.py --video_path video.mp4 --asr_type "aed" --model_dir pretrained_models/FireRedASR-AED-L
$ speech2text.py --input_path video.mp4 --asr_type "llm" --model_dir pretrained_models/FireRedASR-LLM-L

# 批量处理视频目录
$ speech2text.py --video_dir videos/ --asr_type "aed" --model_dir pretrained_models/FireRedASR-AED-L

# 混合处理音频和视频
$ speech2text.py --input_dir media/ --asr_type "aed" --model_dir pretrained_models/FireRedASR-AED-L
$ speech2text.py --input_paths audio.wav video.mp4 --asr_type "aed" --model_dir pretrained_models/FireRedASR-AED-L
```

**支持的视频格式：** MP4, AVI, MOV, MKV, FLV, WMV  
**支持的音频格式：** WAV, MP3, FLAC, M4A, AAC, OGG

### Python 使用示例

**处理音频文件**
```python
from fireredasr.models.fireredasr import FireRedAsr

batch_uttid = ["BAC009S0764W0121"]
batch_wav_path = ["examples/wav/BAC009S0764W0121.wav"]

# FireRedASR-AED
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")
results = model.transcribe(
    batch_uttid,
    batch_wav_path,
    {
        "use_gpu": 1,
        "beam_size": 3,
        "nbest": 1,
        "decode_max_len": 0,
        "softmax_smoothing": 1.25,
        "aed_length_penalty": 0.6,
        "eos_penalty": 1.0
    }
)
print(results)
```

**处理视频文件**
```python
from fireredasr.models.fireredasr import FireRedAsr

# 直接处理视频文件，FireRedASR会自动提取音频
batch_uttid = ["my_video"]
batch_video_path = ["path/to/video.mp4"]

# FireRedASR-LLM 处理视频
model = FireRedAsr.from_pretrained("llm", "pretrained_models/FireRedASR-LLM-L")
results = model.transcribe(
    batch_uttid,
    batch_video_path,
    {
        "use_gpu": 1,
        "beam_size": 3,
        "decode_max_len": 0,
        "decode_min_len": 0,
        "repetition_penalty": 3.0,
        "llm_length_penalty": 1.0,
        "temperature": 1.0
    }
)
print(results)

# 处理完成后，临时文件会自动清理
model.feat_extractor.cleanup_temp_files()
```

**混合处理音频和视频**
```python
# 可以在同一批次中混合处理不同格式的文件
batch_uttid = ["audio_sample", "video_sample"]
batch_media_path = ["audio.wav", "video.mp4"]

model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")
results = model.transcribe(batch_uttid, batch_media_path, config)
```

## 🛠️ 批量处理工具

### 批量转写工具
使用 `batch_transcribe.py` 批量处理多个音频和视频文件：

```bash
# 1. 将音频/视频文件放入 Use/Input/ 文件夹
# 2. 运行批量处理脚本
$ python batch_transcribe.py

# 脚本会自动：
# - 扫描 Use/Input/ 中的所有支持的媒体文件
# - 智能选择使用 AED 或 LLM 模型
# - 批量进行语音识别处理
# - 生成多格式结果：TXT、JSON格式
# - 将结果保存到 Use/Output/ 文件夹
```

**支持格式：**
- 音频：WAV, MP3, FLAC, M4A, AAC, OGG
- 视频：MP4, AVI, MOV, MKV, FLV, WMV

### 长视频转写工具
使用 `long_video_transcribe.py` 处理长时间的音频/视频文件：

```bash
# 智能长视频处理（推荐）
$ python long_video_transcribe.py

# 自定义模型和参数
$ python long_video_transcribe.py --model_type llm --max_duration 45

# 功能特点：
# - 🎯 自动VAD检测语音段，智能切分
# - ⚡ 硬件优化的并行处理
# - 🚀 智能模型加载和内存管理  
# - 📄 多格式输出：TXT、SRT、VTT字幕文件
# - 📊 详细的处理统计和性能指标
# - 🔄 断点续传支持，处理失败自动恢复
```

**长视频处理流程：**
1. **音频预处理** - 自动转换为16kHz单声道WAV格式
2. **智能切片** - 使用Silero VAD检测语音活动区间
3. **并行转写** - 硬件优化的批量处理
4. **结果拼接** - 时间戳对齐，生成完整文本和字幕

**性能优化特性：**
- 📈 自动硬件检测（GPU/CPU）
- 🧠 智能内存管理
- ⚡ 模型预加载和缓存
- 🔧 动态批处理大小调整

### 🔤 中文标点符号恢复（新功能）
FireRedASR 现已集成中文标点符号恢复功能，自动为转写文本添加标点符号：

**支持的标点符号：**
- 逗号（，）、句号（。）、问号（？）
- 感叹号（！）、顿号（、）、分号（；）

**使用方法：**
```bash
# 长视频转写（默认启用标点恢复）
$ python long_video_transcribe.py

# 禁用标点恢复
$ python long_video_transcribe.py --disable-punctuation

# 长音频转写（默认启用标点恢复）
$ python long_audio_transcribe.py --input_audio audio.mp3 --model_dir pretrained_models/FireRedASR-AED-L

# 批量转写（默认启用标点恢复）
$ python batch_transcribe.py --disable-punctuation
```

**高级参数配置：**
```bash
# 自定义标点恢复参数
$ python long_video_transcribe.py \
    --punctuation-model-dir /path/to/custom/model \
    --punctuation-chunk-size 512 \
    --punctuation-stride 256
```

**功能特点：**
- ✅ 自动下载并缓存标点恢复模型
- ✅ 滑动窗口处理长文本，无长度限制
- ✅ GPU/CPU 自适应，自动选择最优设备
- ✅ 错误降级机制，保证转写流程稳定
- ✅ 同时输出原始文本和带标点文本

**输出文件示例：**
```
output/
├── test.txt              # 原始转写文本
├── test_标点.txt         # 带标点文本
├── test.srt              # 原始字幕文件
└── test_标点.srt         # 带标点字幕文件
```
## 📋 使用技巧与最佳实践

### 模型选择建议
| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 短音频（< 30秒） | FireRedASR-LLM | 最高准确率 |
| 长音频（> 60秒） | FireRedASR-AED | 更好的稳定性 |
| 批量处理 | FireRedASR-AED | 速度更快 |
| 高质量转写 | FireRedASR-LLM | 语言理解更好 |

### 输入长度限制与解决方案
- **FireRedASR-AED**: 建议最长 60 秒，超过可能有幻觉问题
- **FireRedASR-LLM**: 建议最长 30 秒，超过行为不确定
- **长音频解决方案**: 自动使用 `long_video_transcribe.py` 工具智能切分处理

### 批量束搜索优化
- 使用 FireRedASR-LLM 时，确保输入语音长度相似
- 长度差异大时，设置 `batch_size=1` 或按长度排序
- 利用硬件管理器自动优化批处理大小

### 性能优化技巧
1. **硬件配置**
   - GPU 内存 ≥ 8GB（LLM模型）
   - 系统内存 ≥ 16GB（长音频）
   - 存储空间预留音频3-5倍大小

2. **处理策略**
   - 短音频：直接处理
   - 长音频：使用VAD自动切分
   - 批量文件：使用批量处理工具

3. **质量控制**
   - 音频预处理：16kHz, 单声道
   - 使用VAD去除静音段
   - 检查输出统计信息验证质量


## 致谢
感谢以下开源项目：
- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [icefall/ASR_LLM](https://github.com/k2-fsa/icefall/tree/master/egs/speech_llm/ASR_LLM)
- [WeNet](https://github.com/wenet-e2e/wenet)
- [Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer)


## 引用
```bibtex
@article{xu2025fireredasr,
  title={FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition Models from Encoder-Decoder to LLM Integration},
  author={Xu, Kai-Tuo and Xie, Feng-Long and Tang, Xu and Hu, Yao},
  journal={arXiv preprint arXiv:2501.14350},
  year={2025}
}
```
