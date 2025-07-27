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

## 使用技巧
### 批量束搜索
- 在使用 FireRedASR-LLM 进行批量束搜索时，请确保输入语音的长度相似。如果语音长度差异较大，较短的语音可能会出现重复问题。您可以按长度对数据集进行排序，或者将 `batch_size` 设置为 1 来避免这个问题。

### 输入长度限制
- FireRedASR-AED 支持最长 60 秒的音频输入。超过 60 秒的输入可能会导致幻觉问题，超过 200 秒的输入将触发位置编码错误。
- FireRedASR-LLM 支持最长 30 秒的音频输入。更长输入的行为目前尚不清楚。


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
