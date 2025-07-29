<div align="center">
  <img src="assets/FireRedASR_logo.png" alt="FireRedASR Logo" width="200" />
  
  <h1>FireRedASR</h1>
  <h3>🔥 开源工业级自动语音识别系统</h3>
  
  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue.svg">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
    <img alt="Stars" src="https://img.shields.io/github/stars/EchoJonhson/video?style=social">
  </p>

  <p>
    <a href="https://arxiv.org/pdf/2501.14350">📄 论文</a> •
    <a href="https://huggingface.co/fireredteam">🤗 模型</a> •
    <a href="https://fireredteam.github.io/demos/firered_asr/">📖 博客</a> •
    <a href="#快速开始">🚀 快速开始</a> •
    <a href="#使用文档">📚 文档</a>
  </p>
</div>

---

## 📌 项目简介

**FireRedASR** 是一个开源的工业级自动语音识别（ASR）系统，提供高精度的中文、方言和英文语音识别能力。项目基于最新的深度学习技术，在公开基准测试中达到了业界领先水平。

### ✨ 核心特性

- 🎯 **高精度识别** - 在公开普通话ASR基准测试中达到SOTA水平
- 🌏 **多语言支持** - 支持普通话、中国方言、英语等多种语言
- 🎵 **歌词识别** - 业界领先的音乐歌词识别能力
- 📹 **视频处理** - 原生支持视频文件，自动提取音频并转写
- ⚡ **高性能** - GPU/CPU自适应，支持批量并行处理
- 🔤 **智能标点** - 自动恢复中文标点符号，提升可读性
- 📝 **多格式输出** - 支持TXT、SRT、VTT、JSON等多种格式

### 🏆 性能指标

<details>
<summary>点击查看详细基准测试结果</summary>

#### 普通话ASR基准测试 (CER%)
| 模型 | aishell1 | aishell2 | ws_net | ws_meeting | 平均 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **FireRedASR-LLM** | **0.76** | **2.15** | **4.60** | **4.67** | **3.05** |
| FireRedASR-AED | 0.55 | 2.52 | 4.88 | 4.76 | 3.18 |
| Seed-ASR | 0.68 | 2.27 | 4.66 | 5.69 | 3.33 |
| Qwen-Audio | 1.30 | 3.10 | 9.50 | 10.87 | 6.19 |

#### 方言与英语测试
| 模型 | KeSpeech | LibriSpeech-clean | LibriSpeech-other |
|:---:|:---:|:---:|:---:|
| **FireRedASR-LLM** | **3.56** | **1.73** | **3.67** |
| FireRedASR-AED | 4.48 | 1.93 | 4.44 |

</details>

---

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
# 克隆项目
git clone https://github.com/EchoJonhson/video.git
cd video

# 创建虚拟环境
conda create -n fireredasr python=3.10
conda activate fireredasr

# 安装依赖
pip install -r requirements.txt
```

### 2️⃣ 下载模型

从 [HuggingFace](https://huggingface.co/fireredteam) 下载预训练模型：

```bash
# 创建模型目录
mkdir -p pretrained_models

# 下载模型（以AED为例）
git clone https://huggingface.co/fireredteam/FireRedASR-AED-L pretrained_models/FireRedASR-AED-L

# 如果使用LLM模型，还需下载Qwen2
git clone https://huggingface.co/Qwen/Qwen2-7B-Instruct pretrained_models/Qwen2-7B-Instruct
```

### 3️⃣ 开始使用

```bash
# 转写单个音频
python fireredasr/speech2text.py --input_path audio.wav --model_dir pretrained_models/FireRedASR-AED-L

# 转写视频文件
python fireredasr/speech2text.py --input_path video.mp4 --model_dir pretrained_models/FireRedASR-AED-L

# 批量转写（推荐）
python batch_transcribe.py
```

---

## 📚 使用文档

### 🎯 典型使用场景

<table>
<tr>
<td width="50%">

#### 场景一：批量视频转文字
```bash
# 1. 将视频放入 Use/Input/
# 2. 运行批量处理
python batch_transcribe.py

# 3. 在 Use/Output/ 查看结果
```

</td>
<td width="50%">

#### 场景二：长视频智能转写
```bash
# 自动VAD切分，并行处理
python long_video_transcribe.py

# 输出完整字幕文件
# ✅ test.srt (原始字幕)
# ✅ test_标点.srt (带标点)
```

</td>
</tr>
</table>

### 🛠️ 高级功能

#### 1. 标点符号恢复
```bash
# 默认启用，可通过参数控制
python long_video_transcribe.py --disable-punctuation
```

#### 2. 段落智能分段
```python
from fireredasr.utils.paragraph_segmentation import ParagraphSegmenter

segmenter = ParagraphSegmenter()
paragraphs = segmenter.segment(text)
```

#### 3. Python API 调用
```python
from fireredasr.models.fireredasr import FireRedAsr

# 初始化模型
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")

# 转写音频
results = model.transcribe(
    ["sample_id"],
    ["audio.wav"],
    {"use_gpu": 1, "beam_size": 3}
)
```

### 📊 模型选择指南

| 使用场景 | 推荐模型 | 理由 |
|---------|---------|------|
| 🎬 短视频/短音频 | FireRedASR-LLM | 最高准确率，语言理解能力强 |
| 📺 长视频/播客 | FireRedASR-AED | 稳定性好，处理速度快 |
| 🎵 音乐/歌词 | FireRedASR-LLM | 更好的歌词识别效果 |
| 💼 批量处理 | FireRedASR-AED | 效率高，资源占用少 |

---

## 🔧 系统架构

FireRedASR 提供两种架构选择：

<div align="center">
  <img src="assets/FireRedASR_model.png" alt="FireRedASR Architecture" width="80%" />
</div>

- **FireRedASR-LLM**：编码器-适配器-LLM架构，追求最高精度
- **FireRedASR-AED**：注意力编码器-解码器架构，平衡性能与效率

---

## 📈 项目特色

### 1. 🚀 工业级优化
- **硬件自适应**：自动检测GPU/CPU，智能分配资源
- **内存管理**：动态批处理，避免OOM
- **断点续传**：长视频处理支持中断恢复
- **并行处理**：多进程/多线程优化

### 2. 🎨 用户体验
- **进度可视化**：实时显示处理进度和统计
- **错误恢复**：自动重试失败片段
- **格式兼容**：支持主流音视频格式
- **一键部署**：简化的安装和配置流程

### 3. 🔬 前沿技术
- **VAD技术**：Silero VAD 精准语音活动检测
- **标点恢复**：基于深度学习的标点符号预测
- **智能分段**：语义相关的段落自动分割
- **混合精度**：FP16/INT8 量化加速

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📝 引用

如果您在研究中使用了 FireRedASR，请引用：

```bibtex
@article{xu2025fireredasr,
  title={FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition Models from Encoder-Decoder to LLM Integration},
  author={Xu, Kai-Tuo and Xie, Feng-Long and Tang, Xu and Hu, Yao},
  journal={arXiv preprint arXiv:2501.14350},
  year={2025}
}
```

---

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

---

## 🙏 致谢

感谢以下开源项目的支持：
- [Qwen2](https://github.com/QwenLM/Qwen2)
- [WeNet](https://github.com/wenet-e2e/wenet)
- [icefall](https://github.com/k2-fsa/icefall)
- [Silero VAD](https://github.com/snakers4/silero-vad)

---

<div align="center">
  <p>
    <b>🌟 如果觉得有帮助，请给个 Star！</b>
  </p>
  <p>
    <sub>Made with ❤️ by FireRed Team</sub>
  </p>
</div>