# FireRedASR 模型配置步骤详细指南

## 概述
FireRedASR 提供两种模型变体：
- **FireRedASR-AED**: 高效的注意力编码器-解码器架构 (1.1B参数)
- **FireRedASR-LLM**: 基于大语言模型的SOTA性能架构 (8.3B参数)

## 第一步：环境准备

### 1.1 克隆项目并创建环境
```bash
git clone https://github.com/FireRedTeam/FireRedASR.git
cd FireRedASR
conda create --name fireredasr python=3.10
conda activate fireredasr
```

### 1.2 安装依赖
```bash
pip install -r requirements.txt
```

### 1.3 配置环境变量
```bash
export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH
```

## 第二步：下载预训练模型

### 2.1 下载 FireRedASR-AED-L 模型
从 [HuggingFace](https://huggingface.co/fireredteam/FireRedASR-AED-L) 下载以下文件到 `pretrained_models/FireRedASR-AED-L/` 目录：
- `model.pth.tar` - 主模型权重
- `cmvn.ark` - 倒谱均值方差归一化文件
- `dict.txt` - 词典文件
- `config.yaml` - 配置文件

### 2.2 下载 FireRedASR-LLM-L 模型（可选）
从 [HuggingFace](https://huggingface.co/fireredteam/FireRedASR-LLM-L) 下载以下文件到 `pretrained_models/FireRedASR-LLM-L/` 目录：
- `model.pth.tar` - 主模型权重
- `asr_encoder.pth.tar` - ASR编码器权重
- `cmvn.ark` - 倒谱均值方差归一化文件
- `config.yaml` - 配置文件

**方案一：使用 HuggingFace 镜像站（推荐）**
```bash
# 进入项目目录
cd /home/gpr/FireRedASR

# 创建目标目录
mkdir -p pretrained_models
cd pretrained_models

# 使用 hf-mirror.com 镜像站
export HF_ENDPOINT=https://hf-mirror.com
git lfs install
git clone https://hf-mirror.com/fireredteam/FireRedASR-LLM-L

# 或者使用其他镜像站
# git clone https://huggingface.co/fireredteam/FireRedASR-LLM-L
```

**方案二：使用官方 HuggingFace（直连，无代理）**
```bash
# 直接连接 HuggingFace 官方（需要网络可达）
cd /home/gpr/FireRedASR/pretrained_models
git lfs install
git clone https://huggingface.co/fireredteam/FireRedASR-LLM-L
cd FireRedASR-LLM-L
git lfs pull  # 确保所有大文件都已下载
```

**方案三：使用代理（如果你有VPN）**
```bash
# 设置HTTP代理（替换为你的代理地址和端口）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

cd /home/gpr/FireRedASR/pretrained_models
git lfs install
git clone https://huggingface.co/fireredteam/FireRedASR-LLM-L
cd FireRedASR-LLM-L
git lfs pull  # 确保所有大文件都已下载
```

**方案四：使用 huggingface-cli（最可靠，支持断点续传）**
```bash
# 安装 huggingface_hub
pip install huggingface_hub[cli]

# 方案4A：通过镜像站下载
cd /home/gpr/FireRedASR/pretrained_models
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download fireredteam/FireRedASR-LLM-L --local-dir FireRedASR-LLM-L

# 方案4B：直连官方下载（无需镜像）
cd /home/gpr/FireRedASR/pretrained_models
unset HF_ENDPOINT  # 清除镜像设置
huggingface-cli download fireredteam/FireRedASR-LLM-L --local-dir FireRedASR-LLM-L
```

**方案五：修复git lfs问题**
```bash
# 先删除错误的下载
cd /home/gpr/FireRedASR/pretrained_models
rm -rf FireRedASR-LLM-L

# 重新安装git lfs
git lfs install --force

# 使用镜像站下载
export HF_ENDPOINT=https://hf-mirror.com
git clone https://hf-mirror.com/fireredteam/FireRedASR-LLM-L
cd FireRedASR-LLM-L
git lfs pull  # 强制下载大文件
```

**方案六：使用 wget 单独下载每个文件**
```bash
# 进入项目目录
cd /home/gpr/FireRedASR

# 删除错误的下载目录
rm -rf pretrained_models/FireRedASR-LLM-L

# 创建目标目录
mkdir -p pretrained_models/FireRedASR-LLM-L
cd pretrained_models/FireRedASR-LLM-L

# 使用镜像站下载模型文件
wget https://hf-mirror.com/fireredteam/FireRedASR-LLM-L/resolve/main/model.pth.tar
wget https://hf-mirror.com/fireredteam/FireRedASR-LLM-L/resolve/main/asr_encoder.pth.tar
wget https://hf-mirror.com/fireredteam/FireRedASR-LLM-L/resolve/main/cmvn.ark
wget https://hf-mirror.com/fireredteam/FireRedASR-LLM-L/resolve/main/config.yaml

# 或直接从官方下载（如果网络可达）
# wget https://huggingface.co/fireredteam/FireRedASR-LLM-L/resolve/main/model.pth.tar
# wget https://huggingface.co/fireredteam/FireRedASR-LLM-L/resolve/main/asr_encoder.pth.tar
# wget https://huggingface.co/fireredteam/FireRedASR-LLM-L/resolve/main/cmvn.ark
# wget https://huggingface.co/fireredteam/FireRedASR-LLM-L/resolve/main/config.yaml
```

### 2.3 下载 Qwen2-7B-Instruct（仅LLM模型需要）
如果使用FireRedASR-LLM，需要下载 [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) 到 `pretrained_models/` 目录，然后创建软链接：

**方案一：使用 HuggingFace 镜像站（推荐）**
```bash
# 进入pretrained_models目录
cd /home/gpr/FireRedASR/pretrained_models

# 使用镜像站下载 Qwen2-7B-Instruct
export HF_ENDPOINT=https://hf-mirror.com
git lfs install
git clone https://hf-mirror.com/Qwen/Qwen2-7B-Instruct
cd Qwen2-7B-Instruct
git lfs pull  # 确保大文件下载完整

# 创建软链接
cd ../FireRedASR-LLM-L
ln -s ../Qwen2-7B-Instruct
```

**方案二：使用官方 HuggingFace（直连）**
```bash
# 进入pretrained_models目录
cd /home/gpr/FireRedASR/pretrained_models

# 直连官方下载 Qwen2-7B-Instruct
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-7B-Instruct
cd Qwen2-7B-Instruct
git lfs pull  # 确保大文件下载完整

# 创建软链接
cd ../FireRedASR-LLM-L
ln -s ../Qwen2-7B-Instruct
```

**方案三：使用 ModelScope 下载**
```bash
# 进入pretrained_models目录
cd /home/gpr/FireRedASR/pretrained_models

# 使用 ModelScope 下载（Qwen2在ModelScope上存在）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2-7B-Instruct', local_dir='Qwen2-7B-Instruct')
"

# 创建软链接
cd FireRedASR-LLM-L
ln -s ../Qwen2-7B-Instruct
```

**方案四：使用 huggingface_hub Python 库下载：**
```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 使用镜像站下载
cd /home/gpr/FireRedASR/pretrained_models
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2-7B-Instruct', local_dir='Qwen2-7B-Instruct')"

# 或直连官方下载
# cd /home/gpr/FireRedASR/pretrained_models
# unset HF_ENDPOINT
# python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2-7B-Instruct', local_dir='Qwen2-7B-Instruct')"

# 创建软链接
cd FireRedASR-LLM-L
ln -s ../Qwen2-7B-Instruct
```

**验证下载完成：**
```bash
# 检查目录结构
cd /home/gpr/FireRedASR
ls -la pretrained_models/FireRedASR-LLM-L/
ls -la pretrained_models/Qwen2-7B-Instruct/
```

## 第三步：预处理音频文件

### 3.1 音频格式要求
将音频转换为 16kHz 16-bit PCM WAV 格式：
```bash
ffmpeg -i input_audio -ar 16000 -ac 1 -acodec pcm_s16le -f wav output.wav
```

### 3.2 音频长度限制
- **FireRedASR-AED**: 支持最长60秒音频，超过200秒会触发位置编码错误
- **FireRedASR-LLM**: 支持最长30秒音频

## 第四步：快速开始测试

### 4.1 使用预置脚本测试
```bash
cd examples
# 测试 AED 模型
bash inference_fireredasr_aed.sh
# 测试 LLM 模型
bash inference_fireredasr_llm.sh
```

### 4.2 命令行使用
```bash
# AED 模型
speech2text.py --wav_path examples/wav/BAC009S0764W0121.wav \
               --asr_type "aed" \
               --model_dir pretrained_models/FireRedASR-AED-L

# LLM 模型
speech2text.py --wav_path examples/wav/BAC009S0764W0121.wav \
               --asr_type "llm" \
               --model_dir pretrained_models/FireRedASR-LLM-L
```

## 第五步：Python API 使用

### 5.1 FireRedASR-AED 使用示例
```python
from fireredasr.models.fireredasr import FireRedAsr

batch_uttid = ["BAC009S0764W0121"]
batch_wav_path = ["examples/wav/BAC009S0764W0121.wav"]

# 加载 AED 模型
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")

# 配置解码参数
decode_config = {
    "use_gpu": 1,
    "beam_size": 3,
    "nbest": 1,
    "decode_max_len": 0,
    "softmax_smoothing": 1.25,
    "aed_length_penalty": 0.6,
    "eos_penalty": 1.0
}

results = model.transcribe(batch_uttid, batch_wav_path, decode_config)
print(results)
```

### 5.2 FireRedASR-LLM 使用示例
```python
from fireredasr.models.fireredasr import FireRedAsr

batch_uttid = ["BAC009S0764W0121"]
batch_wav_path = ["examples/wav/BAC009S0764W0121.wav"]

# 加载 LLM 模型
model = FireRedAsr.from_pretrained("llm", "pretrained_models/FireRedASR-LLM-L")

# 配置解码参数
decode_config = {
    "use_gpu": 1,
    "beam_size": 3,
    "decode_max_len": 0,
    "decode_min_len": 0,
    "repetition_penalty": 3.0,
    "llm_length_penalty": 1.0,
    "temperature": 1.0
}

results = model.transcribe(batch_uttid, batch_wav_path, decode_config)
print(results)
```

## 第六步：高级配置与优化

### 6.1 批处理束搜索注意事项
- 使用 FireRedASR-LLM 进行批处理时，确保输入音频长度相似
- 长度差异较大时建议将 `batch_size` 设为 1 避免重复问题

### 6.2 GPU 配置
通过环境变量指定 GPU：
```bash
CUDA_VISIBLE_DEVICES=0 speech2text.py [其他参数]
```

### 6.3 解码参数调优
根据具体应用场景调整以下参数：
- `beam_size`: 束搜索宽度，影响准确性和速度
- `temperature`: 采样温度，控制输出多样性
- `repetition_penalty`: 重复惩罚，减少重复输出
- `length_penalty`: 长度惩罚，平衡长短序列

## 第七步：目录结构检查

确保你的目录结构如下：
```
FireRedASR/
├── pretrained_models/
│   ├── FireRedASR-AED-L/
│   │   ├── model.pth.tar
│   │   ├── cmvn.ark
│   │   ├── dict.txt
│   │   └── config.yaml
│   ├── FireRedASR-LLM-L/
│   │   ├── model.pth.tar
│   │   ├── asr_encoder.pth.tar
│   │   ├── cmvn.ark
│   │   ├── config.yaml
│   │   └── Qwen2-7B-Instruct/ (软链接)
│   └── Qwen2-7B-Instruct/
├── examples/
├── fireredasr/
└── requirements.txt
```

## 常见问题排查

### 下载问题
1. **git clone 卡住或只下载到 KB 级别文件**:
   - 这是 git lfs 配置问题，文件实际没有下载
   - 解决方案：使用 `huggingface-cli` 或 `wget` 方式下载
   
2. **HuggingFace 连接超时**:
   - 使用镜像站：`export HF_ENDPOINT=https://hf-mirror.com`
   - 或配置代理：`export https_proxy=http://127.0.0.1:7890`

3. **ModelScope 404 错误**:
   - FireRedASR-LLM-L 在 ModelScope 上不存在，只能用 HuggingFace
   - Qwen2-7B-Instruct 在 ModelScope 上存在，可以使用

### 运行问题
4. **模型文件缺失**: 确保所有必需文件都已下载到正确位置
5. **环境变量未设置**: 检查 PATH 和 PYTHONPATH 是否正确配置
6. **CUDA 错误**: 验证 GPU 驱动和 PyTorch CUDA 版本兼容性
7. **音频格式错误**: 确保音频为 16kHz 单声道 WAV 格式
8. **内存不足**: 大模型需要充足的 GPU 内存，考虑减小 batch_size

### 推荐下载顺序
1. 首选：`huggingface-cli download` (最稳定)
2. 备选：`wget` 直接下载 (适合小文件)
3. 最后：修复 git lfs 后重新克隆

## 检查结果 (2025-07-27 18:13)

### FireRedASR-LLM-L 模型文件检查

**✅ 已下载文件:**
- `model.pth.tar` - 3.6GB (正常)
- `asr_encoder.pth.tar` - 1.4KB (正常)
- `cmvn.ark` - 1.3KB (正常)
- `cmvn.txt` - 2.9KB (正常)
- `README.md` - 6.7KB (正常)

**❌ 存在问题:**
- `config.yaml` - **0 字节** (原始仓库中就是空文件，这是正常的)

**修复确认:**
通过检查源代码发现，FireRedASR-LLM 模型不依赖 config.yaml 文件。模型直接从以下路径加载：
- `model.pth.tar` - 主模型权重
- `asr_encoder.pth.tar` - ASR编码器权重  
- `cmvn.ark` - 倒谱均值方差归一化文件
- `Qwen2-7B-Instruct/` - LLM基础模型（软链接）

**结论:** ✅ **FireRedASR-LLM-L 模型文件下载完整，可以正常使用！**

## 下载 FireRedASR-AED-L 模型

### AED 模型优势
- **参数量小**: 1.1B（相比LLM的8.3B）
- **速度快**: 推理速度更快
- **显存友好**: 只需要约2-3GB显存
- **支持长音频**: 最长支持60秒音频（LLM只支持30秒）

### 下载方法

**方案一：使用 huggingface-cli（推荐）**
```bash
# 进入项目目录
cd /home/gpr/FireRedASR/pretrained_models

# 使用镜像站下载
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download fireredteam/FireRedASR-AED-L --local-dir FireRedASR-AED-L

# 检查下载结果
ls -la FireRedASR-AED-L/
```

**方案二：使用 git clone**
```bash
# 进入项目目录
cd /home/gpr/FireRedASR/pretrained_models

# 使用镜像站克隆
export HF_ENDPOINT=https://hf-mirror.com
git lfs install
git clone https://hf-mirror.com/fireredteam/FireRedASR-AED-L
cd FireRedASR-AED-L
git lfs pull  # 确保大文件下载完整
```

**方案三：使用 wget 单独下载**
```bash
# 创建目录
mkdir -p /home/gpr/FireRedASR/pretrained_models/FireRedASR-AED-L
cd /home/gpr/FireRedASR/pretrained_models/FireRedASR-AED-L

# 下载必需文件
wget https://hf-mirror.com/fireredteam/FireRedASR-AED-L/resolve/main/model.pth.tar
wget https://hf-mirror.com/fireredteam/FireRedASR-AED-L/resolve/main/cmvn.ark
wget https://hf-mirror.com/fireredteam/FireRedASR-AED-L/resolve/main/dict.txt
wget https://hf-mirror.com/fireredteam/FireRedASR-AED-L/resolve/main/config.yaml
```

### AED 模型必需文件清单
```
FireRedASR-AED-L/
├── model.pth.tar    - 主模型权重（约2.2GB）
├── cmvn.ark        - 倒谱均值方差归一化文件
├── dict.txt        - 词典文件
└── config.yaml     - 配置文件
```

### 验证下载完成
```bash
# 检查文件大小
ls -lh /home/gpr/FireRedASR/pretrained_models/FireRedASR-AED-L/

# 测试 AED 模型
cd /home/gpr/FireRedASR/examples
bash inference_fireredasr_aed.sh
```

### AED vs LLM 对比

| 特性 | FireRedASR-AED-L | FireRedASR-LLM-L |
|------|------------------|------------------|
| 参数量 | 1.1B | 8.3B |
| 显存需求 | ~3GB | ~16GB |
| 推理速度 | 快 | 较慢 |
| 识别精度 | 高 | 最高 |
| 音频长度 | 最长60s | 最长30s |
| 部署难度 | 简单 | 复杂 |

**推荐使用场景：**
- **生产环境**: 优先使用 AED 模型（速度快、稳定）
- **最高精度**: 使用 LLM 模型（SOTA性能）
- **长音频**: 必须使用 AED 模型