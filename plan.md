# FireRedASR 中文标点符号恢复功能集成方案

## 📋 项目概述

### 核心目标
在 FireRedASR 项目中集成中文标点符号恢复功能，实现从音频输入到带标点文本输出的完整流程。

### 用户需求
- **集成范围**: 所有转写脚本（`long_video_transcribe.py`, `long_audio_transcribe.py`, `batch_transcribe.py`）
- **默认行为**: 默认启用标点恢复功能
- **输出策略**: 同时保留无标点和带标点两个版本
- **模型管理**: 自动下载和缓存 `zh-wiki-punctuation-restore` 模型

### 技术流程
```
音频输入 → 切片/转写 → 文本拼接 → [标点恢复] → 双版本输出
```

## 🏗️ 技术架构设计

### 核心模块结构
```
fireredasr/utils/punctuation_restore.py [新增核心模块]
├── PunctuationRestorer 类
│   ├── 模型自动下载和缓存
│   ├── 滑动窗口文本处理
│   ├── GPU/CPU自适应推理
│   └── 错误处理和降级机制
```

### 集成点设计
- **long_video_transcribe.py**: 在 `concatenate_results()` 方法后集成
- **long_audio_transcribe.py**: 在 `merge_transcripts()` 方法后集成  
- **batch_transcribe.py**: 在 `save_results()` 方法中集成

### 输出文件命名规范
```
原始版本: filename_transcription.txt
标点版本: filename_transcription_with_punctuation.txt
原始SRT: filename_transcription.srt  
标点SRT: filename_transcription_with_punctuation.srt
```

## 🔧 核心功能模块设计

### PunctuationRestorer 类设计

#### 主要功能
- **模型管理**: 自动下载、缓存和加载 `p208p2002/zh-wiki-punctuation-restore` 模型
- **文本处理**: 滑动窗口算法处理长文本
- **硬件适配**: GPU/CPU 自动选择和优化
- **错误处理**: 网络、内存、处理错误的优雅降级

#### 核心方法
```python
class PunctuationRestorer:
    def __init__(self, model_name="p208p2002/zh-wiki-punctuation-restore")
    def load_model(self) -> bool
    def restore_punctuation(self, text: str) -> str
    def process_long_text(self, text: str, chunk_size: int = 256, stride: int = 128) -> str
    def cleanup(self)
```

#### 技术参数
- **支持标点**: `， 、 。 ？ ！ ；`
- **滑动窗口**: stride=128, chunk_size=256
- **模型缓存**: `~/.cache/huggingface/transformers/`
- **批处理**: batch_size=8 (可配置)

## ⚙️ CLI 参数扩展

### 新增命令行参数
```bash
--enable-punctuation        # 启用标点恢复 (默认: True)
--disable-punctuation       # 禁用标点恢复
--punctuation-model-dir     # 自定义模型路径
--punctuation-chunk-size    # 文本块大小 (默认: 256)
--punctuation-stride        # 滑动窗口步长 (默认: 128)
```

## 🛡️ 错误处理策略

### 三级降级机制
1. **网络错误**: 模型下载失败 → 检查本地缓存 → 跳过标点恢复
2. **内存错误**: GPU内存不足 → 切换CPU → 降低批处理大小
3. **处理错误**: 标点恢复失败 → 保留原始文本 → 记录错误日志

## 📦 依赖管理

### requirements.txt 更新
```python
# 新增标点恢复依赖
zhpr>=0.3.0                 # 中文标点恢复核心库
accelerate>=0.20.0          # GPU加速支持 (可选)
```

## 🔧 实施计划

### Phase 1: 核心模块开发
- [ ] **任务1**: 创建标点恢复核心模块 (`punctuation_restore.py`)

### Phase 2: 依赖和配置
- [ ] **任务2**: 更新项目依赖配置 (`requirements.txt`)

### Phase 3: 脚本集成
- [ ] **任务3**: 集成标点恢复到长视频转写脚本 (`long_video_transcribe.py`)
- [ ] **任务4**: 集成标点恢复到长音频转写脚本 (`long_audio_transcribe.py`)
- [ ] **任务5**: 集成标点恢复到批量转写脚本 (`batch_transcribe.py`)

### Phase 4: 功能完善
- [ ] **任务6**: 创建 CLI 命令行参数支持
- [ ] **任务7**: 实现模型自动下载和缓存机制
- [ ] **任务8**: 添加配置管理和错误处理

### Phase 5: 文档和测试
- [ ] **任务9**: 更新项目文档 (`README.md`)
- [ ] **任务10**: 创建使用示例和测试脚本

## 📝 实施细节

### 文件修改清单
```
新增文件:
├── fireredasr/utils/punctuation_restore.py  [核心模块]

修改文件:
├── requirements.txt                          [依赖更新]
├── long_video_transcribe.py                 [集成标点恢复]
├── long_audio_transcribe.py                 [集成标点恢复]
├── batch_transcribe.py                      [集成标点恢复]
├── README.md                                [文档更新]
```

### 兼容性保证
- **API兼容**: 所有现有函数签名保持不变
- **参数兼容**: 新参数都有合理默认值
- **输出兼容**: 原有输出格式完全保持
- **性能兼容**: 不影响原始转写性能

## 🎯 预期成果

### 功能目标
- ✅ 实现完整的"音频 → 带标点文本"一键处理流程
- ✅ 支持所有现有转写脚本的标点恢复功能
- ✅ 提供灵活的参数配置和控制选项
- ✅ 保持100%向后兼容性

### 性能目标
- ⚡ 标点恢复处理时间 < 原始转写时间的20%
- 💾 额外内存开销 < 1GB (GPU模式)
- 🎯 标点恢复准确率 > 95%
- 🔧 错误率 < 1% (优雅降级机制)

---

**项目状态**: 方案制定完成，等待实施  
**预计完成时间**: 3-5 个工作日  
**技术风险**: 低风险，方案经过充分设计