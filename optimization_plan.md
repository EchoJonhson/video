# FireRedASR 性能优化方案 - 简洁实用版

## 硬件现状
- CPU: Intel i9-14900KF (32线程) - **充足**
- GPU: 2×RTX 4000 (各20GB) - **受限于ollama占用**
- 内存: 128GB - **充足**

## 核心问题
当前代码仅使用2-4个CPU线程，严重浪费了32线程的计算能力。

## 优化方案（按简洁度排序）

### 方案一：最小改动（5分钟实施）
**只改一个参数，立即提速2-3倍**

```python
# long_video_transcribe.py 第492行
# 原代码：max_workers = 2
# 改为：
max_workers = 8  # 充分利用CPU多核
```

### 方案二：简单优化（2小时实施）
**改动20行代码，提速3-5倍**

1. **增加并行度**
   ```python
   # 根据CPU核心数动态调整
   max_workers = min(12, multiprocessing.cpu_count() // 2)
   ```

2. **批量处理音频段**
   ```python
   # 合并小音频段，减少模型调用次数
   def batch_audio_segments(segments, batch_size=4):
       batched = []
       for i in range(0, len(segments), batch_size):
           batched.append(merge_segments(segments[i:i+batch_size]))
       return batched
   ```

3. **优化VAD参数**
   ```python
   # 减少音频碎片
   self.min_speech_duration_ms = 2000  # 从1000增加到2000
   self.max_speech_duration_s = 60     # 从30增加到60
   ```

### 方案三：中等优化（1天实施）
**改动100行代码，提速6-8倍**

1. **多进程替代多线程**
   ```python
   from multiprocessing import Pool
   
   # 创建进程池
   with Pool(processes=8) as pool:
       results = pool.map(transcribe_segment, audio_segments)
   ```

2. **实现音频预加载**
   ```python
   # 使用队列预加载下一批音频
   from queue import Queue
   audio_queue = Queue(maxsize=16)
   
   # 后台线程预加载
   def preload_audio(segments):
       for seg in segments:
           audio_data = load_audio(seg)
           audio_queue.put(audio_data)
   ```

3. **内存映射优化**
   ```python
   # 大文件使用内存映射
   import mmap
   with open(audio_file, 'r+b') as f:
       with mmap.mmap(f.fileno(), 0) as mmapped:
           # 直接操作映射内存
   ```

### 方案四：可选高级优化（3天实施）
**如需极致性能，可考虑**

1. **ONNX Runtime**（提速2-3倍）
   - 一键转换脚本
   - 推理代码基本不变

2. **INT8量化**（提速2倍）
   - PyTorch自带API
   - 精度损失极小

## 实施建议

### 第一步：先试方案一
- 改一行代码
- 测试效果
- 如果满意就停止

### 第二步：如需更快，实施方案二
- 逐个功能测试
- 确保稳定性

### 第三步：追求极致才考虑方案三、四

## 预期效果

| 方案 | 改动量 | 实施时间 | 预期提速 |
|------|--------|----------|----------|
| 方案一 | 1行 | 5分钟 | 2-3倍 |
| 方案二 | 20行 | 2小时 | 3-5倍 |
| 方案三 | 100行 | 1天 | 6-8倍 |
| 方案四 | 新增模块 | 3天 | 10-12倍 |

## 风险控制
- 每步都可独立测试
- 改动都可回滚
- 不破坏原有架构

## 总结
**少即是多** - 先用最简单的方法，效果不够再加码。