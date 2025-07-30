# FireRedASR 性能优化方案二 - 详细实施指南

## 方案概述
**目标：通过最小改动实现3-5倍性能提升**
- 改动代码：约20行
- 实施时间：2小时
- 预期提速：3-5倍
- 核心思路：充分利用32线程CPU，减少模型调用开销

## 硬件环境
- CPU: Intel i9-14900KF (24核32线程)
- GPU: 2×RTX 4000 (各20GB，被ollama占用)
- 内存: 128GB
- 当前瓶颈：仅使用2-4个CPU线程

## 实施步骤

### 步骤1：动态并行度优化
**目标：根据任务规模动态调整并行线程数**

#### 1.1 修改 `fireredasr/utils/cpu_optimization_config.py`
```python
# 第89-122行，优化 get_dynamic_config 方法
def get_dynamic_config(self, segment_count: int, model_type: str) -> Dict[str, Any]:
    if model_type == "llm":
        base_config = self.get_llm_optimization_config()
        
        # 动态调整策略 - 更激进的并行度
        if segment_count < 10:
            base_config["max_workers"] = 4  # 原2，现4
        elif segment_count < 50:
            base_config["max_workers"] = 6  # 原3，现6
        elif segment_count < 100:
            base_config["max_workers"] = 8  # 原4，现8
        else:
            base_config["max_workers"] = 12  # 原5，现12
            
        # 动态调整预读取
        base_config["memory_config"]["prefetch_segments"] = min(segment_count // 4, 16)
```

#### 1.2 修改 `long_video_transcribe.py`
```python
# 第495行附近，使用动态配置
max_workers = min(opt_config["max_workers"], multiprocessing.cpu_count() // 2)
```

### 步骤2：批量音频处理
**目标：合并短音频段，减少模型调用次数**

#### 2.1 添加批量处理函数
在 `long_video_transcribe.py` 中添加：
```python
def batch_audio_segments(self, segments, segments_dir, batch_size=4):
    """批量合并相邻的短音频段"""
    batched_segments = []
    current_batch = []
    current_duration = 0
    
    for segment in segments:
        segment_duration = segment['end'] - segment['start']
        
        # 如果当前批次加上这个段会超过最大时长，或批次已满
        if (current_duration + segment_duration > 30 or 
            len(current_batch) >= batch_size) and current_batch:
            # 保存当前批次
            batched_segments.append(self._merge_batch(current_batch, segments_dir))
            current_batch = [segment]
            current_duration = segment_duration
        else:
            current_batch.append(segment)
            current_duration += segment_duration
    
    # 处理最后一个批次
    if current_batch:
        batched_segments.append(self._merge_batch(current_batch, segments_dir))
    
    return batched_segments

def _merge_batch(self, batch, segments_dir):
    """合并一个批次的音频段"""
    if len(batch) == 1:
        return batch[0]
    
    # 合并音频文件
    first_segment = batch[0]
    merged_file = f"batch_{first_segment['index']:03d}_to_{batch[-1]['index']:03d}.wav"
    
    # 返回合并后的段信息
    return {
        'file': merged_file,
        'start': batch[0]['start'],
        'end': batch[-1]['end'],
        'index': first_segment['index'],
        'batch_count': len(batch),
        'original_segments': batch
    }
```

#### 2.2 在转录流程中使用批量处理
```python
# 在 transcribe_segments 方法中，第540行附近
if self.model_type == "llm" and segment_count > 20:
    # 对大量分段使用批量处理
    segments = self.batch_audio_segments(segments, segments_dir, batch_size=4)
    self.beautifier.print_info(f"批量处理优化：{len(segments)} 个批次（原 {segment_count} 段）")
```

### 步骤3：VAD参数优化
**目标：减少音频碎片化，提高处理效率**

#### 3.1 修改默认VAD参数
```python
# long_video_transcribe.py 第53-55行
self.min_speech_duration_ms = 2000  # 原1000，现2000
self.max_speech_duration_s = 45     # 原30，现45
self.min_silence_duration_ms = 800  # 原500，现800
```

#### 3.2 添加场景自适应配置
```python
# 在 __init__ 方法中添加
def set_vad_params_for_scenario(self, scenario="general"):
    """根据场景设置VAD参数"""
    scenarios = {
        "podcast": {  # 播客/访谈
            "min_speech_duration_ms": 3000,
            "max_speech_duration_s": 60,
            "min_silence_duration_ms": 1000
        },
        "lecture": {  # 讲座/教学
            "min_speech_duration_ms": 2500,
            "max_speech_duration_s": 90,
            "min_silence_duration_ms": 1200
        },
        "general": {  # 通用场景
            "min_speech_duration_ms": 2000,
            "max_speech_duration_s": 45,
            "min_silence_duration_ms": 800
        }
    }
    
    params = scenarios.get(scenario, scenarios["general"])
    self.min_speech_duration_ms = params["min_speech_duration_ms"]
    self.max_speech_duration_s = params["max_speech_duration_s"]
    self.min_silence_duration_ms = params["min_silence_duration_ms"]
```

### 步骤4：性能监控
**目标：实时了解优化效果**

#### 4.1 添加性能统计
```python
# 在 transcribe_segments 方法开始处添加
start_time = time.time()
segment_times = []

# 在每个段处理完成后
segment_times.append(time.time() - segment_start)

# 在方法结束时
total_time = time.time() - start_time
avg_time = sum(segment_times) / len(segment_times) if segment_times else 0
self.beautifier.print_performance_stats({
    "总耗时": f"{total_time:.1f}秒",
    "平均每段": f"{avg_time:.1f}秒",
    "处理速度": f"{len(segments)/total_time:.1f}段/秒",
    "并行效率": f"{(sum(segment_times)/total_time)*100:.1f}%"
})
```

## 原子提交计划

### Commit 1: feat: 实现动态并行度调整
```bash
git add fireredasr/utils/cpu_optimization_config.py
git add long_video_transcribe.py
git commit -m "feat: 实现基于任务规模的动态并行度调整

- 根据音频段数量动态设置工作线程数(4-12)
- 优化预读取段数配置
- 充分利用i9-14900KF的32线程能力"
```

### Commit 2: feat: 添加音频段批量处理功能
```bash
git add long_video_transcribe.py
git commit -m "feat: 添加音频段批量处理以减少模型调用

- 实现batch_audio_segments方法
- 合并相邻短音频段(最多4段)
- 保持总时长不超过30秒"
```

### Commit 3: feat: 优化VAD参数配置
```bash
git add long_video_transcribe.py
git commit -m "feat: 优化VAD参数减少音频碎片化

- 最小语音段从1秒提升到2秒
- 最大语音段从30秒提升到45秒
- 添加场景自适应VAD配置"
```

### Commit 4: feat: 添加性能监控统计
```bash
git add long_video_transcribe.py
git add utils/terminal_beautifier.py
git commit -m "feat: 添加转写性能实时监控

- 统计总耗时和平均处理时间
- 显示处理速度和并行效率
- 优化终端输出展示"
```

## 测试验证

### 1. 单元测试
```python
# test_batch_processing.py
def test_batch_audio_segments():
    # 测试批量合并逻辑
    pass

def test_dynamic_config():
    # 测试动态配置生成
    pass
```

### 2. 性能基准测试
```bash
# 优化前
time python long_video_transcribe.py --test

# 优化后
time python long_video_transcribe.py --test --optimized
```

### 3. 稳定性测试
- 测试不同长度视频（5分钟、30分钟、2小时）
- 测试不同场景（访谈、讲座、日常对话）
- 验证内存使用是否稳定

## 注意事项

1. **内存管理**
   - 批量处理时注意内存占用
   - 大文件处理时启用垃圾回收

2. **错误处理**
   - 批量处理失败时的回退策略
   - 单个段失败不影响整体

3. **向后兼容**
   - 保留原有参数接口
   - 通过环境变量控制新特性

## 预期效果

| 优化项 | 性能提升 | 说明 |
|--------|----------|------|
| 动态并行度 | 2-3倍 | 从2线程提升到4-12线程 |
| 批量处理 | 1.5倍 | 减少50%的模型调用 |
| VAD优化 | 1.2倍 | 减少20%的音频段数 |
| **综合提升** | **3-5倍** | 多项优化叠加效果 |

## 后续优化建议

如果需要进一步提升性能，可考虑：
1. 使用多进程替代多线程（方案三）
2. 实现模型量化（INT8）
3. 引入ONNX Runtime加速

---

**实施要点：每步独立提交，确保可回滚，注重代码简洁性**