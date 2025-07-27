#!/usr/bin/env python3
"""
并行处理器模块
- 温和的多线程音频处理
- 智能负载均衡
- 资源监控和自适应调整
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from typing import List, Dict, Callable, Any
import psutil
from pathlib import Path


class ParallelProcessor:
    """并行处理器"""
    
    def __init__(self, max_workers: int = 8, resource_check_interval: int = 5):
        self.max_workers = max_workers
        self.resource_check_interval = resource_check_interval
        self.executor = None
        self.running = False
        self.processed_count = 0
        self.total_count = 0
        self.start_time = None
        self.resource_monitor_thread = None
        self.should_throttle = False
        
        print(f"🔧 初始化并行处理器: {max_workers} 线程")
    
    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.running = True
        self.start_time = time.time()
        
        # 启动资源监控线程
        self.resource_monitor_thread = threading.Thread(
            target=self._monitor_resources, 
            daemon=True
        )
        self.resource_monitor_thread.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.resource_monitor_thread:
            self.resource_monitor_thread.join(timeout=1)
    
    def _monitor_resources(self):
        """监控系统资源使用情况"""
        while self.running:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # 如果 CPU 或内存使用率过高，启用节流
                if cpu_percent > 80 or memory_percent > 85:
                    if not self.should_throttle:
                        print(f"⚠️ 系统负载过高 (CPU: {cpu_percent:.1f}%, 内存: {memory_percent:.1f}%)，启用节流模式")
                        self.should_throttle = True
                else:
                    if self.should_throttle:
                        print("✅ 系统负载正常，关闭节流模式")
                        self.should_throttle = False
                
                # 节流等待
                if self.should_throttle:
                    time.sleep(2)
                else:
                    time.sleep(self.resource_check_interval)
                    
            except Exception as e:
                print(f"⚠️ 资源监控出错: {e}")
                time.sleep(self.resource_check_interval)
    
    def process_batch(self, 
                     items: List[Any], 
                     process_func: Callable,
                     description: str = "处理",
                     batch_size: int = None) -> List[Any]:
        """批量处理任务"""
        
        if not items:
            return []
        
        self.total_count = len(items)
        self.processed_count = 0
        results = []
        
        print(f"🚀 开始{description}: {self.total_count} 个任务，{self.max_workers} 线程")
        
        # 如果指定了批次大小，分批处理
        if batch_size and batch_size < len(items):
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            
            for batch_idx, batch in enumerate(batches, 1):
                print(f"📦 处理第 {batch_idx}/{len(batches)} 批次 ({len(batch)} 个任务)")
                batch_results = self._process_single_batch(batch, process_func, description)
                results.extend(batch_results)
                
                # 批次间暂停，避免系统过载
                if batch_idx < len(batches):
                    time.sleep(1)
        else:
            results = self._process_single_batch(items, process_func, description)
        
        elapsed = time.time() - self.start_time
        print(f"✅ {description}完成: {len(results)}/{self.total_count} 成功 (耗时: {elapsed:.2f}s)")
        
        return results
    
    def _process_single_batch(self, items: List[Any], process_func: Callable, description: str) -> List[Any]:
        """处理单个批次"""
        futures = {}
        results = []
        
        # 判断是否是音频转录任务（需要特殊处理）
        is_audio_task = "音频转录" in description
        
        # 提交任务
        for i, item in enumerate(items):
            # 如果系统负载过高，等待
            while self.should_throttle:
                time.sleep(0.5)
            
            # 对于音频转录任务，在提交之间增加小延迟，避免瞬时压力
            if is_audio_task and self.max_workers > 1 and i > 0:
                time.sleep(0.1)  # 100ms延迟
            
            future = self.executor.submit(self._safe_process, process_func, item)
            futures[future] = item
        
        # 收集结果
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                
                self.processed_count += 1
                self._print_progress(description)
                
                # 对于音频转录，每处理完一个就清理一次内存
                if is_audio_task and self.processed_count % 5 == 0:
                    import gc
                    gc.collect()
                
            except Exception as e:
                item = futures[future]
                print(f"❌ 处理失败: {item} - {str(e)}")
        
        return results
    
    def _safe_process(self, process_func: Callable, item: Any) -> Any:
        """安全处理单个任务"""
        try:
            return process_func(item)
        except Exception as e:
            print(f"❌ 任务处理异常: {str(e)}")
            return None
    
    def _print_progress(self, description: str):
        """打印进度信息"""
        if self.total_count > 0:
            progress = (self.processed_count / self.total_count) * 100
            elapsed = time.time() - self.start_time
            
            # 每 10% 或每 10 个任务打印一次进度
            if self.processed_count % max(1, self.total_count // 10) == 0 or self.processed_count % 10 == 0:
                avg_time = elapsed / self.processed_count if self.processed_count > 0 else 0
                remaining = (self.total_count - self.processed_count) * avg_time
                
                print(f"📊 {description}进度: {self.processed_count}/{self.total_count} "
                      f"({progress:.1f}%) - 剩余: {remaining:.1f}s")


class AudioBatchProcessor:
    """音频批处理器"""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
    
    def process_audio_segments(self, 
                             segments: List[Path], 
                             transcribe_func: Callable,
                             batch_size: int = 2,
                             model_type: str = 'aed') -> List[Dict]:
        """并行处理音频片段（支持LLM模型优化）"""
        
        def process_segment(segment_path):
            """处理单个音频片段"""
            try:
                start_time = time.time()
                
                # 调用转录函数
                result = transcribe_func(segment_path)
                
                process_time = time.time() - start_time
                
                if result:
                    if 'process_time' not in result:
                        result['process_time'] = process_time
                    return result
                else:
                    print(f"⚠️ 转录失败: {segment_path}")
                    return None
                    
            except Exception as e:
                print(f"❌ 处理音频片段失败 {segment_path}: {str(e)}")
                return None
        
        # 对于LLM模型，增加批次间的延迟
        if model_type == 'llm' and self.max_workers > 1:
            print("🔧 LLM模型并行优化：增加批次间延迟，减少内存压力")
        
        # 使用并行处理器
        with ParallelProcessor(max_workers=self.max_workers) as processor:
            # 如果是LLM模型且使用并行，减小批次大小
            if model_type == 'llm' and self.max_workers > 1:
                batch_size = 1  # LLM模型每次只处理一个
                
            results = processor.process_batch(
                segments, 
                process_segment, 
                description="音频转录",
                batch_size=batch_size
            )
        
        # 过滤有效结果
        valid_results = [r for r in results if r is not None]
        
        if valid_results:
            total_audio_time = sum(r.get('duration', 0) for r in valid_results)
            total_process_time = sum(r.get('process_time', 0) for r in valid_results)
            avg_rtf = total_process_time / total_audio_time if total_audio_time > 0 else 0
            
            print(f"📈 并行处理统计:")
            print(f"   音频总时长: {total_audio_time:.2f}s")
            print(f"   处理总时间: {total_process_time:.2f}s")
            print(f"   平均实时因子: {avg_rtf:.4f}")
            print(f"   并行加速比: {total_audio_time / total_process_time:.2f}x")
        
        return valid_results


def test_parallel_processor():
    """测试并行处理器"""
    import random
    
    def dummy_task(item):
        # 模拟处理任务
        time.sleep(random.uniform(0.1, 0.5))
        return f"处理完成: {item}"
    
    test_items = [f"任务_{i}" for i in range(20)]
    
    with ParallelProcessor(max_workers=4) as processor:
        results = processor.process_batch(test_items, dummy_task, "测试任务")
    
    print(f"测试结果: {len(results)} 个任务完成")


if __name__ == "__main__":
    test_parallel_processor()