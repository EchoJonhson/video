#!/usr/bin/env python3
"""
CPU优化配置 - 针对高端CPU（i9-14900KF）的优化建议
"""

import multiprocessing
import psutil
from typing import Dict, Any

class CPUOptimizationConfig:
    """CPU优化配置管理器"""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()  # 32线程
        self.physical_cores = psutil.cpu_count(logical=False)  # 24核心
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)  # 128GB
        
    def get_llm_optimization_config(self, model_size: str = "7B") -> Dict[str, Any]:
        """
        获取LLM模型的优化配置
        
        针对CPU为主、GPU为辅的场景优化
        """
        config = {
            "7B": {
                # 基础并行配置
                "max_workers": 4,  # 从2提升到4，充分利用高端CPU
                "batch_size": 1,   # 保持为1，避免内存峰值
                
                # CPU亲和性配置
                "cpu_affinity": {
                    "enable": True,
                    "encoder_cores": [0, 1, 2, 3],  # GPU编码器相关的核心
                    "llm_cores": list(range(4, 24)),  # LLM推理使用的核心
                    "io_cores": list(range(24, 32)),  # IO和辅助任务
                },
                
                # 内存优化
                "memory_config": {
                    "prefetch_segments": 8,  # 预读取的音频段数
                    "cache_size_mb": 4096,   # 4GB缓存
                    "gc_threshold": 0.7,     # 内存使用达70%时触发GC
                },
                
                # 线程池配置
                "thread_pool_config": {
                    "encoder_threads": 2,     # 编码器线程数
                    "llm_threads": 4,         # LLM推理线程数
                    "io_threads": 2,          # IO线程数
                    "prefetch_threads": 2,    # 预读取线程数
                },
                
                # 批处理优化
                "batch_optimization": {
                    "dynamic_batching": True,  # 动态批处理
                    "min_batch_wait_ms": 50,   # 最小等待时间
                    "max_batch_wait_ms": 200,  # 最大等待时间
                    "priority_queue": True,    # 优先级队列
                },
                
                # GPU辅助配置
                "gpu_assist": {
                    "encoder_gpu_id": 0,       # 编码器使用GPU 0
                    "feature_cache_gpu": True,  # GPU特征缓存
                    "async_transfer": True,     # 异步数据传输
                }
            }
        }
        
        return config.get(model_size, config["7B"])
    
    def get_aed_optimization_config(self) -> Dict[str, Any]:
        """AED模型的优化配置"""
        return {
            "max_workers": 8,  # AED可以使用更高并行度
            "batch_size": 4,   # 支持批处理
            
            "thread_pool_config": {
                "inference_threads": 8,
                "io_threads": 4,
            },
            
            "memory_config": {
                "prefetch_segments": 16,
                "cache_size_mb": 2048,
            }
        }
    
    def get_dynamic_config(self, segment_count: int, model_type: str) -> Dict[str, Any]:
        """
        根据任务动态调整配置
        """
        if model_type == "llm":
            base_config = self.get_llm_optimization_config()
            
            # 根据分段数动态调整
            if segment_count < 10:
                base_config["max_workers"] = 2
                base_config["memory_config"]["prefetch_segments"] = 4
            elif segment_count < 50:
                base_config["max_workers"] = 3
                base_config["memory_config"]["prefetch_segments"] = 6
            elif segment_count < 100:
                base_config["max_workers"] = 4
                base_config["memory_config"]["prefetch_segments"] = 8
            else:
                # 大量分段时的特殊优化
                base_config["max_workers"] = 5
                base_config["memory_config"]["prefetch_segments"] = 10
                base_config["batch_optimization"]["dynamic_batching"] = True
                
        else:  # AED
            base_config = self.get_aed_optimization_config()
            
            if segment_count < 20:
                base_config["max_workers"] = 4
            elif segment_count < 100:
                base_config["max_workers"] = 6
            else:
                base_config["max_workers"] = 8
                
        return base_config
    
    def estimate_memory_usage(self, model_type: str, parallel_workers: int) -> Dict[str, float]:
        """估算内存使用"""
        if model_type == "llm":
            # Qwen2-7B 估算
            model_base_gb = 14  # 模型基础内存
            per_worker_gb = 3   # 每个工作线程额外内存
            buffer_gb = 4       # 缓冲区
        else:
            # AED 模型
            model_base_gb = 2
            per_worker_gb = 0.5
            buffer_gb = 2
            
        total_gb = model_base_gb + (per_worker_gb * parallel_workers) + buffer_gb
        
        return {
            "model_base_gb": model_base_gb,
            "workers_gb": per_worker_gb * parallel_workers,
            "buffer_gb": buffer_gb,
            "total_gb": total_gb,
            "available_gb": self.total_memory_gb,
            "usage_percent": (total_gb / self.total_memory_gb) * 100
        }


# 使用示例
if __name__ == "__main__":
    optimizer = CPUOptimizationConfig()
    
    # LLM优化配置
    llm_config = optimizer.get_llm_optimization_config()
    print("LLM优化配置:")
    print(f"  最大并行数: {llm_config['max_workers']}")
    print(f"  预读取段数: {llm_config['memory_config']['prefetch_segments']}")
    
    # 内存估算
    memory_est = optimizer.estimate_memory_usage("llm", 4)
    print(f"\n内存使用估算 (4并行):")
    print(f"  总计: {memory_est['total_gb']:.1f}GB / {memory_est['available_gb']:.1f}GB")
    print(f"  使用率: {memory_est['usage_percent']:.1f}%")