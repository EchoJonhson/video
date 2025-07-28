#!/usr/bin/env python3
"""
硬件资源管理模块
- GPU 显存检测
- CPU 核心数检测
- 智能资源分配策略
"""

import os
import psutil
import torch
from typing import Dict, Tuple


class HardwareManager:
    """硬件资源管理器"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_info = self._detect_gpu()
        self.strategy = self._select_strategy()
    
    def _detect_gpu(self) -> Dict:
        """检测 GPU 信息"""
        gpu_info = {
            'available': False,
            'memory_gb': 0,
            'device_name': 'None'
        }
        
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                gpu_info['available'] = True
                gpu_info['memory_gb'] = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                gpu_info['device_name'] = torch.cuda.get_device_name(device)
                
                # 获取当前可用显存（减去已占用的）
                torch.cuda.empty_cache()
                available_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                gpu_info['available_gb'] = (available_memory - allocated_memory) / (1024**3)
                
            except Exception as e:
                print(f"⚠️ GPU 检测失败: {e}")
                
        return gpu_info
    
    def _select_strategy(self) -> Dict:
        """智能选择处理策略"""
        # 默认策略：CPU为主，温和使用资源
        strategy = {
            'name': 'cpu_primary',
            'use_gpu': False,
            'cpu_threads': min(16, max(8, self.cpu_count - 2)),  # 使用更多CPU核心，但保留2核给系统
            'batch_size': 1,
            'gpu_batch_size': 1,
            'memory_management': 'balanced'
        }
        
        if self.gpu_info['available']:
            gpu_memory = self.gpu_info.get('available_gb', 0)
            
            # 即使有GPU，也优先使用CPU，GPU只作为辅助
            if gpu_memory >= 8:
                # 大显存：CPU为主，GPU加速关键部分
                strategy.update({
                    'name': 'cpu_primary_gpu_assist',
                    'use_gpu': True,
                    'cpu_threads': min(16, max(12, self.cpu_count - 2)),
                    'batch_size': 2,
                    'gpu_batch_size': 1,
                    'memory_management': 'cpu_optimized',
                    'gpu_role': 'encoder_only'  # GPU仅用于编码器
                })
            elif gpu_memory >= 4:
                # 中等显存：CPU处理主体，GPU处理小部分
                strategy.update({
                    'name': 'cpu_primary_minimal_gpu',
                    'use_gpu': True,
                    'cpu_threads': min(16, max(10, self.cpu_count - 2)),
                    'batch_size': 1,
                    'gpu_batch_size': 1,
                    'memory_management': 'cpu_focused',
                    'gpu_role': 'feature_extraction'  # GPU仅用于特征提取
                })
            elif gpu_memory >= 2:
                # 小显存：主要依赖CPU
                strategy.update({
                    'name': 'cpu_dominant',
                    'use_gpu': False,  # 小显存时不使用GPU，避免OOM
                    'cpu_threads': min(16, max(8, self.cpu_count - 2)),
                    'batch_size': 1,
                    'gpu_batch_size': 0,
                    'memory_management': 'cpu_only'
                })
        
        return strategy
    
    def get_optimal_config(self) -> Dict:
        """获取优化配置"""
        return {
            'hardware': {
                'cpu_cores': self.cpu_count,
                'memory_gb': self.memory_gb,
                'gpu_available': self.gpu_info['available'],
                'gpu_memory_gb': self.gpu_info.get('memory_gb', 0),
                'gpu_available_gb': self.gpu_info.get('available_gb', 0),
                'gpu_name': self.gpu_info.get('device_name', 'None')
            },
            'strategy': self.strategy,
            'limits': {
                'max_cpu_usage': 80,  # 最大 CPU 使用率
                'max_memory_usage': 85,  # 最大内存使用率
                'max_gpu_memory_usage': 90  # 最大显存使用率
            }
        }
    
    def monitor_resources(self) -> Dict:
        """监控当前资源使用情况"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        status = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available_gb': memory.available / (1024**3)
        }
        
        if self.gpu_info['available']:
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = self.gpu_info['memory_gb']
                status['gpu_memory_usage'] = (gpu_memory_used / gpu_memory_total) * 100
                status['gpu_memory_used_gb'] = gpu_memory_used
            except:
                status['gpu_memory_usage'] = 0
                status['gpu_memory_used_gb'] = 0
        
        return status
    
    def should_reduce_load(self) -> bool:
        """检查是否需要降低负载"""
        status = self.monitor_resources()
        
        # CPU 使用率过高
        if status['cpu_usage'] > 85:
            return True
        
        # 内存使用率过高
        if status['memory_usage'] > 90:
            return True
        
        # GPU 显存使用率过高
        if self.gpu_info['available'] and status.get('gpu_memory_usage', 0) > 95:
            return True
        
        return False
    
    def print_hardware_info(self):
        """打印硬件信息"""
        print("🖥️ 硬件配置检测:")
        print(f"   CPU: {self.cpu_count} 核心")
        print(f"   内存: {self.memory_gb:.1f} GB")
        
        if self.gpu_info['available']:
            print(f"   GPU: {self.gpu_info['device_name']}")
            print(f"   显存: {self.gpu_info['memory_gb']:.1f} GB (可用: {self.gpu_info.get('available_gb', 0):.1f} GB)")
        else:
            print("   GPU: 未检测到或不可用")
        
        print(f"\n⚙️ 选择策略: {self.strategy['name']}")
        print(f"   CPU 线程: {self.strategy['cpu_threads']}")
        print(f"   批处理大小: {self.strategy['batch_size']}")
        print(f"   使用 GPU: {'是' if self.strategy['use_gpu'] else '否'}")
        print(f"   内存管理: {self.strategy['memory_management']}")


def get_hardware_manager() -> HardwareManager:
    """获取硬件管理器实例"""
    return HardwareManager()


if __name__ == "__main__":
    # 测试硬件检测
    manager = HardwareManager()
    manager.print_hardware_info()
    
    print("\n📊 当前资源使用:")
    status = manager.monitor_resources()
    for key, value in status.items():
        print(f"   {key}: {value}")