#!/usr/bin/env python3
"""
测试CPU优化效果
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from fireredasr.utils.cpu_optimization_config import CPUOptimizationConfig

def test_optimization_config():
    """测试优化配置"""
    print("🧪 测试CPU优化配置\n")
    
    # 创建优化器
    optimizer = CPUOptimizationConfig()
    
    print(f"📊 系统信息:")
    print(f"   CPU核心数: {optimizer.physical_cores}")
    print(f"   逻辑线程数: {optimizer.cpu_count}")
    print(f"   总内存: {optimizer.total_memory_gb:.1f}GB")
    
    print("\n📈 LLM模型优化配置:")
    for segment_count in [5, 20, 50, 100, 200]:
        config = optimizer.get_dynamic_config(segment_count, "llm")
        memory_est = optimizer.estimate_memory_usage("llm", config["max_workers"])
        
        print(f"\n分段数: {segment_count}")
        print(f"  - 并行线程: {config['max_workers']}")
        print(f"  - 预读取段数: {config['memory_config']['prefetch_segments']}")
        print(f"  - 预估内存: {memory_est['total_gb']:.1f}GB ({memory_est['usage_percent']:.1f}%)")
    
    print("\n📈 AED模型优化配置:")
    for segment_count in [10, 50, 100]:
        config = optimizer.get_dynamic_config(segment_count, "aed")
        memory_est = optimizer.estimate_memory_usage("aed", config["max_workers"])
        
        print(f"\n分段数: {segment_count}")
        print(f"  - 并行线程: {config['max_workers']}")
        print(f"  - 批处理大小: {config['batch_size']}")
        print(f"  - 预估内存: {memory_est['total_gb']:.1f}GB ({memory_est['usage_percent']:.1f}%)")

def benchmark_parallel_vs_serial():
    """对比串行和并行处理的性能"""
    print("\n\n🏃 性能对比测试")
    print("=" * 60)
    
    # 模拟音频处理任务
    def simulate_audio_processing(duration=0.1):
        """模拟音频处理（CPU密集型）"""
        start = time.time()
        # 模拟CPU密集计算
        result = 0
        for i in range(int(1e6)):
            result += i ** 0.5
        time.sleep(duration)
        return time.time() - start
    
    segment_counts = [20, 50]
    
    for count in segment_counts:
        print(f"\n处理 {count} 个音频段:")
        
        # 串行处理
        start = time.time()
        for _ in range(count):
            simulate_audio_processing(0.05)
        serial_time = time.time() - start
        print(f"  串行处理: {serial_time:.2f}秒")
        
        # 并行处理（模拟4线程）
        from concurrent.futures import ThreadPoolExecutor
        start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(simulate_audio_processing, 0.05) for _ in range(count)]
            for future in futures:
                future.result()
        parallel_time = time.time() - start
        print(f"  并行处理(4线程): {parallel_time:.2f}秒")
        print(f"  加速比: {serial_time/parallel_time:.2f}x")

def main():
    """主函数"""
    print("🔥 FireRedASR CPU优化测试")
    print("=" * 60)
    
    # 测试优化配置
    test_optimization_config()
    
    # 性能对比
    benchmark_parallel_vs_serial()
    
    print("\n\n✅ 测试完成！")
    print("\n💡 优化建议:")
    print("1. LLM模型在GPU辅助模式下可以使用2-5个并行线程")
    print("2. 预读取功能可以减少IO等待时间")
    print("3. 合理的并行度可以获得2-3倍的性能提升")

if __name__ == "__main__":
    main()