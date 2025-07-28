#!/usr/bin/env python3
"""
智能模型加载器
- 支持小显存的动态模型加载
- 内存映射和分层加载
- GPU/CPU 混合模式管理
"""

import os
import gc
import time
import torch
from pathlib import Path
from typing import Dict, Optional, Any
from contextlib import contextmanager

from fireredasr.models.fireredasr import FireRedAsr
from .hardware_manager import HardwareManager


class SmartModelLoader:
    """智能模型加载器"""
    
    def __init__(self, hardware_manager: HardwareManager):
        self.hardware_manager = hardware_manager
        self.config = hardware_manager.get_optimal_config()
        self.strategy = self.config['strategy']
        self.model = None
        self.model_parts = {}
        self.current_device = 'cpu'
        
    def load_model(self, model_type: str, model_dir: str) -> Optional[FireRedAsr]:
        """智能加载模型"""
        print(f"🤖 加载 {model_type.upper()} 模型...")
        print(f"📋 使用策略: {self.strategy['name']}")
        
        start_time = time.time()
        
        try:
            # 根据策略选择加载方式
            if self.strategy['name'] in ['cpu_primary', 'cpu_dominant', 'cpu_only']:
                self.model = self._load_cpu_only(model_type, model_dir)
            elif self.strategy['name'] == 'cpu_primary_gpu_assist':
                self.model = self._load_cpu_primary_gpu_assist(model_type, model_dir)
            elif self.strategy['name'] == 'cpu_primary_minimal_gpu':
                self.model = self._load_cpu_primary_minimal_gpu(model_type, model_dir)
            elif self.strategy['name'] == 'minimal_gpu':
                self.model = self._load_minimal_gpu(model_type, model_dir)
            elif self.strategy['name'] == 'smart_management':
                self.model = self._load_smart_management(model_type, model_dir)
            elif self.strategy['name'] == 'hybrid_large':
                self.model = self._load_hybrid_large(model_type, model_dir)
            else:
                # 默认 CPU 模式
                self.model = self._load_cpu_only(model_type, model_dir)
            
            load_time = time.time() - start_time
            print(f"✅ 模型加载完成 (耗时: {load_time:.2f}s)")
            
            self._print_memory_usage()
            return self.model
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            # 尝试降级到 CPU 模式
            print("🔄 尝试降级到 CPU 模式...")
            try:
                self.model = self._load_cpu_only(model_type, model_dir)
                print("✅ CPU 模式加载成功")
                return self.model
            except Exception as e2:
                print(f"❌ CPU 模式也失败: {str(e2)}")
                return None
    
    def _load_cpu_only(self, model_type: str, model_dir: str) -> FireRedAsr:
        """纯 CPU 模式加载"""
        print("🖥️ 纯 CPU 模式")
        
        # 确保在 CPU 上加载
        with torch.no_grad():
            model = FireRedAsr.from_pretrained(model_type, model_dir)
            model.model.cpu()
            
        self.current_device = 'cpu'
        return model
    
    def _load_minimal_gpu(self, model_type: str, model_dir: str) -> FireRedAsr:
        """最小 GPU 使用模式"""
        print("🎯 最小 GPU 模式")
        
        model = FireRedAsr.from_pretrained(model_type, model_dir)
        
        if model_type == "llm":
            # 仅编码器用 GPU，LLM 保持 CPU
            if hasattr(model.model, 'encoder'):
                model.model.encoder.cuda()
            if hasattr(model.model, 'encoder_projector'):
                model.model.encoder_projector.cuda()
            # LLM 保持在 CPU
            if hasattr(model.model, 'llm'):
                model.model.llm.cpu()
        else:
            # AED 模式全部 GPU
            model.model.cuda()
            
        self.current_device = 'cuda'
        return model
    
    def _load_smart_management(self, model_type: str, model_dir: str) -> FireRedAsr:
        """智能显存管理模式（方案 C）"""
        print("🧠 智能显存管理模式")
        
        model = FireRedAsr.from_pretrained(model_type, model_dir)
        
        # 获取当前可用显存
        available_memory = self._get_available_gpu_memory()
        print(f"💾 可用显存: {available_memory:.2f} GB")
        
        if model_type == "llm":
            # 分层加载策略
            if available_memory >= 6:
                # 足够显存：编码器 + 部分 LLM 层
                print("📈 高显存配置：编码器 + 部分 LLM 层在 GPU")
                self._smart_llm_placement(model, available_memory)
            elif available_memory >= 3:
                # 中等显存：仅编码器
                print("📊 中等显存配置：仅编码器在 GPU")
                if hasattr(model.model, 'encoder'):
                    model.model.encoder.cuda()
                if hasattr(model.model, 'encoder_projector'):
                    model.model.encoder_projector.cuda()
                if hasattr(model.model, 'llm'):
                    model.model.llm.cpu()
            else:
                # 低显存：全部 CPU
                print("📉 低显存配置：全部在 CPU")
                model.model.cpu()
        else:
            # AED 模式
            if available_memory >= 3:
                model.model.cuda()
            else:
                model.model.cpu()
        
        self.current_device = 'cuda' if available_memory >= 1 else 'cpu'
        return model
    
    def _load_hybrid_large(self, model_type: str, model_dir: str) -> FireRedAsr:
        """混合大显存模式"""
        print("🚀 混合大显存模式")
        
        model = FireRedAsr.from_pretrained(model_type, model_dir)
        model.model.cuda()  # 全部放 GPU
        
        self.current_device = 'cuda'
        return model
    
    def _load_cpu_primary_gpu_assist(self, model_type: str, model_dir: str) -> FireRedAsr:
        """CPU为主，GPU辅助模式（大显存）"""
        print("💻 CPU为主，GPU辅助模式")
        
        model = FireRedAsr.from_pretrained(model_type, model_dir)
        
        if model_type == "llm":
            # LLM 主体在 CPU，仅编码器在 GPU
            if hasattr(model.model, 'encoder'):
                try:
                    model.model.encoder.cuda()
                    print("✅ 编码器已放置到 GPU")
                except Exception as e:
                    print(f"⚠️ 编码器放置 GPU 失败: {e}")
                    model.model.encoder.cpu()
            
            # LLM 主体保持在 CPU
            if hasattr(model.model, 'llm'):
                model.model.llm.cpu()
                print("✅ LLM 主体保持在 CPU")
        else:
            # AED 模式，考虑部分放 GPU
            try:
                if hasattr(model.model, 'encoder'):
                    model.model.encoder.cuda()
                # 其他部分保持 CPU
                if hasattr(model.model, 'decoder'):
                    model.model.decoder.cpu()
            except:
                # 如果失败，全部回退到 CPU
                model.model.cpu()
        
        self.current_device = 'mixed'
        return model
    
    def _load_cpu_primary_minimal_gpu(self, model_type: str, model_dir: str) -> FireRedAsr:
        """CPU为主，最小GPU使用模式"""
        print("💻 CPU为主，最小GPU使用模式")
        
        model = FireRedAsr.from_pretrained(model_type, model_dir)
        
        # 全部在 CPU，只在需要时临时使用 GPU
        model.model.cpu()
        print("✅ 模型主体在 CPU，GPU 仅用于临时加速")
        
        self.current_device = 'cpu'
        return model
    
    def _smart_llm_placement(self, model, available_memory: float):
        """智能 LLM 层放置"""
        try:
            if hasattr(model.model, 'encoder'):
                model.model.encoder.cuda()
            if hasattr(model.model, 'encoder_projector'):
                model.model.encoder_projector.cuda()
            
            # 根据显存大小决定 LLM 层放置
            if hasattr(model.model, 'llm') and available_memory >= 8:
                # 尝试将部分 LLM 层放到 GPU
                try:
                    model.model.llm.cuda()
                    print("✅ LLM 层已放置到 GPU")
                except Exception as e:
                    print(f"⚠️ LLM 层放置 GPU 失败，回退到 CPU: {e}")
                    model.model.llm.cpu()
            else:
                model.model.llm.cpu()
                
        except Exception as e:
            print(f"⚠️ 智能放置失败: {e}")
            # 回退到保守策略
            model.model.cpu()
    
    def _get_available_gpu_memory(self) -> float:
        """获取可用 GPU 显存（GB）"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            torch.cuda.empty_cache()
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            available_memory = (total_memory - allocated_memory) / (1024**3)
            return max(0, available_memory - 0.5)  # 保留 0.5GB 缓冲
        except:
            return 0.0
    
    def _print_memory_usage(self):
        """打印内存使用情况"""
        print("\n💾 内存使用情况:")
        
        # 系统内存
        import psutil
        memory = psutil.virtual_memory()
        print(f"   系统内存: {memory.percent:.1f}% ({memory.used / (1024**3):.2f} GB / {memory.total / (1024**3):.2f} GB)")
        
        # GPU 显存
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                
                print(f"   GPU 显存: {allocated:.2f} GB 分配, {reserved:.2f} GB 保留, {total:.2f} GB 总计")
                print(f"   显存使用率: {(allocated / total) * 100:.1f}%")
            except:
                print("   GPU 显存: 检测失败")
    
    @contextmanager
    def temporary_gpu_mode(self):
        """临时 GPU 模式上下文管理器"""
        if not self.model or not torch.cuda.is_available():
            yield
            return
        
        original_device = self.current_device
        
        try:
            # 临时移动到 GPU
            if hasattr(self.model.model, 'encoder') and original_device == 'cpu':
                self.model.model.encoder.cuda()
                if hasattr(self.model.model, 'encoder_projector'):
                    self.model.model.encoder_projector.cuda()
            
            yield
            
        finally:
            # 恢复原始状态
            if original_device == 'cpu':
                if hasattr(self.model.model, 'encoder'):
                    self.model.model.encoder.cpu()
                if hasattr(self.model.model, 'encoder_projector'):
                    self.model.model.encoder_projector.cpu()
                torch.cuda.empty_cache()
    
    def optimize_for_inference(self):
        """为推理优化模型"""
        if not self.model:
            return
        
        print("⚡ 优化模型以进行推理...")
        
        # 设置评估模式
        self.model.model.eval()
        
        # 禁用梯度计算
        for param in self.model.model.parameters():
            param.requires_grad = False
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 垃圾回收
        gc.collect()
        
        print("✅ 模型优化完成")
    
    def get_transcribe_config(self) -> Dict:
        """获取转录配置"""
        base_config = {
            "use_gpu": self.strategy['use_gpu'] and torch.cuda.is_available(),
            "batch_size": self.strategy.get('batch_size', 1)
        }
        
        if hasattr(self.model, 'asr_type') and self.model.asr_type == "aed":
            base_config.update({
                "beam_size": 3,
                "nbest": 1,
                "decode_max_len": 0,
                "softmax_smoothing": 1.25,
                "aed_length_penalty": 0.6,
                "eos_penalty": 1.0
            })
        else:  # llm
            base_config.update({
                "beam_size": min(3, self.strategy.get('batch_size', 1)),  # 小显存时减少 beam size
                "decode_max_len": 0,
                "decode_min_len": 0,
                "repetition_penalty": 3.0,
                "llm_length_penalty": 1.0,
                "temperature": 1.0
            })
        
        return base_config


def create_smart_loader(hardware_manager: HardwareManager = None) -> SmartModelLoader:
    """创建智能模型加载器"""
    if hardware_manager is None:
        from .hardware_manager import get_hardware_manager
        hardware_manager = get_hardware_manager()
    
    return SmartModelLoader(hardware_manager)