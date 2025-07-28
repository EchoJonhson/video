#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文标点符号恢复模块
使用 zh-wiki-punctuation-restore 模型为中文文本添加标点符号
支持的标点：， 、 。 ？ ！ ；
"""

import os
import logging
import torch
from typing import List, Optional, Tuple
from transformers import AutoModelForTokenClassification, AutoTokenizer
import warnings
import gc

# 屏蔽一些不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class PunctuationRestorer:
    """中文标点符号恢复器"""
    
    def __init__(self, 
                 model_name: str = "p208p2002/zh-wiki-punctuation-restore",
                 device: Optional[str] = None,
                 batch_size: int = 8,
                 chunk_size: int = 256,
                 stride: int = 128,
                 cache_dir: Optional[str] = None):
        """
        初始化标点恢复器
        
        Args:
            model_name: 模型名称或路径
            device: 设备类型 ('cuda', 'cpu' 或 None 自动选择)
            batch_size: 批处理大小
            chunk_size: 文本块大小
            stride: 滑动窗口步长
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.stride = stride
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/transformers/")
        
        # 自动选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.tokenizer = None
        self.label_map = {
            0: '',      # LABEL_0: 无标点
            1: '，',    # LABEL_1: 逗号
            2: '。',    # LABEL_2: 句号
            3: '？',    # LABEL_3: 问号
            4: '！',    # LABEL_4: 感叹号
            5: '、',    # LABEL_5: 顿号
            6: '；'     # LABEL_6: 分号
        }
        
        logger.info(f"初始化标点恢复器 - 设备: {self.device}, 批大小: {batch_size}")
        
    def load_model(self) -> bool:
        """
        加载模型和分词器
        
        Returns:
            bool: 加载成功返回 True，失败返回 False
        """
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # 将模型移到指定设备
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False
            
    def _prepare_text_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """
        将长文本分割成重叠的文本块
        
        Args:
            text: 输入文本
            
        Returns:
            List[Tuple[str, int, int]]: [(文本块, 起始位置, 结束位置)]
        """
        chunks = []
        text_length = len(text)
        
        if text_length <= self.chunk_size:
            # 短文本直接处理
            chunks.append((text, 0, text_length))
        else:
            # 长文本使用滑动窗口
            start = 0
            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunk = text[start:end]
                chunks.append((chunk, start, end))
                
                # 如果已经到达文本末尾，停止
                if end >= text_length:
                    break
                    
                # 滑动窗口
                start += self.stride
                
        return chunks
        
    def _merge_predictions(self, chunks_results: List[Tuple[str, int, int]]) -> str:
        """
        合并重叠文本块的预测结果
        
        Args:
            chunks_results: [(带标点的文本块, 起始位置, 结束位置)]
            
        Returns:
            str: 合并后的完整文本
        """
        if not chunks_results:
            return ""
            
        if len(chunks_results) == 1:
            return chunks_results[0][0]
            
        # 按起始位置排序
        chunks_results.sort(key=lambda x: x[1])
        
        # 合并结果
        merged_text = chunks_results[0][0]
        last_end = chunks_results[0][2]
        
        for chunk_text, start, end in chunks_results[1:]:
            # 计算重叠区域
            overlap_size = last_end - start
            
            if overlap_size > 0:
                # 跳过重叠部分
                non_overlap_text = chunk_text[overlap_size:]
                merged_text += non_overlap_text
            else:
                # 没有重叠，直接拼接
                merged_text += chunk_text
                
            last_end = end
            
        return merged_text
        
    def _predict_chunk(self, text: str) -> str:
        """
        对单个文本块进行标点预测
        
        Args:
            text: 文本块
            
        Returns:
            str: 带标点的文本
        """
        if not text.strip():
            return text
            
        try:
            # 编码文本
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 将输入移到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
            # 解码预测结果
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            labels = predictions[0].cpu().numpy()
            
            # 构建带标点的文本
            result_chars = []
            for token, label in zip(tokens, labels):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                    
                # 处理子词
                if token.startswith('##'):
                    token = token[2:]
                    
                result_chars.append(token)
                
                # 添加标点
                if label > 0 and label in self.label_map:
                    result_chars.append(self.label_map[label])
                    
            return ''.join(result_chars)
            
        except Exception as e:
            logger.warning(f"文本块预测失败: {str(e)}")
            return text
            
    def restore_punctuation(self, text: str) -> str:
        """
        恢复文本的标点符号
        
        Args:
            text: 无标点的中文文本
            
        Returns:
            str: 带标点的文本
        """
        if not text or not text.strip():
            return text
            
        # 确保模型已加载
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                logger.error("模型加载失败，返回原始文本")
                return text
                
        try:
            # 准备文本块
            chunks = self._prepare_text_chunks(text)
            logger.info(f"文本长度: {len(text)}, 分成 {len(chunks)} 个块")
            
            # 处理每个文本块
            chunks_results = []
            for i, (chunk_text, start, end) in enumerate(chunks):
                logger.debug(f"处理第 {i+1}/{len(chunks)} 个文本块")
                punctuated_chunk = self._predict_chunk(chunk_text)
                chunks_results.append((punctuated_chunk, start, end))
                
            # 合并结果
            result = self._merge_predictions(chunks_results)
            
            logger.info("标点恢复完成")
            return result
            
        except Exception as e:
            logger.error(f"标点恢复失败: {str(e)}")
            return text
            
    def process_long_text(self, text: str, show_progress: bool = True) -> str:
        """
        处理长文本的标点恢复（带进度显示）
        
        Args:
            text: 长文本
            show_progress: 是否显示进度
            
        Returns:
            str: 带标点的文本
        """
        return self.restore_punctuation(text)
        
    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        # 强制垃圾回收
        gc.collect()
        
        # 如果使用 GPU，清理显存
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
        logger.info("资源清理完成")
        

def restore_punctuation_from_file(input_file: str, 
                                output_file: str,
                                model_name: str = "p208p2002/zh-wiki-punctuation-restore",
                                device: Optional[str] = None) -> bool:
    """
    从文件恢复标点的便捷函数
    
    Args:
        input_file: 输入文件路径（无标点文本）
        output_file: 输出文件路径（带标点文本）
        model_name: 模型名称
        device: 设备类型
        
    Returns:
        bool: 处理成功返回 True，失败返回 False
    """
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 创建恢复器并处理
        restorer = PunctuationRestorer(model_name=model_name, device=device)
        punctuated_text = restorer.restore_punctuation(text)
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(punctuated_text)
            
        # 清理资源
        restorer.cleanup()
        
        logger.info(f"标点恢复完成: {input_file} -> {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python punctuation_restore.py <输入文件> <输出文件>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 处理文件
    success = restore_punctuation_from_file(input_file, output_file)
    sys.exit(0 if success else 1)