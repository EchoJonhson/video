import os
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip


def extract_audio_from_video(video_path, target_sr=16000, temp_dir=None):
    """
    从视频文件中提取音频并转换为指定采样率的单声道音频
    
    Args:
        video_path (str): 视频文件路径
        target_sr (int): 目标采样率，默认16000Hz
        temp_dir (str): 临时文件目录，默认为None使用系统临时目录
        
    Returns:
        str: 临时WAV文件路径
        
    Raises:
        ValueError: 如果视频文件不存在或无法处理
        RuntimeError: 如果音频提取失败
    """
    if not os.path.exists(video_path):
        raise ValueError(f"视频文件不存在: {video_path}")
    
    video_path = Path(video_path)
    if video_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
        raise ValueError(f"不支持的视频格式: {video_path.suffix}")
    
    try:
        # 使用moviepy提取音频
        with VideoFileClip(str(video_path)) as video:
            if video.audio is None:
                raise RuntimeError(f"视频文件中没有音频流: {video_path}")
            
            # 创建临时WAV文件
            if temp_dir is None:
                temp_dir = tempfile.gettempdir()
            
            temp_wav = tempfile.NamedTemporaryFile(
                suffix='.wav', 
                dir=temp_dir, 
                delete=False
            ).name
            
            # 提取音频到临时文件
            # 兼容不同版本的moviepy
            try:
                video.audio.write_audiofile(
                    temp_wav, 
                    verbose=False, 
                    logger=None,
                    temp_audiofile=None
                )
            except TypeError:
                # 旧版本moviepy不支持temp_audiofile参数
                video.audio.write_audiofile(
                    temp_wav, 
                    verbose=False, 
                    logger=None
                )
        
        # 使用librosa重新加载并确保格式正确
        audio_data, sr = librosa.load(temp_wav, sr=target_sr, mono=True)
        
        # 归一化音频数据到int16范围
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 重新保存为正确格式的WAV文件
        sf.write(temp_wav, audio_data, target_sr, subtype='PCM_16')
        
        return temp_wav
        
    except Exception as e:
        # 清理临时文件
        if 'temp_wav' in locals() and os.path.exists(temp_wav):
            try:
                os.unlink(temp_wav)
            except:
                pass
        raise RuntimeError(f"从视频中提取音频失败: {str(e)}")


def is_video_file(file_path):
    """
    检查文件是否为支持的视频格式
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        bool: 如果是支持的视频格式返回True
    """
    if not isinstance(file_path, (str, Path)):
        return False
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
    file_ext = Path(file_path).suffix.lower()
    return file_ext in video_extensions


def is_audio_file(file_path):
    """
    检查文件是否为支持的音频格式
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        bool: 如果是支持的音频格式返回True
    """
    if not isinstance(file_path, (str, Path)):
        return False
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    file_ext = Path(file_path).suffix.lower()
    return file_ext in audio_extensions


def cleanup_temp_files(temp_files):
    """
    清理临时文件列表
    
    Args:
        temp_files (list): 临时文件路径列表
    """
    if not isinstance(temp_files, list):
        temp_files = [temp_files]
    
    for temp_file in temp_files:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                warnings.warn(f"无法删除临时文件 {temp_file}: {str(e)}")


class TempFileManager:
    """临时文件管理器，确保在退出时清理临时文件"""
    
    def __init__(self):
        self.temp_files = []
    
    def add_temp_file(self, file_path):
        """添加临时文件到管理列表"""
        self.temp_files.append(file_path)
        return file_path
    
    def cleanup(self):
        """清理所有临时文件"""
        cleanup_temp_files(self.temp_files)
        self.temp_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()