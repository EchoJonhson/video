import math
import os

import kaldiio
import kaldi_native_fbank as knf
import numpy as np
import torch

from fireredasr.utils.video_audio import (
    extract_audio_from_video, 
    is_video_file, 
    is_audio_file,
    TempFileManager
)


class ASRFeatExtractor:
    def __init__(self, kaldi_cmvn_file):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        self.fbank = KaldifeatFbank(num_mel_bins=80, frame_length=25,
            frame_shift=10, dither=0.0)
        self.temp_file_manager = TempFileManager()

    def __call__(self, input_paths):
        """
        处理音频或视频文件列表，提取特征
        
        Args:
            input_paths: 音频或视频文件路径列表
            
        Returns:
            feats_pad: 填充后的特征张量
            lengths: 每个样本的长度
            durs: 每个样本的时长
        """
        feats = []
        durs = []
        
        try:
            for input_path in input_paths:
                # 检查文件类型并处理
                if is_video_file(input_path):
                    # 如果是视频文件，先提取音频
                    wav_path = extract_audio_from_video(input_path, target_sr=16000)
                    self.temp_file_manager.add_temp_file(wav_path)
                elif is_audio_file(input_path) or input_path.endswith('.wav'):
                    # 如果是音频文件，直接使用
                    wav_path = input_path
                else:
                    # 尝试作为WAV文件处理（向后兼容）
                    wav_path = input_path
                
                # 加载音频数据
                sample_rate, wav_np = kaldiio.load_mat(wav_path)
                dur = wav_np.shape[0] / sample_rate
                
                # 提取fbank特征
                fbank = self.fbank((sample_rate, wav_np))
                if self.cmvn is not None:
                    fbank = self.cmvn(fbank)
                fbank = torch.from_numpy(fbank).float()
                
                feats.append(fbank)
                durs.append(dur)
                
        except Exception as e:
            # 如果出错，清理临时文件
            self.temp_file_manager.cleanup()
            raise e
            
        lengths = torch.tensor([feat.size(0) for feat in feats]).long()
        feats_pad = self.pad_feat(feats, 0.0)
        return feats_pad, lengths, durs
    
    def cleanup_temp_files(self):
        """手动清理临时文件"""
        self.temp_file_manager.cleanup()

    def pad_feat(self, xs, pad_value):
        # type: (List[Tensor], int) -> Tensor
        n_batch = len(xs)
        max_len = max([xs[i].size(0) for i in range(n_batch)])
        pad = torch.ones(n_batch, max_len, *xs[0].size()[1:]).to(xs[0].device).to(xs[0].dtype).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]
        return pad




class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = \
            self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variences = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            varience = (stats[1, d] / count) - mean*mean
            if varience < floor:
                varience = floor
            istd = 1.0 / math.sqrt(varience)
            inverse_std_variences.append(istd)
        return dim, np.array(means), np.array(inverse_std_variences)



class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10,
                 dither=1.0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav, is_train=False):
        if type(wav) is str:
            sample_rate, wav_np = kaldiio.load_mat(wav)
        elif type(wav) in [tuple, list] and len(wav) == 2:
            sample_rate, wav_np = wav
        assert len(wav_np.shape) == 1

        dither = self.dither if is_train else 0.0
        self.opts.frame_opts.dither = dither
        fbank = knf.OnlineFbank(self.opts)

        fbank.accept_waveform(sample_rate, wav_np.tolist())
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if len(feat) == 0:
            print("Check data, len(feat) == 0", wav, flush=True)
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack(feat)
        return feat
