import librosa
import soundfile as sf
import numpy as np
from typing import Tuple

import torch
import torchaudio

class AudioUtils:
  @staticmethod
  def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(
            f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

  @staticmethod
  def load_audio_torch(path: str) -> Tuple[torch.FloatTensor, int]:
    """load wav file to float tensor

    Args:
        path (str): wav file path

    Returns:
        Tuple[torch.FloatTensor, int]:
          torch.FloatTensor [1, T]: wav data
          int : sampling rate
    """
    wav, sr = torchaudio.load(path)
    return wav, sr

  @staticmethod
  def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """ load wav file to numpy array

    Args:
        path (str): wav file path

    Returns:
        np.ndarray [T,]: wav data
        int : sampling rate
    """
    wav, sr = sf.read(path)
    return wav, sr

  @staticmethod
  def save_audio(path: str, wav: np.ndarray, sr: int) -> str:
    sf.write(path, wav, sr)
    return path

  @staticmethod
  def save_audio_torch(path: str, wav: torch.FloatTensor, sr: int) -> str:
    """save audio to .wav file

    Args:
        path (str): output file path
        wav (torch.FloatTensor) [C, T]: wav data
        sr (int): sampling rate

    Returns:
        str: output file path
    """
    torchaudio.save(path, wav, sr)
    return path

  @staticmethod
  def normalize(wav: np.ndarray) -> np.ndarray:
    """ amplitude normalization

    Args:
        wav (np.ndarray) [T,]: wav file data

    Returns:
        np.ndarray [T,]: normalized audio
    """
    return librosa.util.normalize(wav)

  @staticmethod
  def trim_silence(wav: np.ndarray, 
                   frame_length=1024, 
                   hop_length=256, 
                   top_db=60) -> np.ndarray:
    """ trim audio silence

    Args:
        wav (np.ndarray): wav file data
        frame_length (int, optional): frame length. Defaults to 1024.
        hop_length (int, optional): hop length. Defaults to 256.
        top_db (int, optional): top db, 낮을수록 많이 잘림. Defaults to 60.

    Returns:
        np.ndarray [T,]: trimmed wav
    """
    return librosa.effects.trim(wav, 
                                top_db=top_db, 
                                frame_length=frame_length, 
                                hop_length=hop_length)[0]

  @staticmethod
  def to_mono(wav: np.ndarray) -> np.ndarray:
    """ multi-channel to mono by averaging

    Args:
        wav (np.ndarray) [T, N]: wav data

    Returns:
        np.ndarray [T,]: mono wav data
    """
    return librosa.to_mono(wav)

if __name__=="__main__":
  wav_path = "/Users/hrnoh/Documents/dev/deeplearning/datasets/KSS/kss/1/1_0173.wav"
  wav, sr = AudioUtils.load_audio_torch(wav_path)
  # print("Save :", AudioUtils.save_audio("test.wav", wav, sr))
  wav = AudioUtils.normalize(wav)
  wav = AudioUtils.to_mono(wav)
  wav = AudioUtils.trim_silence(wav, top_db=30)
  # print("Save :", AudioUtils.save_audio("test_trimmed.wav", wav, sr))
  # print(type(wav))
  # print(wav)