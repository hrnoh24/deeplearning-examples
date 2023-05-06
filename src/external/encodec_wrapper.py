# for test
import sys
sys.path.append("/Users/hrnoh/Documents/github/deeplearning-examples/")

from pathlib import Path
from encodec import EncodecModel

from einops import rearrange
import torchaudio
import torch
from torch import Tensor

from src.utils.audio_utils import AudioUtils

class EncodecWrapper:
  def __init__(self, device="cpu") -> None:
      self.model = None
      self.device = device
      self._load_model()

  def _load_model(self):
      self.model = EncodecModel.encodec_model_48khz()
      self.model.set_target_bandwidth(24.0)
      self.model.to(self.device)

  @torch.inference_mode()
  def decode(self, codes: Tensor):
      """
      Args:
          codes: (b q t)
      """
      assert codes.dim() == 3
      return self.model.decode([(codes, None)]), self.model.sample_rate

  def decode_to_file(self, resps: Tensor, path: Path):
      assert resps.dim() == 2, f"Require shape (t q), but got {resps.shape}."
      resps = rearrange(resps, "t q -> 1 q t")
      wavs, sr = self.decode(resps)
      AudioUtils.save_audio(str(path), wavs.cpu()[0, 0], sr)

  def _replace_file_extension(self, path, suffix):
      return (path.parent / path.name.split(".")[0]).with_suffix(suffix)

  @torch.inference_mode()
  def encode(self, wav: Tensor, sr: int):
      """
      Args:
          wav: (c, t)
          sr: int

      Return:
          codes: (b, q, t)
      """
      wav = wav.unsqueeze(0)
      wav = AudioUtils.convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
      wav = wav.to(self.device)
      encoded_frames = self.model.encode(wav)
      qnt = torch.cat([encoded[0]
                      for encoded in encoded_frames], dim=-1)  # (b q t)
      return qnt

  def encode_from_file(self, path):
      wav, sr = torchaudio.load(str(path))
      if wav.shape[0] == 2:
          wav = wav[:1]
      return self.encode(wav, sr)

if __name__ == "__main__":
  encodec = EncodecWrapper(device="cpu")
  wav_path = "/Users/hrnoh/Documents/dev/deeplearning/datasets/KSS/kss/1/1_0173.wav"
  codes = encodec.encode_from_file(wav_path)
  print(codes.size())
  encodec.decode_to_file(codes.squeeze(0).t(), "encodec_test.wav")
  print(codes)
  