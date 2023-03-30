from typing import Union
import torch
from torch import nn
import numpy as np

class ResBlock(nn.Module):
  def __init__(self, in_channel: int, out_channel: int, kernel_size: int) -> None:
    """CNN on frequency-axis

    Args:
        in_channel (int): input channel
        out_channel (int): output channel
        kernel_size (int): kernel size on frequency axis
    """
    super().__init__()
    self.shortcut = nn.Conv2d(in_channel, out_channel, 1)
    self.conv_layers = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.GELU(),
        nn.Conv2d(in_channel, out_channel, (1, kernel_size), stride=1, dilation=1, padding="same"),
        nn.BatchNorm2d(in_channel),
        nn.GELU(),
        nn.Conv2d(in_channel, out_channel, (1, kernel_size), stride=1, dilation=1, padding="same")
    )
    self.max_pool = nn.MaxPool2d((3, 1), (2, 1), padding=(1, 0))

  def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
    """_summary_

    Args:
        x (torch.FloatTensor): [B, C, F, N]

    Returns:
        torch.FloatTensor: _description_
    """
    residual = self.shortcut(x)
    outputs = self.conv_layers(x)

    x = self.max_pool(outputs) + self.max_pool(residual)
    return x
  
def exponential_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Exponential sigmoid.
    Args:
        x: [torch.float32; [...]], input tensors.
    Returns:
        sigmoid outputs.
    """
    return 2.0 * torch.sigmoid(x) ** np.log(10) + 1e-7

class PitchEncoder(nn.Module):
  def __init__(self, 
               freq: int,
               prekernels: int,
               kernels: int,
               channels: int,
               blocks: int,
               gru: int,
               hiddens: int,
               f0_bins: int) -> None:
    """Initializer.
    Args:
        freq: the number of the frequency bins.
        prekernels: size of the first convolutional kernels.
        kernels: size of the frequency-convolution kernels.
        channels: size of the channels.
        blocks: the number of the residual blocks.
        gru: size of the GRU hidden states.
        hiddens: size of the hidden channels.
        f0_bins: size of the output f0-bins.
    """
    super().__init__()
    self.f0_bins = f0_bins
    self.preconv = nn.Conv2d(1, channels, (prekernels, 1), padding=(prekernels // 2, 0))

    self.resblock = nn.Sequential(*[ResBlock(channels, channels, kernels) for _ in range(blocks)])

    self.gru = nn.GRU(freq * channels // (2 * blocks), gru, batch_first=True, bidirectional=True)
    self.proj = nn.Sequential(
      nn.Linear(gru * 2, hiddens * 2),
      nn.ReLU(),
      nn.Linear(hiddens * 2, f0_bins + 2)
    )

  def forward(self, x: torch.FloatTensor) -> Union[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the pitch from inputs.
      Args:
          inputs: [torch.float32; [B, F, N]], input tensor.
      Returns:
          f0: [torch.float32; [B, N, f0_bins]], f0 outputs, based on frequency bins.
          p_amp, ap_amp: [torch.float32; [B, N]], amplitude values.
      """
    bsize, _, timesteps = x.shape
    # [B, C, F, N]
    x = self.preconv(x[:, None])
    # [B, C F // 4, N]
    x = self.resblock(x)
    # [B, N, C x F // 4]
    x = x.permute(0, 3, 1, 2).reshape(bsize, timesteps, -1)
    # [B, N, G x 2]
    x, _ = self.gru(x)
    # [B, N, f0_bins], [B, N, 1], [B, N, 1]
    f0_prob, p_amp, ap_amp = torch.split(self.proj(x), [self.f0_bins, 1, 1], dim=-1)
    return \
        torch.softmax(f0_prob, dim=-1), \
        exponential_sigmoid(p_amp).squeeze(dim=-1), \
        exponential_sigmoid(ap_amp).squeeze(dim=-1)


if __name__=="__main__":
  import hydra
  import omegaconf
  import pyrootutils
  import matplotlib.pyplot as plt
  import soundfile as sf
  
  root = pyrootutils.setup_root(__file__, pythonpath=True)
  cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "nansy.yaml")
  pitch_encoder = hydra.utils.instantiate(cfg.pitch_encoder)
  cqt_layer = hydra.utils.instantiate(cfg.cqt_wrapper)

  # cqt related
  # cqt_layer = CQTWrapper(
  #   strides=256,
  #   fmin=32.7,
  #   bins=191,
  #   bins_per_octave=24,
  #   sr=16000
  # )
  n_bins = 191
  scope_size = 160
  cqt_center = (n_bins - scope_size) // 2

  wav, sr = sf.read("/root/data/KSS/kss/1/1_0173.wav")
  wav = torch.FloatTensor(wav).unsqueeze(0)
  print(wav.shape, sr)
  cqt = cqt_layer(wav)
  plt.imshow(cqt[0])
  plt.gca().invert_yaxis()
  plt.show()
  plt.savefig("cqt.png")
  print(cqt.size())

  f0_prob, p_amp, ap_amp = pitch_encoder(cqt[:, cqt_center:cqt_center+scope_size])
  print(f0_prob.size(), p_amp.size(), ap_amp.size())