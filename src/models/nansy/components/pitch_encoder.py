from typing import Union
from unicodedata import bidirectional
import torch
from torch import nn

class ResBlock(nn.Module):
  def __init__(self, in_channel: int, out_channel: int, kernel_size: int) -> None:
    """CNN on frequency-axis

    Args:
        in_channel (int): input channel
        out_channel (int): output channel
        kernel_size (int): kernel size on frequency axis
    """
    super().__init__()
    self.linear = nn.Linear(in_channel, out_channel)
    self.conv_layers = nn.ModuleList([
      nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.GELU(),
        nn.Conv2d(in_channel, out_channel, (1, kernel_size), stride=1, dilation=1, padding="same")
      ),
      nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.GELU(),
        nn.Conv2d(in_channel, out_channel, (1, kernel_size), stride=1, dilation=1, padding="same")
      ),
    ])
    self.max_pool = nn.MaxPool2d((3, 1), (2, 1), padding=(1, 0))

  def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
    """_summary_

    Args:
        x (torch.FloatTensor): [B, C, F, N]

    Returns:
        torch.FloatTensor: _description_
    """
    residual = self.linear(x.transpose(1, -1)).transpose(1, -1)

    for conv in self.conv_layers:
      x = conv(x)

    x = self.max_pool(x) + self.max_pool(residual)
    return x

class PitchEncoder(nn.Module):
  def __init__(self, 
               frequency_size: int, 
               hidden_channel: int, 
               init_conv_kernel_size: int, 
               resblock_kernel_size: int,
               num_f0_probs: int) -> None:
    super().__init__()
    self.init_conv = nn.Conv2d(1, hidden_channel, (init_conv_kernel_size, 1), padding="same")
    self.resblocks = nn.ModuleList([
      ResBlock(hidden_channel, hidden_channel, resblock_kernel_size) for _ in range(2)
    ])
    self.gru = nn.GRU(32 * frequency_size, hidden_channel, 1, batch_first=True, bidirectional=True)

    self.linear = nn.Sequential(
      nn.Linear(2 * hidden_channel, 2 * hidden_channel),
      nn.ReLU()
    )
    self.f0_head = nn.Linear(2 * hidden_channel, num_f0_probs)
    self.p_amp_head = nn.Linear(2 * hidden_channel, 1)
    self.ap_amp_head = nn.Linear(2 * hidden_channel, 1)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x: torch.FloatTensor) -> Union[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """_summary_

    Args:
        x (torch.FloatTensor): [B, 1, T]

    Returns:
        Union[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]: _description_
    """
    # Todo : CQT 추가해야 함
    B, _, _, N = x.size()

    x = self.init_conv(x)
    for resblock in self.resblocks:
      x = resblock(x)

    x = x.view(B, -1, N).transpose(1, 2)
    x, _ = self.gru(x)
    x = self.linear(x)

    f0_prob = self.softmax(self.f0_head(x).transpose(1, 2))
    p_amp = self.p_amp_head(x).transpose(1, 2)
    ap_amp = self.ap_amp_head(x).transpose(1, 2)

    # Todo : Exp.Sigmoid 구현해야 함
    return f0_prob, p_amp, ap_amp

if __name__=="__main__":
  import hydra
  import omegaconf
  import pyrootutils
  
  root = pyrootutils.setup_root(__file__, pythonpath=True)
  cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "nansy.yaml")
  pitch_encoder = hydra.utils.instantiate(cfg.pitch_encoder)

  cqt = torch.rand(4, 1, 160, 5)
  pitch_encoder(cqt)