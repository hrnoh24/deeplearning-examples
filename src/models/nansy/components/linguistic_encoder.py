from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGLU(nn.Module):
  def __init__(self, in_channels: int, kernels: int, dropout: float) -> None:
    """Initializer.

    Args:
        in_channels (int): size of input channels
        kernels (int): size of convolution kernels
        dropout (float): dropout rate
    """
    super().__init__()
    self.conv = nn.Sequential(
      nn.Dropout(dropout),
      nn.Conv1d(in_channels, in_channels * 2, kernels, padding=kernels // 2),
      nn.GLU(dim=1)
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """transform the inputs.

    Args:
        inputs (torch.Tensor): [torch.float32; [B, C, N]], input 1D Tensor

    Returns:
        torch.Tensor: [torch.float32, [B, C, N]], transformed
    """
    return inputs + self.conv(inputs)


class LinguisticEncoder(nn.Module):
    """Additional linguistic information encoder.
    """

    def __init__(self,
                 in_channels: int,
                 hiddens: int,
                 preconv: int,
                 kernels: List[int],
                 leak: float,
                 dropout: float):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            hiddens: size of the hidden channes.
            preconv: the number of the pre-convolution blocks.
            kernels: size of the ConvGLU kernels.
            leak: negative slope of leaky relu.
            dropout: dropout rate.
        """
        super().__init__()
        # in_channels=1024, hiddens=128, preconv=2
        # unknown `leak`, `dropout`
        self.preconv = nn.Sequential(
            nn.Conv1d(in_channels, hiddens, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(dropout),
            *[
                nn.Sequential(
                    nn.Conv1d(hiddens, hiddens, 1),
                    nn.LeakyReLU(leak),
                    nn.Dropout(dropout))
                for _ in range(preconv - 1)])
        # kernels=[3] * 8 + [1] * 2
        self.convglu = nn.Sequential(*[
            ConvGLU(hiddens, kernel, dropout)
            for kernel in kernels])

        self.proj = nn.Conv1d(hiddens, hiddens, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Filter the linguistic informations.
        Args:
            inputs: [torch.float32; [B, in_channels, N]], input features.
        Returns:
            [torch.float32; [B, hiddens, N]], linguistic informations.
        """
        # [B, hiddens, N]
        x = self.preconv(inputs)
        # [B, hiddens, N]
        x = self.convglu(x)
        # [B, hiddens, N]
        return F.normalize(self.proj(x), dim=1)


if __name__ == "__main__":
  import hydra
  import omegaconf
  import pyrootutils

  root = pyrootutils.setup_root(__file__, pythonpath=True)
  cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "nansy.yaml")
  linguistic_encoder = hydra.utils.instantiate(cfg.linguistic_encoder)

  x = torch.rand(4, 1024, 50) # [B, C, N]
  x_lin = linguistic_encoder(x)
  print(x_lin.size())