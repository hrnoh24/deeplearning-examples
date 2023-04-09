from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGLU(nn.Module):
    """Dropout - Conv1d - GLU and conditional layer normalization.
    """

    def __init__(self,
                 channels: int,
                 kernels: int,
                 dilations: int,
                 dropout: float,
                 cond: Optional[int] = None):
        """Initializer.
        Args:
            channels: size of the input channels.
            kernels: size of the convolutional kernels.
            dilations: dilation rate of the convolution.
            dropout: dropout rate.
            cond: size of the condition channels, if provided.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels * 2, kernels, dilation=dilations,
                      padding=(kernels - 1) * dilations // 2),
            nn.GLU(dim=1))

        self.cond = cond
        if cond is not None:
            self.cond = nn.Conv1d(cond, channels * 2, 1)

    def forward(self, inputs: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """Transform the inputs with given conditions.
        Args:
            inputs: [torch.float32; [B, channels, T]], input channels.
            cond: [torch.float32; [B, cond, T]], if provided.
        Returns:
            [torch.float32; [B, channels, T]], transformed.
        """
        # [B, channels, T]
        x = inputs + self.conv(inputs)
        if cond is not None:
            assert self.cond is not None, 'condition module does not exists'
            # [B, channels, T]
            x = F.instance_norm(x, use_input_stats=True)
            # [B, channels, T]
            weight, bias = self.cond(cond).chunk(2, dim=1)
            # [B, channels, T]
            x = x * weight + bias
        return x


class CondSequential(nn.Module):
    """Sequential pass with conditional inputs.
    """

    def __init__(self, modules: List[nn.Module]):
        """Initializer.
        Args:
            modules: list of torch modules.
        """
        super().__init__()
        self.lists = nn.ModuleList(modules)

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Pass the inputs to modules.
        Args:
            inputs: arbitary input tensors.
            args, kwargs: positional, keyword arguments.
        Returns:
            output tensor.
        """
        x = inputs
        for module in self.lists:
            x = module.forward(x, *args, **kwargs)
        return x


class FrameLevelSynthesizer(nn.Module):
    """Frame-level synthesizer.
    """

    def __init__(self,
                 channels: int,
                 embed: int,
                 kernels: int,
                 dilations: List[int],
                 blocks: int,
                 leak: float,
                 dropout: float):
        """Initializer.
        Args:
            channels: size of the input channels.
            embed: size of the time-varying timbre embeddings.
            kernels: size of the convolutional kernels.
            dilations: dilation rates.
            blocks: the number of the 1x1 ConvGLU blocks after dilated ConvGLU.
            leak: negative slope of the leaky relu.
            dropout: dropout rates.
        """
        super().__init__()
        # channels=1024
        # unknown `leak`, `dropout`
        self.preconv = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(dropout))
        # kernels=3, dilations=[1, 3, 9, 27, 1, 3, 9, 27], blocks=2
        self.convglu = CondSequential(
            [
                ConvGLU(channels, kernels, dilation, dropout, cond=embed)
                for dilation in dilations]
            + [
                ConvGLU(channels, 1, 1, dropout, cond=embed)
                for _ in range(blocks)])

        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, inputs: torch.Tensor, timbre: torch.Tensor) -> torch.Tensor:
        """Synthesize in frame-level.
        Args:
            inputs: [torch.float32; [B, channels, T]], input features.
            timbre: [torch.float32; [B, embed, T]], time-varying timbre embeddings.
        Returns;
            [torch.float32; [B, channels, T]], outputs.
        """
        # [B, channels, T]
        x = self.preconv(inputs)
        # [B, channels, T]
        x = self.convglu(x, timbre)
        # [B, channels, T]
        return self.proj(x)

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "nansy.yaml")
    flv_synthesizer = hydra.utils.instantiate(cfg.frame_synthesizer)

    x_ling = torch.rand(4, 128, 50) # [B, C, N]
    timbre = torch.rand(4, 192, 50)
    frame_feats = flv_synthesizer(x_ling, timbre)
    print(frame_feats.size())