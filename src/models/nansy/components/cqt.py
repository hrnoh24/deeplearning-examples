import torch
import torch.nn as nn

from nnAudio.features.cqt import CQT2010v2


class CQTWrapper(nn.Module):
    """Constant Q-Transform.
    """
    def __init__(self,
                 strides: int,
                 fmin: float,
                 bins: int,
                 bins_per_octave: int,
                 sr: int = 16000):
        """Initializer.
        Args:
            strides: the number of the samples between adjacent frame.
            fmin: frequency min.
            bins: the number of the output bins.
            bins_per_octave: the number of the frequency bins per octave.
            sr: sampling rate.
        """
        super().__init__()
        # unknown `strides`
        # , since linguistic information is 50fps, strides could be 441
        # fmin=32.7(C0)
        # bins=191, bins_per_octave=24
        # , fmax = 2 ** (bins / bins_per_octave) * fmin
        #        = 2 ** (191 / 24) * 32.7
        #        = 8132.89
        self.cqt = CQT2010v2(
            sr,
            strides,
            fmin,
            n_bins=bins,
            bins_per_octave=bins_per_octave,
            trainable=False,
            output_format='Magnitude')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply CQT on inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, bins, T / strides]], CQT magnitudes.
        """
        return self.cqt(inputs[:, None])