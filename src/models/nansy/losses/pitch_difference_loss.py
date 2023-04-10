import torch
import torch.nn as nn
import torch.nn.functional as F

class PitchDifferenceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, f0, f0_shift, d):
        diff = torch.log2(f0) - torch.log2(f0_shift)
        return F.huber_loss(diff, d / 24)