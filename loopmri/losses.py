"""Training loss over the iterates of the gradient window: 1 - SSIM (fastMRI's SSIMLoss), weighted over loops.

loss.loops = all:  every iterate in the window is scored, weights w_t proportional to t = 1..T (later loops weigh more), summing to 1
                   (sampled depth; T = 8; no config here);
loss.loops = last: only the final output is scored (the fixed-depth variant; the gradient still flows through every loop of the window).
"""
from typing import List

import torch
import torch.nn as nn

from fastmri.data import transforms
from fastmri.losses import SSIMLoss


class WindowSSIMLoss(nn.Module):
    def __init__(self, loops: str = "all"):
        super().__init__()
        if loops not in ("all", "last"):
            raise ValueError(f"loss.loops must be all or last, got {loops}")
        self.loops = loops
        self.ssim = SSIMLoss()

    def single(self, output: torch.Tensor, target: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
        target, output = transforms.center_crop_to_smallest(target, output)
        return self.ssim(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)

    def forward(self, images: List[torch.Tensor], target: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
        if self.loops == "last":
            images = images[-1:]
        w = torch.arange(1, len(images) + 1, dtype=torch.float32, device=target.device)
        w = w / w.sum()
        total = 0.0
        for wt, img in zip(w, images):
            total = total + wt * self.single(img, target, max_value)
        return total
