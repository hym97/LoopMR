"""The denoiser block applied at every loop: SDUM's Restormer backbone inside a normalise / pad wrapper.

SDUMRestormer = NVIDIA's Restormer class (vendor/sdum/restormer_sdum.py, verbatim) at SDUM's released settings except the width:
2 levels, num_blocks [3, 6], num_heads [1, 2], 2 refinement blocks, expansion 3, LayerNorm, drop_path 0.1 (decoder and refinement
blocks only), no MoE / time / label conditioning; channels [128, 256] (SDUM: [256, 512]). It already maps in_channel -> out_channel
with no global input residual; the wrapper only drops the second return value (SDUM's cross-cascade skip features, unused here).

NormWrapper generalises fastMRI's NormUnet to several complex input images: the state image and every extra input are standardised
with the STATE's (real, imag) mean / std (so their relative scale is kept), concatenated as channels, zero-padded to a multiple of the
block's down-sampling factor, run through the block, and the output is cropped and un-normalised with the same statistics.
"""
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .vendor.sdum.restormer_sdum import Restormer as NVRestormer


class SDUMRestormer(NVRestormer):
    def __init__(self, in_chans: int, out_chans: int, channels: Sequence[int] = (128, 256), num_blocks: Sequence[int] = (3, 6),
                 num_heads: Sequence[int] = (1, 2), num_refinement: int = 2, expansion_factor: float = 3.0, drop_path: float = 0.1,
                 norm_type: str = "ln"):
        super().__init__(in_channel=in_chans, out_channel=out_chans, num_blocks=list(num_blocks), num_heads=list(num_heads),
                         channels=list(channels), num_refinement=num_refinement, expansion_factor=expansion_factor,
                         drop_path=drop_path, norm_type=norm_type)
        self.pad_multiple = self.pad_factor             # 2 ** (levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[0]


class NormWrapper(nn.Module):
    """Complex images [B, n, H, W, 2] -> standardised real channels -> `inner` -> complex images [B, m, H, W, 2]."""

    def __init__(self, inner: nn.Module, pad_multiple: int):
        super().__init__()
        self.inner = inner
        self.pad_multiple = pad_multiple
        self.checkpoint_inner = False    # train.checkpoint_block: recompute the inner block in backward (same gradients, less memory)

    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        """[B, c, H, W, 2] -> [B, 2c, H, W] with channels (re_1..re_c, im_1..im_c), as fastMRI's NormUnet."""
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        return x.view(b, 2, c2 // 2, h, w).permute(0, 2, 3, 4, 1).contiguous()

    @staticmethod
    def norm_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean / std over the real group and over the imaginary group (fastMRI NormUnet.norm), each [B, 2, 1, 1]."""
        b, c, h, w = x.shape
        xg = x.view(b, 2, c // 2 * h * w)
        return xg.mean(dim=2).view(b, 2, 1, 1), xg.std(dim=2).view(b, 2, 1, 1)

    @staticmethod
    def apply_norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xg = x.view(b, 2, c // 2, h, w)
        xg = (xg - mean.view(b, 2, 1, 1, 1)) / (std.view(b, 2, 1, 1, 1) + 1e-12)
        return xg.view(b, c, h, w)

    @staticmethod
    def unnorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xg = x.view(b, 2, c // 2, h, w)
        xg = xg * std.view(b, 2, 1, 1, 1) + mean.view(b, 2, 1, 1, 1)
        return xg.view(b, c, h, w)

    def pad(self, x: torch.Tensor):
        """Zero-pad H and W up to a multiple of pad_multiple, split evenly (one more on the far side if odd)."""
        _, _, h, w = x.shape
        m = self.pad_multiple
        w_mult = ((w - 1) | (m - 1)) + 1
        h_mult = ((h - 1) | (m - 1)) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        return F.pad(x, w_pad + h_pad), (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, extras: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """x: the state image(s) [B, A, H, W, 2]; extras: further complex images, each [B, n_i, H, W, 2]."""
        x = self.complex_to_chan_dim(x)
        mean, std = self.norm_stats(x)
        x = self.apply_norm(x, mean, std)
        if extras:
            x = torch.cat([x] + [self.apply_norm(self.complex_to_chan_dim(e), mean, std) for e in extras], dim=1)
        x, pad_sizes = self.pad(x)
        if self.checkpoint_inner and torch.is_grad_enabled():
            x = cp.checkpoint(self.inner, x, use_reentrant=False)
        else:
            x = self.inner(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        return self.chan_complex_to_last_dim(x)
