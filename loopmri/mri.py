"""MRI operators shared by the model, the data and the metrics.

Layout conventions (fastMRI's):
  * complex tensors carry real/imag on a last axis of size 2;
  * multicoil k-space is [B, C, H, W, 2]; with A adjacent slices the slices are stacked SLICE-MAJOR along the coil axis,
    [B, A*C, H, W, 2] (slice 0's coils, then slice 1's, ...), and the sensitivity maps use the same layout;
  * a coil-combined image is [B, A, H, W, 2] (one complex channel per slice);
  * FFTs are centred and orthonormal (fastmri.fft2c / ifft2c).
"""
import torch

import fastmri


def sens_expand(x: torch.Tensor, sens: torch.Tensor, adj: int = 1) -> torch.Tensor:
    """E: coil-combined image(s) [B, A, H, W, 2] -> multicoil k-space [B, A*C, H, W, 2] (k = F S x)."""
    if adj > 1:
        x = x.repeat_interleave(sens.shape[1] // adj, dim=1)   # each slice's image to each of its C coils
    return fastmri.fft2c(fastmri.complex_mul(x, sens))


def sens_reduce(k: torch.Tensor, sens: torch.Tensor, adj: int = 1) -> torch.Tensor:
    """E^H: multicoil k-space [B, A*C, H, W, 2] -> coil-combined image(s) [B, A, H, W, 2] (x = sum_c conj(S_c) F^H k_c)."""
    x = fastmri.complex_mul(fastmri.ifft2c(k), fastmri.complex_conj(sens))
    if adj > 1:
        b, c, h, w, _ = x.shape
        return x.view(b, adj, c // adj, h, w, 2).sum(dim=2)
    return x.sum(dim=1, keepdim=True)


def normal_op(x: torch.Tensor, mask: torch.Tensor, sens: torch.Tensor, adj: int = 1) -> torch.Tensor:
    """A^H A x for a coil-combined image x [B, A, H, W, 2]: expand to k-space, keep the sampled locations, reduce."""
    zero = torch.zeros(1, 1, 1, 1, 1, dtype=x.dtype, device=x.device)
    return sens_reduce(torch.where(mask, sens_expand(x, sens, adj), zero), sens, adj)


def soft_dc(k: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """E2E-VarNet's soft data-consistency term: weight * (k - y) on the sampled locations, 0 elsewhere."""
    zero = torch.zeros(1, 1, 1, 1, 1, dtype=k.dtype, device=k.device)
    return torch.where(mask, k - y, zero) * weight


def centre_chunk(t: torch.Tensor, adj: int = 1) -> torch.Tensor:
    """The centre slice's part of a slice-major stack along dim 1 (k-space, maps or images)."""
    return t if adj == 1 else torch.chunk(t, adj, dim=1)[adj // 2]


def kspace_to_rss(k: torch.Tensor) -> torch.Tensor:
    """Multicoil k-space [B, C, H, W, 2] -> root-sum-of-squares magnitude image [B, H, W]."""
    return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(k)), dim=1)
