# sens_expand / sens_reduce with adjacent-slice support, from PromptMR-plus mri_utils/coil_combine.py (same commit;
# that file is derived from fastMRI, MIT licence). Only these two functions are kept; FFTs come from fastmri.
import torch
from fastmri import complex_conj, complex_mul, fft2c, ifft2c


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor, num_adj_slices: int = 1) -> torch.Tensor:
    _, c, _, _, _ = sens_maps.shape
    return fft2c(complex_mul(x.repeat_interleave(c // num_adj_slices, dim=1), sens_maps))


def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor, num_adj_slices: int = 1) -> torch.Tensor:
    b, c, h, w, _ = x.shape
    x = ifft2c(x)
    x = complex_mul(x, complex_conj(sens_maps))
    return x.view(b, num_adj_slices, c // num_adj_slices, h, w, 2).sum(dim=2, keepdim=False)
