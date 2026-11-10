"""fastMRI multicoil knee with PromptMR+'s knee data path (hellopipu/PromptMR-plus 934eeda: data/mri_data.py FastmriSliceDataset +
data/transforms.py FastmriDataTransform, configs/train/pmr-plus/fm-knee.yaml). data.name: fastmri_knee.

Per sample, as upstream:
  * 5 adjacent slices of the same volume (clamped at the ends), stacked slice-major on the coil axis;
  * uniform resolution 384 x 384: each slice's coil images are centre-cropped to 384 rows (removes the 2x readout oversampling) and
    reflect-padded to 384 columns (368 -> 384), then transformed back to k-space;
  * one fastMRI random line mask (4x with 8% centre lines, 8x with 4%) shared by all slices and coils, the acquisition padding zeroed
    (padding_left / padding_right of the ORIGINAL width, as upstream); held-out masks are seeded by the file name, so every held-out
    volume always gets the same mask (and with it one acceleration);
  * target = the stored `reconstruction_rss` of the centre slice (320 x 320, raw scale), max_value = the volume's `max` attribute.
Split = the folders `root/multicoil_train` and `root/multicoil_val` (the knee pilot: 159 / 40 volumes of fastMRI's official val set).
"""
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

import fastmri
from fastmri.data import SliceDataset
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data.transforms import apply_mask, to_tensor


class UniformResolution:
    """Upstream FastmriDataTransform._to_uniform_size: ifft -> centre crop / reflect pad (rows, cols) -> fft. k [..., H, W, 2]."""

    def __init__(self, size: Tuple[int, int] = (384, 384)):
        self.size = (int(size[0]), int(size[1]))

    def _crop(self, image):
        w_from = h_from = 0
        if self.size[0] < image.shape[-3]:
            w_from = (image.shape[-3] - self.size[0]) // 2
            w_to = w_from + self.size[0]
        else:
            w_to = image.shape[-3]
        if self.size[1] < image.shape[-2]:
            h_from = (image.shape[-2] - self.size[1]) // 2
            h_to = h_from + self.size[1]
        else:
            h_to = image.shape[-2]
        return image[..., w_from:w_to, h_from:h_to, :]

    def _pad(self, image):
        pad_w = self.size[0] - image.shape[-3]
        pad_h = self.size[1] - image.shape[-2]
        pw_l, pw_r = (pad_w // 2, pad_w - pad_w // 2) if pad_w > 0 else (0, 0)
        ph_l, ph_r = (pad_h // 2, pad_h - pad_h // 2) if pad_h > 0 else (0, 0)
        return torch.nn.functional.pad(image.permute(0, 3, 1, 2), (ph_l, ph_r, pw_l, pw_r), "reflect").permute(0, 2, 3, 1)

    def __call__(self, kspace: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(self._pad(self._crop(fastmri.ifft2c(kspace))))


def acceleration_from_mask(mask: torch.Tensor, accelerations: Sequence[int]) -> int:
    """The configured acceleration closest to the mask's density (a random line mask does not record which one it drew)."""
    est = mask.numel() / max(float(mask.sum()), 1.0)
    return int(min(accelerations, key=lambda a: abs(a - est)))


class FastMRIKneeSliceDataset(SliceDataset):
    """fastmri's SliceDataset (file list, header metadata, cache) with upstream PromptMR+'s adjacent-slice loading and transform."""

    def __init__(self, root, train: bool, center_fractions: Sequence[float], accelerations: Sequence[int], num_adj_slices: int = 5,
                 uniform_resolution: Optional[Sequence[int]] = (384, 384), **kwargs):
        if num_adj_slices < 1 or num_adj_slices % 2 == 0:
            raise ValueError("num_adj_slices must be odd")
        super().__init__(root=root, challenge="multicoil", transform=None, **kwargs)
        self.train = train
        self.num_adj_slices = int(num_adj_slices)
        self.accelerations = [int(a) for a in accelerations]
        self.mask_func = RandomMaskFunc(list(center_fractions), self.accelerations)      # allow_any_combination=False, as upstream
        self.uniform = UniformResolution(uniform_resolution) if uniform_resolution else None

    def reseed(self, seed: int):
        """Per DataLoader worker (data.worker_init_fn): independent training masks per worker."""
        self.mask_func.rng.seed(seed % (2**32 - 1))

    def slice_ids(self, s: int, num_slices: int):
        """Upstream FastmriSliceDataset._get_frames_indices: centre +- (A-1)/2, clamped."""
        h = self.num_adj_slices // 2
        return [min(max(s + d, 0), num_slices - 1) for d in range(-h, h + 1)]

    def __getitem__(self, i: int) -> Dict:
        fname, s, metadata = self.raw_samples[i]
        with h5py.File(fname, "r") as hf:
            ids = self.slice_ids(s, hf["kspace"].shape[0])
            ks = [hf["kspace"][z] for z in ids]
            target = hf["reconstruction_rss"][s]
            attrs = dict(hf.attrs)
            attrs.update(metadata)
        k = to_tensor(np.concatenate(ks, axis=0))                        # [A*C, H, W, 2], slice-major
        if self.uniform is not None:
            k = self.uniform(k)
        seed = None if self.train else tuple(map(ord, fname.name))
        masked, mask, nlf = apply_mask(k, self.mask_func, seed=seed, padding=(attrs["padding_left"], attrs["padding_right"]))
        mask = mask.to(torch.bool)                                       # [1, 1, W, 1]
        return {"masked_kspace": masked, "mask": mask, "num_low_frequencies": nlf, "target": torch.from_numpy(target).float(),
                "max_value": float(attrs["max"]), "fname": fname.name, "slice_num": s,
                "accel": acceleration_from_mask(mask, self.accelerations)}


def build_knee_dataset(dcfg: Dict, split: str, train: bool, accelerations: Sequence[int] = None) -> FastMRIKneeSliceDataset:
    """split: train | val (the folders multicoil_<split>). `accelerations` (one value, evaluation only): every volume at that R."""
    accels, cfs = list(dcfg["accelerations"]), list(dcfg["center_fractions"])
    if accelerations:
        cfs = [cfs[accels.index(int(a))] for a in accelerations]
        accels = [int(a) for a in accelerations]
    from .data import repo_path
    cache_dir = repo_path(dcfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return FastMRIKneeSliceDataset(root=repo_path(dcfg["root"]) / f"multicoil_{split}", train=train, center_fractions=cfs,
                                   accelerations=accels, num_adj_slices=int(dcfg["num_adj_slices"]),
                                   uniform_resolution=dcfg.get("uniform_resolution", [384, 384]),
                                   use_dataset_cache=True, dataset_cache_file=cache_dir / f"cache_{split}.pkl")
