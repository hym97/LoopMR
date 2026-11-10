"""Data: the CC359 loader (this file), the fastMRI knee loader (data_knee.py) and the CMRxRecon2023 loader (data_cmr.py), chosen by
data.name (cc359 | fastmri_knee | cmr23); the loaders, batch_to_device and the metrics share one sample format (a dict with masked_kspace,
mask, num_low_frequencies, target, max_value, fname, slice_num, accel). Batch size is always 1.

Calgary-Campinas (CC359) 12-channel brain data in its native format, with adjacent slices.

Files: one h5 per volume, `kspace[x, ky, kz, 2*coils] float32`, hybrid (already inverse-FFT'd along the readout x, so axis 0 indexes 2-D
slices), real/imag interleaved on the last axis. The index json (data.index) holds per volume: the folder (`split_dir`), `shape`, the
volume `max` (k-space is divided by it, so targets live in [0, 1] and max_value = 1) and the scored `slice_range`. Split membership
comes from a frozen split json (data.split_json), not from the folder names.

Masks: the challenge's 2-D Poisson-disc banks, `masks_dir/R{5,10}_218x{kz}.npy`, 100 masks each, all with a fully sampled central disc
of radius 16 (the calibration region of the sensitivity estimator). Training draws R and the bank index at random per sample; a held-out
volume uses ONE fixed mask: the split json's `mask_index` if it has one (PromptMR+'s test split), else crc32(fname) % 100.
Validation (train=False) scores each volume at one R, alternating over the sorted file list so the R groups are balanced; the test
evaluation (evaluate.py) builds one dataset per R instead.
"""
import json
import random
import zlib
from pathlib import Path
from typing import Dict, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import fastmri

BANK_SIZE = 100
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(p) -> Path:
    """Paths in a config (data/..., configs/splits/..., runs, weights/...) are relative to the repo; an absolute path stays as it is."""
    return REPO_ROOT / p


def to_complex(a: np.ndarray) -> np.ndarray:
    """Stored layout: real/imag interleaved on the last axis."""
    return a[..., ::2] + 1j * a[..., 1::2]


def checkerboard(k: np.ndarray) -> np.ndarray:
    """k-space is stored centred but for an image with its origin at the corner, so a centred ifft (fastmri.ifft2c) would return a
    half-FOV-shifted image. Multiplying k by (-1)^(ky+kz) moves it back (exact: all sizes are even)."""
    h, w = k.shape[:2]
    sign = (1 - 2 * (np.arange(h) % 2))[:, None] * (1 - 2 * (np.arange(w) % 2))[None, :]
    return k * sign[..., None]


def heldout_mask_index(fname: str) -> int:
    return zlib.crc32(fname.encode()) % BANK_SIZE


class CC359SliceDataset(Dataset):
    def __init__(self, index: str, split_json: str, masks_dir: str, split: str, train: bool, accelerations: Sequence[int] = (5, 10),
                 acs_radius: int = 16, num_adj_slices: int = 5):
        if num_adj_slices < 1 or num_adj_slices % 2 == 0:
            raise ValueError("num_adj_slices must be odd")
        self.num_adj_slices = int(num_adj_slices)
        idx = json.loads(Path(index).read_text())
        self.root = Path(index).parent / idx["root"]      # the index's `root` is relative to the index file (or absolute)
        self.volumes: Dict[str, Dict] = idx["volumes"]
        self.masks_dir = Path(masks_dir)
        self.train = train
        self.accelerations = [int(a) for a in accelerations]
        self.acs_radius = int(acs_radius)
        self.rng = np.random.default_rng(0)        # reseeded per DataLoader worker (worker_init_fn)
        self._banks: Dict = {}

        manifest = json.loads(Path(split_json).read_text())
        fnames = list(manifest[split])
        self.mask_index: Dict[str, int] = {f: int(i) for f, i in manifest.get("mask_index", {}).items()}
        missing = [f for f in fnames if f not in self.volumes]
        if missing:
            raise FileNotFoundError(f"{split}: not in the index: {missing}")
        self.heldout_accel = {f: self.accelerations[i % len(self.accelerations)] for i, f in enumerate(sorted(fnames))}
        self.samples = []                           # (path, slice index)
        for f in fnames:
            v = self.volumes[f]
            lo, hi = v["slice_range"]
            self.samples += [(self.root / v["split_dir"] / f, s) for s in range(lo, hi)]

    def reseed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def _bank(self, accel: int, kz: int) -> np.ndarray:
        if (accel, kz) not in self._banks:
            self._banks[(accel, kz)] = np.load(self.masks_dir / f"R{accel}_218x{kz}.npy")
        return self._banks[(accel, kz)]

    def pick_mask(self, fname: str, kz: int):
        """(mask [ky, kz] bool, acceleration)."""
        if self.train:
            accel = self.accelerations[int(self.rng.integers(len(self.accelerations)))]
            return self._bank(accel, kz)[int(self.rng.integers(BANK_SIZE))], accel
        accel = self.heldout_accel[fname]
        return self._bank(accel, kz)[self.mask_index.get(fname, heldout_mask_index(fname))], accel

    def __getitem__(self, i: int) -> Dict:
        path, s = self.samples[i]
        v = self.volumes[path.name]
        half = self.num_adj_slices // 2
        n_vol = int(v["shape"][0])
        ids = [min(max(s + d, 0), n_vol - 1) for d in range(-half, half + 1)]        # neighbours clamped at the volume ends
        with h5py.File(path, "r") as hf:
            block = hf["kspace"][ids[0]:ids[-1] + 1]
        ks = []
        for j in ids:
            kc = checkerboard(to_complex(block[j - ids[0]])) / v["max"]                 # [ky, kz, coils] complex
            ks.append(torch.from_numpy(np.stack([kc.real, kc.imag], axis=-1).astype(np.float32)).permute(2, 0, 1, 3).contiguous())
        target = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(ks[half])), dim=0)      # centre slice, fully sampled, [ky, kz]
        k = torch.cat(ks, dim=0)                                                         # [A*coils, ky, kz, 2], slice-major
        m, accel = self.pick_mask(path.name, k.shape[2])
        mask = torch.from_numpy(np.ascontiguousarray(m))[None, :, :, None]               # [1, ky, kz, 1]
        return {"masked_kspace": k * mask, "mask": mask, "num_low_frequencies": self.acs_radius, "target": target,
                "max_value": 1.0, "fname": path.name, "slice_num": s, "accel": accel}


def worker_init_fn(worker_id):
    """Give each DataLoader worker its own mask RNG (otherwise every worker draws the same training masks)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.reseed(info.seed % (2**32 - 1))


def build_dataset(dcfg: Dict, split: str, train: bool, split_json: str = None, accelerations: Sequence[int] = None) -> Dataset:
    """split: train | val (| test for CC359). `accelerations` (evaluation only): CC359 and knee: the R values to draw from; CMRxRecon:
    one R for every file. `split_json`: CC359 only (e.g. PromptMR+'s test split)."""
    name = dcfg.get("name", "cc359")
    if name == "fastmri_knee":
        from .data_knee import build_knee_dataset
        return build_knee_dataset(dcfg, split, train, accelerations)
    if name == "cmr23":
        from .data_cmr import build_cmr_dataset
        return build_cmr_dataset(dcfg, split, train, fixed_accel=accelerations[0] if accelerations else None)
    if name != "cc359":
        raise ValueError(f"data.name must be cc359, fastmri_knee or cmr23, got {name}")
    return CC359SliceDataset(index=repo_path(dcfg["index"]), split_json=repo_path(split_json or dcfg["split_json"]),
                             masks_dir=repo_path(dcfg["masks_dir"]), split=split,
                             train=train, accelerations=accelerations or dcfg["accelerations"], acs_radius=dcfg["acs_radius"],
                             num_adj_slices=dcfg["num_adj_slices"])


def build_loader(ds: Dataset, train: bool, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=1, shuffle=train, num_workers=num_workers, worker_init_fn=worker_init_fn,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)


def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {"masked_kspace": batch["masked_kspace"].to(device, non_blocking=True),
            "mask": batch["mask"].to(device, non_blocking=True).bool(),
            "num_low_frequencies": batch["num_low_frequencies"].to(device),
            "target": batch["target"].to(device, non_blocking=True).float(),
            "max_value": batch["max_value"].to(device).float(),
            "fname": list(batch["fname"]),
            "slice_num": [int(s) for s in batch["slice_num"]],
            "accel": int(batch["accel"][0])}
