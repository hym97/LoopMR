"""CMRxRecon2023 multicoil cardiac data (cine LAX / SAX, T1 / T2 mapping) in PromptMR+'s h5 format. data.name: cmr23.

Files: made by PromptMR+'s `prepare_h5_dataset_cmrxrecon.py` (commit 934eeda, unchanged) from the challenge's TrainingSet/FullSample, split
by its `configs/data_split/cmr23-cardiac.json` into `<root>/train` (P001-P100, 386 files) and `<root>/val` (P101-P120, 76 files).
Per file: `kspace [t, z, coils, H, W]` complex64 (t = time frame or weighting; H = readout, W = phase encoding), `reconstruction_rss`,
attrs `max` (volume max of the RSS).

As upstream's CMR23 training (PromptMR+ data/mri_data.py CmrxReconSliceDataset, data/subsample.py FixedLowEquiSpacedMaskFunc,
configs/train/pmr-plus/cmr23-cardiac.yaml):
  * one sample per (t, z); the 5 "adjacent slices" are ADJACENT TIME FRAMES / WEIGHTINGS at the same z, circular in t; with fewer than 5
    frames (T2 mapping: 3 weightings) the first / last index is repeated (upstream `_get_ti_adj_idx_list`);
  * training files repeated by type (upstream BalanceSampler): T1map x2, T2map x5, cine_lax x3, cine_sax x1;
  * masks: equispaced columns from offset 0 (`mask[0::R]`) plus a fixed block of 24 central lines, R drawn from {4, 8, 10} per sample;
    one mask for all frames and coils.
As our other loaders: k-space divided by the volume max (targets in [0, 1], max_value 1), batch size 1, held-out files at one fixed R
each (cycled over the files of each scan type in name order, so every type sees every R); `fixed_accel` scores every file at one R.
"""
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

import fastmri

UPSTREAM_BALANCE = {"T1map": 2, "T2map": 5, "cine_lax": 3, "cine_sax": 1}
NUM_ACS = 24


def scan_type(fname: str) -> str:
    """'P001_cine_lax.h5' -> 'cine_lax'."""
    m = re.match(r"P\d+_(.+)\.h5$", fname)
    if not m:
        raise ValueError(f"unexpected CMRxRecon file name {fname}")
    return m.group(1)


def adj_frame_indices(ti: int, num_t: int, num_adj: int) -> list:
    """Upstream CmrxReconSliceDataset._get_ti_adj_idx_list: circular neighbours in t, first / last repeated when num_t < num_adj."""
    start_adj, end_adj = -(num_adj // 2), num_adj // 2 + 1
    start_lim, end_lim = -(num_t // 2), num_t // 2 + 1
    start, end = max(start_adj, start_lim), min(end_adj, end_lim)
    idx = [(i + ti) % num_t for i in range(start, end)]
    return max(start_lim - start_adj, 0) * idx[0:1] + idx + max(end_adj - end_lim, 0) * idx[-1:]


def equispaced_mask(num_cols: int, accel: int, num_acs: int = NUM_ACS) -> np.ndarray:
    """Upstream FixedLowEquiSpacedMaskFunc with offset 0: columns 0::R plus num_acs central columns (pad = (W - num_acs + 1) // 2)."""
    mask = np.zeros(num_cols, dtype=bool)
    pad = (num_cols - num_acs + 1) // 2
    mask[pad:pad + num_acs] = True
    mask[0::accel] = True
    return mask


class CMRxReconSliceDataset(Dataset):
    def __init__(self, root: str, split: str, train: bool, accelerations: Sequence[int] = (4, 8, 10), num_adj_slices: int = 5,
                 balance: Optional[Dict[str, int]] = None, fixed_accel: Optional[int] = None):
        if num_adj_slices < 1 or num_adj_slices % 2 == 0:
            raise ValueError("num_adj_slices must be odd")
        self.num_adj_slices = int(num_adj_slices)
        self.train = train
        self.accelerations = [int(a) for a in accelerations]
        self.fixed_accel = int(fixed_accel) if fixed_accel else None
        self.rng = np.random.default_rng(0)              # reseeded per DataLoader worker (data.worker_init_fn)

        files = sorted((Path(root) / split).glob("P*.h5"))
        if not files:
            raise FileNotFoundError(f"no CMRxRecon h5 files in {Path(root) / split}")
        self.meta: Dict[str, Dict] = {}
        samples = []
        for f in files:
            with h5py.File(f, "r") as hf:
                nt, nz = hf["kspace"].shape[:2]
                self.meta[f.name] = {"path": f, "num_t": int(nt), "num_z": int(nz), "max": float(hf.attrs["max"])}
            samples += [(f.name, s) for s in range(nt * nz)]      # s = ti * num_z + zi, upstream's slice numbering
        self.heldout_accel: Dict[str, int] = {}
        by_type: Dict[str, list] = {}
        for f in files:
            by_type.setdefault(scan_type(f.name), []).append(f.name)
        for names in by_type.values():
            for i, n in enumerate(sorted(names)):
                self.heldout_accel[n] = self.accelerations[i % len(self.accelerations)]
        if train and balance:                                     # upstream BalanceSampler: repeat each type's samples
            samples = [x for x in samples for _ in range(int(balance[scan_type(x[0])]))]
        self.samples = samples

    def reseed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def pick_accel(self, fname: str) -> int:
        if self.fixed_accel:
            return self.fixed_accel
        if self.train:
            return self.accelerations[int(self.rng.integers(len(self.accelerations)))]
        return self.heldout_accel[fname]

    def __getitem__(self, i: int) -> Dict:
        fname, s = self.samples[i]
        m = self.meta[fname]
        ti, zi = divmod(s, m["num_z"])
        frames = adj_frame_indices(ti, m["num_t"], self.num_adj_slices)
        with h5py.File(m["path"], "r") as hf:
            ds = hf["kspace"]
            cache = {t: ds[t, zi] for t in sorted(set(frames))}               # [coils, H, W] complex64 each
        ks = []
        for t in frames:
            kc = cache[t] / m["max"]
            ks.append(torch.from_numpy(np.stack([kc.real, kc.imag], axis=-1).astype(np.float32)))   # [coils, H, W, 2]
        target = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(ks[self.num_adj_slices // 2])), dim=0)
        k = torch.cat(ks, dim=0)                                               # frames stacked frame-major on the coil axis
        accel = self.pick_accel(fname)
        mask = torch.from_numpy(equispaced_mask(k.shape[2], accel))[None, None, :, None]   # [1, 1, W, 1]
        return {"masked_kspace": k * mask, "mask": mask, "num_low_frequencies": NUM_ACS, "target": target, "max_value": 1.0,
                "fname": fname, "slice_num": s, "accel": accel}


def build_cmr_dataset(dcfg: Dict, split: str, train: bool, fixed_accel: Optional[int] = None) -> CMRxReconSliceDataset:
    """split: train | val. Training uses the upstream type balance (data.balance: upstream | none)."""
    balance = UPSTREAM_BALANCE if (train and dcfg.get("balance", "upstream") == "upstream") else None
    from .data import repo_path
    return CMRxReconSliceDataset(root=repo_path(dcfg["root"]), split=split, train=train, accelerations=dcfg["accelerations"],
                                 num_adj_slices=int(dcfg["num_adj_slices"]), balance=balance, fixed_accel=fixed_accel)
