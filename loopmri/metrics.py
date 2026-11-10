"""Metrics: one per dataset in the reported results (loopmri.evaluate). Every score is computed per slice / image, averaged within a
volume, then averaged over volumes.

VolumeMetrics:   fastMRI's convention (MriModule): data_range = the volume max (the knee volume's `max` attribute; 1 after the CC359 /
                 CMRxRecon loaders' scaling); SSIM = mean of the slice SSIMs (fastmri.evaluate.ssim, 7x7 window); PSNR and NMSE from the
                 volume's mean MSE. THE PAPER'S KNEE NUMBERS. Also the trainer's validation on every dataset (model selection only: on
                 CMRxRecon it is uncropped, ~0.99, and not a reported number).
challenge_volume: the official Calgary-Campinas MC-MRI-Rec scoring (mrrec/metrics.py), which PromptMR+ uses for its CC359 table: per slice,
                 both images divided by the reference slice max, skimage SSIM / PSNR with data_range = max(ref, rec) - min(ref, rec).
                 THE PAPER'S CC359 TEST NUMBERS ARE THESE.
CMRChallengeMetrics: the CMRxRecon2023 ranking protocol (the challenge's SubmissionFormat_Task{1,2}.txt `run4Ranking` +
                 github.com/CmrxRecon/CMRxRecon Evaluation/Evaluation.py `calmetric`): only the first 3 frames / weightings and the
                 central 2 slices of a file (cmr_challenge_images), each image centre-cropped to round(H/3) x round(W/2) (H = readout incl.
                 the 2x oversampling, W = phase encoding), each image divided by ITS OWN max (reference and reconstruction separately),
                 skimage SSIM / PSNR with data_range 1, NMSE; mean over a file's images, then over files.
                 THE PAPER'S CMRxRecon NUMBERS ARE THESE.
"""
import math
import re
from collections import defaultdict
from typing import Dict

import numpy as np
from fastmri import evaluate
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def challenge_volume(tgt: np.ndarray, rec: np.ndarray) -> Dict[str, float]:
    ssim, psnr = [], []
    for t, r in zip(tgt.astype(np.float64), rec.astype(np.float64)):
        m = t.max()
        t, r = t / m, r / m
        dr = max(t.max(), r.max()) - min(t.min(), r.min())
        ssim.append(structural_similarity(t, r, data_range=dr))
        psnr.append(peak_signal_noise_ratio(t, r, data_range=dr))
    return {"ssim": float(np.mean(ssim)), "psnr": float(np.mean(psnr))}


class VolumeMetrics:
    """Collects slices during validation; `compute` gives fastMRI-convention means over volumes, overall, per acceleration (x4, x5, ...)
    and, for CMRxRecon files (P001_cine_lax.h5), per scan type; `per_volume` gives each volume's own values."""

    def __init__(self):
        self.mse = defaultdict(dict)
        self.target_norm = defaultdict(dict)
        self.ssim = defaultdict(dict)
        self.maxval, self.accel = {}, {}

    def update(self, fname: str, slice_num: int, output: np.ndarray, target: np.ndarray, maxval: float, accel: int) -> None:
        self.mse[fname][slice_num] = float(evaluate.mse(target, output))
        self.target_norm[fname][slice_num] = float(evaluate.mse(target, np.zeros_like(target)))
        self.ssim[fname][slice_num] = float(evaluate.ssim(target[None], output[None], maxval=maxval))
        self.maxval[fname], self.accel[fname] = float(maxval), int(accel)

    def per_volume(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for f in self.mse:
            mse = np.mean(list(self.mse[f].values()))
            out[f] = {"nmse": float(mse / np.mean(list(self.target_norm[f].values()))),
                      "psnr": float(20 * np.log10(self.maxval[f]) - 10 * np.log10(max(mse, 1e-20))),
                      "ssim": float(np.mean(list(self.ssim[f].values()))), "accel": self.accel[f], "n_slices": len(self.mse[f])}
        return out

    def compute(self) -> Dict[str, Dict[str, float]]:
        groups = defaultdict(list)
        for f, v in self.per_volume().items():
            groups["all"].append(v)
            groups[f"x{v['accel']}"].append(v)
            m = re.match(r"P\d+_(.+)\.h5$", f)        # CMRxRecon: also by scan type
            if m:
                groups[m.group(1)].append(v)
        return {g: {**{m: float(np.mean([v[m] for v in vols])) for m in ("ssim", "psnr", "nmse")}, "num_volumes": len(vols)}
                for g, vols in groups.items()}


def _mround(x: float) -> int:
    """MATLAB round (half away from zero), for x >= 0."""
    return int(math.floor(x + 0.5))


def cmr_challenge_images(num_t: int, num_z: int) -> list:
    """(t, z) of the images the CMRxRecon2023 ranking scores in a file: frames / weightings 1:3 and slices round(nz/2)-1 : round(nz/2)
    (MATLAB, 1-based), here 0-based; a single-slice file scores its one slice."""
    zs = [z for z in range(_mround(num_z / 2) - 2, _mround(num_z / 2)) if 0 <= z < num_z] or [0]
    return [(t, z) for t in range(min(3, num_t)) for z in zs]


def cmr_challenge_image(tgt: np.ndarray, rec: np.ndarray) -> Dict[str, float]:
    """One magnitude image pair [H, W]: centre crop to round(H/3) x round(W/2), each divided by its own max, then calmetric's SSIM / PSNR
    (skimage, data_range = the normalised reference max = 1) and NMSE."""
    H, W = tgt.shape
    h, w = _mround(H / 3), _mround(W / 2)
    r0, c0 = (H - h) // 2, (W - w) // 2
    t = tgt.astype(np.float64)[r0:r0 + h, c0:c0 + w]
    r = rec.astype(np.float64)[r0:r0 + h, c0:c0 + w]
    t, r = t / t.max(), r / r.max()
    return {"ssim": float(structural_similarity(t, r, data_range=t.max())),
            "psnr": float(peak_signal_noise_ratio(t, r, data_range=t.max())),
            "nmse": float(np.linalg.norm(t - r) ** 2 / np.linalg.norm(t) ** 2)}


class CMRChallengeMetrics(VolumeMetrics):
    """The CMRxRecon2023 ranking protocol: collect only the images cmr_challenge_images selects (the caller filters); per file the mean of
    the image scores; `compute` (inherited) gives the means over files, overall, per R and per scan type."""

    def __init__(self):
        self.scores = defaultdict(dict)
        self.accel = {}

    def update(self, fname: str, slice_num: int, output: np.ndarray, target: np.ndarray, accel: int) -> None:
        self.scores[fname][slice_num] = cmr_challenge_image(target, output)
        self.accel[fname] = int(accel)

    def per_volume(self) -> Dict[str, Dict[str, float]]:
        return {f: {**{m: float(np.mean([v[m] for v in s.values()])) for m in ("ssim", "psnr", "nmse")},
                    "accel": self.accel[f], "n_slices": len(s)} for f, s in self.scores.items()}
