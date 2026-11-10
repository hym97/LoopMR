#!/usr/bin/env python
"""Calgary-Campinas (CC359) 12-channel raw data -> the files and index the CC359 loader reads (loopmri/data.py).

  python data_prep/prep_cc359.py --zip <download>/Raw-data/Multi-channel/12-channel/train_val_12_channel.zip \
      --dst data/cc359/12ch --index data/cc359/index_12ch.json
  python data_prep/prep_cc359.py --dst data/cc359/12ch --index data/cc359/index_12ch.json      # already extracted: index only

Files stay as shipped: one h5 per volume, hybrid k-space [x, ky, kz, 2*coils] float32 (already inverse-FFT'd along the readout x, so
axis 0 indexes 2-D slices), extracted to <dst>/train and <dst>/val. The index holds per volume what the loader needs: the folder, the
shape, the scored slices (the challenge drops 50 outer slices on each side: 50..205) and the volume max of the fully sampled RSS over
those slices (k-space is divided by it, so targets live in [0, 1]). The split is PromptMR+'s, committed as
configs/splits/cc359_promptmr.json (train = the challenge's 47 Train volumes, val / test = its 10 / 10 cut of the challenge's 20 Val volumes).
The masks are the challenge's own Poisson-disc banks: copy R5_218x{170,174,180}.npy and R10_218x{170,174,180}.npy from the download's
poisson_sampling folder to data/cc359/masks.
"""
import argparse
import json
import os
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopmri.data import repo_path, to_complex  # noqa: E402

CROP = 50  # outer slices dropped on each side (challenge convention: 256 -> 156 scored slices)


def rss(k):
    """Fully sampled RSS image of one slice [ky, kz, coils]: centred orthonormal inverse FFT (fastmri.ifft2c's convention)."""
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(0, 1)), axes=(0, 1), norm="ortho"), axes=(0, 1))
    return np.sqrt((np.abs(img) ** 2).sum(-1))


def extract(zip_path, dst):
    zf = zipfile.ZipFile(zip_path)
    for info in zf.infolist():
        if not info.filename.endswith(".h5"):
            continue
        split, name = info.filename.split("/")[-2:]
        out = dst / split.lower() / name
        if out.exists() and out.stat().st_size == info.file_size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(str(out) + ".tmp", "wb") as f:
            while True:
                buf = src.read(64 << 20)
                if not buf:
                    break
                f.write(buf)
        Path(str(out) + ".tmp").rename(out)
        print(f"[extract] {out} {info.file_size / 1e6:.0f} MB", flush=True)


def index_one(path):
    path = Path(path)
    with h5py.File(path, "r") as hf:
        ds = hf["kspace"]
        shape, n = list(ds.shape), ds.shape[0]
        vmax = max(float(rss(to_complex(ds[s])).max()) for s in range(CROP, n - CROP))
    return path.name, {"split_dir": path.parent.name, "shape": shape, "slice_range": [CROP, n - CROP], "max": vmax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="the challenge's train_val_12_channel.zip (omit if --dst is already extracted)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--split-json", default="configs/splits/cc359_promptmr.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    dst, index_path = Path(args.dst), Path(args.index)

    if args.zip:
        extract(args.zip, dst)
    files = sorted(dst.glob("*/*.h5"))
    print(f"[index] {len(files)} volumes", flush=True)
    with Pool(args.workers) as pool:
        volumes = dict(sorted(pool.imap_unordered(index_one, [str(f) for f in files])))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({
        "root": os.path.relpath(dst.resolve(), index_path.resolve().parent),     # relative to this file
        "crop_outer_slices": CROP,
        "conventions": {"layout": "kspace[x, ky, kz, 2*coils] float32, hybrid (already ifft along x); slice = index on axis 0",
                        "complex": "interleaved: real = [..., ::2], imag = [..., 1::2]",
                        "centering": "multiply k by (-1)^(ky+kz) before a centred ifft2 (fastmri.ifft2c)",
                        "max": "max over slice_range of the fully sampled RSS, orthonormal ifft (fastmri convention)"},
        "volumes": volumes}, indent=1))
    mx = np.array([v["max"] for v in volumes.values()])
    print(f"[index] -> {index_path}; shapes {sorted({str(v['shape']) for v in volumes.values()})}; "
          f"max min {mx.min():.4g} median {np.median(mx):.4g} max {mx.max():.4g}")

    split = json.loads(repo_path(args.split_json).read_text())
    for name in ("train", "val", "test"):
        missing = [f for f in split[name] if f not in volumes]
        assert not missing, f"{name}: {len(missing)} volumes of the split are missing, e.g. {missing[:3]}"
        print(f"[split] {name}: {len(split[name])} volumes")
    print("CC359 PREP DONE")


if __name__ == "__main__":
    main()
