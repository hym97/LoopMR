#!/usr/bin/env python
"""fastMRI knee: build the train / val folders the knee loader reads (loopmri/data_knee.py) from fastMRI's official multicoil_val set.

  python data_prep/make_knee_pilot.py --src <fastMRI>/knee/multicoil_val --dst data/knee_pilot

Creates <dst>/multicoil_train (159 volumes) and <dst>/multicoil_val (40 volumes, the reported held-out set) as symlinks to the source
files, following the committed split configs/splits/knee_pilot.json; nothing is copied or modified.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopmri.data import repo_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="fastMRI knee multicoil_val folder (199 .h5 volumes)")
    ap.add_argument("--dst", required=True, help="creates <dst>/multicoil_train and <dst>/multicoil_val")
    ap.add_argument("--split-json", default="configs/splits/knee_pilot.json")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    available = {p.name for p in src.glob("*.h5")}
    split = json.loads(repo_path(args.split_json).read_text())
    for name in ("train", "val"):
        missing = [f for f in split[name] if f not in available]
        if missing:
            raise SystemExit(f"{len(missing)} {name} volumes of the split are missing in {src}, e.g. {missing[:3]}")
        d = Path(args.dst) / f"multicoil_{name}"
        d.mkdir(parents=True, exist_ok=True)
        for f in split[name]:
            link = d / f
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(src / f, link)
        print(f"{d}: {len(split[name])} volumes")


if __name__ == "__main__":
    main()
