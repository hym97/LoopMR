"""Evaluation of a trained model: the paper's numbers, one protocol and ONE metric per dataset (chosen by the checkpoint's data.name).

CC359 (cc_test): PromptMR+'s test protocol. Every volume of the split at EVERY acceleration, one fixed mask per volume (the split json's
  `mask_index`, PromptMR+'s test_mask_indices: same bank index at R5 and R10), the scored slices of the index (50..205), full field of
  view. Metric: the official Calgary-Campinas challenge metric (metrics.challenge_volume = the MC-MRI-Rec code within 4e-9,
  scripts/official_metric_check.py).

  python -m loopmri.evaluate --ckpt weights/cc359_rs5_fd_146k.pt --out runs/eval/cc359_fd_test.json [--loops 4 8 12]

fastMRI knee (heldout): the held-out set the training run validates on (multicoil_val, 40 volumes), each volume at the one R its
  file-name-seeded mask draws, every slice. Metric: fastMRI volume metrics (metrics.VolumeMetrics, per volume and per R; = the trainer's
  validation, so it reproduces a run's events.jsonl).

  python -m loopmri.evaluate --ckpt weights/knee5_rs5_fd_100k.pt --out runs/eval/knee5_fd_val.json [--loops 4 8 12] [--all-accels]

CMRxRecon (cmr_challenge): the held-out files P101-P120 (76), as in the challenge EVERY file at EVERY R (x4, x8, x10), scored with the
  CMRxRecon2023 ranking protocol (metrics.CMRChallengeMetrics): only the first 3 frames / weightings x central 2 slices of each file
  are reconstructed, each image centre-cropped to H/3 x W/2 and divided by its own max; per R and per scan type.

  python -m loopmri.evaluate --ckpt weights/cmr23_rs5_fd_466k.pt --out runs/eval/cmr23_fd_challenge.json [--loops 4 8 12]

`--all-accels` (knee only) scores every volume at every R, each with a mask drawn for that R, instead of its one fixed R.
`--loops` with several values scores every loop count from one pass per slice (the loop sweep).

The PromptMR+ baseline goes through the same protocols and metrics with its weights (loopmri/promptmr_plus.py; `--loops` = cascades,
default 12 = its output):

  python -m loopmri.evaluate --ckpt weights/promptmr_plus/cc359_promptmr_plus_released.pt --out runs/eval/cc359_promptmr_plus.json
  python -m loopmri.evaluate --ckpt weights/promptmr_plus/knee5_promptmr_plus_100k.pt --out runs/eval/knee5_promptmr_plus.json
  python -m loopmri.evaluate --ckpt weights/promptmr_plus/cmr23_promptmr_plus_466k.pt --out runs/eval/cmr23_promptmr_plus.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from fastmri.data import transforms

from .data import batch_to_device, build_dataset, build_loader
from .hub import weights_path
from .metrics import CMRChallengeMetrics, VolumeMetrics, challenge_volume, cmr_challenge_images
from .model import build_model
from .mri import kspace_to_rss
from .promptmr_plus import build_promptmr_plus, is_promptmr_plus


def load_model(ckpt: str, device) -> tuple:
    """(model in eval mode, config, step) from a training checkpoint or a released weights file (downloaded if missing): LoopMR, or
    the PromptMR+ baseline (weights/promptmr_plus/*.pt)."""
    ck = torch.load(weights_path(ckpt), map_location="cpu", weights_only=False)
    model = (build_promptmr_plus if is_promptmr_plus(ck["cfg"]) else build_model)(ck["cfg"])
    model.load_state_dict(ck["model"])          # strict
    return model.to(device).eval(), ck["cfg"], int(ck["step"])


def iterate_images(model, b: dict, max_loops: int):
    """Yield (t, centre slice's RSS image [B, H, W]) after loop (LoopMR) or cascade (PromptMR+) t = 0 .. max_loops."""
    if hasattr(model, "iterate_images"):
        yield from model.iterate_images(b["masked_kspace"], b["mask"], b["num_low_frequencies"], max_loops=max_loops)
        return
    for t, k in model.iterate(b["masked_kspace"], b["mask"], b["num_low_frequencies"], max_loops=max_loops):
        yield t, kspace_to_rss(k)


@torch.no_grad()
def evaluate_cc_test(model, cfg, split_json: str, split: str, accels, loops, device, num_workers: int = 4) -> dict:
    out = {}
    for a in accels:
        ds = build_dataset(cfg["data"], split, train=False, split_json=split_json, accelerations=[a])
        vols = {}
        for batch in build_loader(ds, False, num_workers):
            b = batch_to_device(batch, device)
            v = vols.setdefault(b["fname"][0], {"tgt": [], "rec": {t: [] for t in loops}, "slices": []})
            v["tgt"].append(b["target"][0].cpu().numpy())
            v["slices"].append(b["slice_num"][0])
            for t, img in iterate_images(model, b, max(loops)):
                if t in v["rec"]:
                    v["rec"][t].append(img[0].float().cpu().numpy())
        for t in loops:
            per_vol = {}
            for f, v in sorted(vols.items()):
                tgt, rec = np.stack(v["tgt"]), np.stack(v["rec"][t])
                per_vol[f] = {"challenge": challenge_volume(tgt, rec),
                              "n_slices": len(v["slices"]), "slices": [min(v["slices"]), max(v["slices"]) + 1],
                              "mask_index": ds.mask_index.get(f)}
            mean = {"challenge": {m: float(np.mean([p["challenge"][m] for p in per_vol.values()])) for m in ("ssim", "psnr")}}
            out.setdefault(f"loops{t}", {})[f"x{a}"] = {"mean": mean, "num_volumes": len(per_vol), "volumes": per_vol}
            print(f"[eval] CC359 test x{a} loops {t}: {len(per_vol)} volumes | challenge SSIM {mean['challenge']['ssim']:.4f} "
                  f"PSNR {mean['challenge']['psnr']:.2f}", flush=True)
    return out


@torch.no_grad()
def evaluate_heldout(model, cfg, split: str, loops, device, all_accels: bool = False, num_workers: int = 4,
                     cmr_challenge: bool = False) -> dict:
    """Knee (the CLI): the trainer's validation (train.validate) for several loop counts at once, with per-volume values.
    CMRxRecon (the CLI passes cmr_challenge=True): the same files, R and masks, but only the images the CMRxRecon2023 ranking scores,
    with its metric. cmr_challenge=False on CMRxRecon = the trainer's uncropped fastMRI validation (model selection only), not a
    reported number."""
    runs = [(f"x{a}", [a]) for a in cfg["data"]["accelerations"]] if all_accels else [("fixed_R_per_volume", None)]
    out = {}
    for tag, accels in runs:
        ds = build_dataset(cfg["data"], split, train=False, accelerations=accels)
        if cmr_challenge:       # keep the scored (t, z) of each file; s = t * num_z + z (data_cmr)
            keep = {f: {t * m["num_z"] + z for t, z in cmr_challenge_images(m["num_t"], m["num_z"])} for f, m in ds.meta.items()}
            ds.samples = [(f, s) for f, s in ds.samples if s in keep[f]]
        metrics = {t: CMRChallengeMetrics() if cmr_challenge else VolumeMetrics() for t in loops}
        for batch in build_loader(ds, False, num_workers):
            b = batch_to_device(batch, device)
            for t, img in iterate_images(model, b, max(loops)):
                if t in metrics:
                    target, output = transforms.center_crop_to_smallest(b["target"], img)
                    if cmr_challenge:
                        metrics[t].update(b["fname"][0], b["slice_num"][0], output[0].float().cpu().numpy(),
                                          target[0].cpu().numpy(), b["accel"])
                    else:
                        metrics[t].update(b["fname"][0], b["slice_num"][0], output[0].cpu().numpy(), target[0].cpu().numpy(),
                                          float(b["max_value"][0]), b["accel"])
        for t in loops:
            res = metrics[t].compute()
            out.setdefault(f"loops{t}", {})[tag] = {"groups": res, "volumes": metrics[t].per_volume()}
            name = "CMRxRecon challenge" if cmr_challenge else "held-out fastMRI"
            print(f"[eval] {name} {tag} loops {t}: " + "  ".join(f"{g} {v['ssim']:.4f}/{v['psnr']:.2f} (n={v['num_volumes']})"
                                                                for g, v in res.items()), flush=True)
    return out


PROTOCOL = {"cc359": "cc_test", "fastmri_knee": "heldout", "cmr23": "cmr_challenge"}
METRIC = {"cc_test": "Calgary-Campinas challenge (MC-MRI-Rec)", "heldout": "fastMRI volume", "cmr_challenge": "CMRxRecon2023 ranking"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-json", default="configs/splits/cc359_promptmr.json", help="CC359 only: the split file (PromptMR+'s test split)")
    ap.add_argument("--split", default=None, help="default: test (CC359) / val (knee, CMRxRecon)")
    ap.add_argument("--accels", type=int, nargs="+", default=[5, 10], help="CC359 only")
    ap.add_argument("--all-accels", action="store_true", help="knee only: every volume at every R (CMRxRecon always does this)")
    ap.add_argument("--loops", type=int, nargs="+", default=None, help="default: the run's loops.eval_loops")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model, cfg, step = load_model(args.ckpt, device)
    loops = sorted(set(args.loops or [int(cfg["loops"]["eval_loops"])]))
    protocol = PROTOCOL[cfg["data"].get("name", "cc359")]      # one protocol (and metric) per dataset: the paper's
    nw = int(cfg["data"]["num_workers"])
    res = {"ckpt": args.ckpt, "model": "PromptMR+" if is_promptmr_plus(cfg) else "LoopMR", "ckpt_step": step, "loops": loops,
           "protocol": protocol, "metric": METRIC[protocol]}
    if protocol == "cc_test":
        split = args.split or "test"
        res.update(split_json=args.split_json, split=split,
                   results=evaluate_cc_test(model, cfg, args.split_json, split, args.accels, loops, device, nw))
    else:
        split = args.split or "val"
        all_accels = args.all_accels or protocol == "cmr_challenge"     # CMRxRecon: every file at every R, as the challenge
        res.update(split=split, all_accels=all_accels,
                   results=evaluate_heldout(model, cfg, split, loops, device, all_accels, nw,
                                            cmr_challenge=protocol == "cmr_challenge"))
    res["eval_seconds"] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"[eval] step {step} loops {loops} ({protocol}) -> {args.out}")


if __name__ == "__main__":
    main()
