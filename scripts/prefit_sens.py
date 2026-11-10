#!/usr/bin/env python
"""Pre-fit the sensitivity estimator to the PLAIN calibration maps before training (knee), so a from-scratch run starts from coherent maps.

  python scripts/prefit_sens.py --config configs/knee5_rs5_fd.yaml --out weights/knee5_sens_prefit
  -> <out>/sens_plain.pt (keys sens_net.*, loaded by model.sens_init) and <out>/prefit.json (map error / coherence before, during, after)

Why: on the knee data PromptMR+'s estimator, trained from scratch inside our model, drifted to incoherent maps early in training. The fix
used for every knee run: regress the estimator (built exactly as training builds it) onto the plain maps S0 = I_calib / RSS(I_calib)
(sens.plain_maps: the calibration region = the central lines for knee, inverse FFT, RSS-normalised per slice) for 4,000 steps, start
training from the result (model.sens_init) and hold it fixed for the first 2,000 training steps (optim.sens_hold_steps).
Loss: complex MSE to S0, weighted 1 inside the object (the plain calibration image's RSS > 5% of its max, per slice) and 0.1 outside,
normalised by the weight sum x 2. AdamW lr 1e-3, no weight decay, 200-step warm-up then cosine to 0, gradient-norm clip 1.0.
Reported: map error ||S - S0||^2 / ||S0||^2 inside the object, and coherence ||E^H y|| / ||RSS(F^-1 y)|| (1 = perfect coil combination).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fastmri  # noqa: E402
from loopmri.data import batch_to_device, build_dataset, build_loader  # noqa: E402
from loopmri.model import build_model  # noqa: E402
from loopmri.mri import sens_reduce  # noqa: E402
from loopmri.sens import calibration_region, plain_maps  # noqa: E402
from loopmri.train import load_config, seed_everything  # noqa: E402


def object_weight(y, mask, nlf, adj, calibration, bg=0.1, thr=0.05):
    """Per-coil weights [B, A*C, H, W, 1] (1 inside each slice's object, bg outside) and the object mask [B, A, H, W]."""
    k = calibration_region(y, mask, nlf, calibration)
    b, ac, h, w, _ = k.shape
    r = fastmri.rss_complex(fastmri.ifft2c(k).view(b * adj, ac // adj, h, w, 2), dim=1)        # [B*A, H, W]
    obj = (r > thr * r.flatten(1).amax(1).view(-1, 1, 1)).view(b, adj, h, w)
    wt = (obj.float() * (1 - bg) + bg).repeat_interleave(ac // adj, dim=1).unsqueeze(-1)
    return wt, obj


@torch.no_grad()
def evaluate(student, loader, device, adj, calibration, max_batches):
    err, coh, n = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        b = batch_to_device(batch, device)
        y, m, nlf = b["masked_kspace"], b["mask"], b["num_low_frequencies"]
        S0, S = plain_maps(y, m, nlf, adj, calibration).float(), student(y, m, nlf).float()
        _, obj = object_weight(y, m, nlf, adj, calibration)
        o = obj.repeat_interleave(S.shape[1] // adj, dim=1).unsqueeze(-1).float()
        err += float((((S - S0) ** 2) * o).sum() / ((S0 ** 2) * o).sum().clamp_min(1e-12))
        bb, ac, h, w, _ = y.shape
        ref = fastmri.rss_complex(fastmri.ifft2c(y.float()).view(bb, adj, ac // adj, h, w, 2), dim=2).norm()
        coh += float(sens_reduce(y.float(), S, adj).norm() / ref)
        n += 1
    return {"map_err": err / n, "coh": coh / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action="extend", default=[])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed_everything(int(cfg["seed"]))
    dev = torch.device("cuda")
    model = build_model(cfg)
    student = model.sens_net.to(dev).float()                  # the run's estimator, exactly as training builds it
    adj, calibration = model.A, cfg["model"].get("sens_calibration", "disc")
    dcfg = cfg["data"]
    train_loader = build_loader(build_dataset(dcfg, "train", train=True), True, int(dcfg["num_workers"]))
    val_loader = build_loader(build_dataset(dcfg, "val", train=False), False, int(dcfg["num_workers"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"estimator {sum(p.numel() for p in student.parameters())} params; calibration {calibration}; {adj} adjacent slice(s)", flush=True)

    student.eval()
    before = evaluate(student, val_loader, dev, adj, calibration, a.eval_batches)
    print("before:", json.dumps(before), flush=True)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    student.train()
    s, t0, run, curve = 0, time.time(), [], []
    while s < a.steps:
        for batch in train_loader:
            b = batch_to_device(batch, dev)
            y, m, nlf = b["masked_kspace"], b["mask"], b["num_low_frequencies"]
            with torch.no_grad():
                S0 = torch.nan_to_num(plain_maps(y, m, nlf, adj, calibration).float())
                w, _ = object_weight(y, m, nlf, adj, calibration)
            loss = (((student(y, m, nlf) - S0) ** 2) * w).sum() / (w.sum() * 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step(); s += 1
            run.append(loss.item())
            if s % 500 == 0 or s == a.steps:
                student.eval()
                ev = evaluate(student, val_loader, dev, adj, calibration, a.eval_batches // 4)
                student.train()
                row = {"step": s, "loss": sum(run) / len(run), **ev, "seconds": round(time.time() - t0)}
                curve.append(row); run = []
                print(json.dumps(row), flush=True)
            if s >= a.steps:
                break
    student.eval()
    after = evaluate(student, val_loader, dev, adj, calibration, a.eval_batches)
    print("after:", json.dumps(after), flush=True)
    torch.save({"model": {"sens_net." + k: v.cpu() for k, v in student.state_dict().items()}, "step": 0,
                "prefit": {"steps": a.steps, "lr": a.lr, "before": before, "after": after}}, out / "sens_plain.pt")
    json.dump({"config": a.config, "set": a.set, "steps": a.steps, "lr": a.lr, "before": before, "after": after, "curve": curve},
              open(out / "prefit.json", "w"), indent=1)
    print(f"PREFIT DONE: map_err {before['map_err']:.3f} -> {after['map_err']:.4f}, coh {before['coh']:.3f} -> {after['coh']:.3f}; "
          f"saved {out / 'sens_plain.pt'}", flush=True)


if __name__ == "__main__":
    main()
