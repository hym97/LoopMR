"""Training.

  python -m loopmri.train --config configs/cc359_rs5_fd.yaml [--set optim.lr=1e-4 --set run_name=x ...]

Run directory runs/<run_name>/: config.yaml (the resolved config), log.csv, events.jsonl (validation results), ckpt_last.pt,
ckpt_best.pt (best validation SSIM), ckpt_step<N>.pt for N in train.keep_steps. Resumes automatically from ckpt_last.pt.
On SIGUSR1 (Slurm, shortly before the wall-time limit; slurm/train.sbatch forwards it) it saves ckpt_last.pt and exits with code 3 so
the batch script can requeue the job.

One step: batch size 1 (one slice stack); r loops (loops.train), the last loops.loss_window of them with gradient; SSIM loss
(losses.py); AdamW with the sensitivity estimator in its own parameter group at optim.sens_lr; gradient-norm clipping at
optim.grad_clip; learning rate = linear warm-up over optim.warmup_steps, then x gamma every optim.step_size steps.
A step whose loss or state is non-finite is skipped (counted; the run stops after train.max_nonfinite of them).
Knee only: `model.sens_init` starts the sensitivity estimator from a pre-fitted one (scripts/prefit_sens.py) and `optim.sens_hold_steps`
keeps it fixed for the first N steps (its gradients are dropped before clipping, so AdamW skips it: no update, no weight decay).
"""
import argparse
import csv
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from fastmri.data import transforms

from .data import batch_to_device, build_dataset, build_loader, repo_path
from .hub import weights_path
from .losses import WindowSSIMLoss
from .metrics import VolumeMetrics
from .model import build_model, sample_num_loops
from .sens import load_sens_init

REQUEUE_EXIT_CODE = 3


# ----------------------------------------------------------------------------- config
def load_config(path: str, sets=()) -> Dict:
    """YAML config plus dotted overrides 'a.b=value' (value parsed as YAML; '1e-4' accepted as a float)."""
    cfg = yaml.safe_load(open(path))
    for s in sets:
        key, val = s.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        parsed = yaml.safe_load(val)
        if isinstance(parsed, str):
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        node[parts[-1]] = parsed
    return cfg


# ----------------------------------------------------------------------------- RNG
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        try:
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])
        except RuntimeError:
            pass        # different GPU count than the saved run; not fatal


# ----------------------------------------------------------------------------- optimisation
def build_optimizer(model, ocfg: Dict) -> torch.optim.Optimizer:
    sens = {id(p) for p in model.sens_net.parameters()}
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if id(p) not in sens]},
              {"params": [p for p in params if id(p) in sens], "lr": float(ocfg["sens_lr"])}]
    return torch.optim.AdamW(groups, lr=float(ocfg["lr"]), weight_decay=float(ocfg["weight_decay"]))


def hold_sens(model, step: int, hold_steps: int) -> bool:
    """optim.sens_hold_steps: drop the sensitivity estimator's gradients (after backward, before clipping) for the first `hold_steps`
    steps, so AdamW skips its parameters entirely. Returns whether it was held."""
    if hold_steps <= 0 or step >= hold_steps:
        return False
    for p in model.sens_net.parameters():
        p.grad = None
    return True


def build_scheduler(optim, ocfg: Dict) -> torch.optim.lr_scheduler.LambdaLR:
    warm, size, gamma = int(ocfg["warmup_steps"]), int(ocfg["step_size"]), float(ocfg["gamma"])

    def factor(step):
        w = (step + 1) / warm if step < warm else 1.0
        return w * gamma ** (step // size)
    return torch.optim.lr_scheduler.LambdaLR(optim, factor)


# ----------------------------------------------------------------------------- checkpoints
def save_ckpt(path: Path, model, optim, sched, step: int, best: float, cfg: Dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(), "step": step, "best": best,
                "cfg": cfg, "rng": get_rng_state()}, tmp)
    os.replace(tmp, path)


def load_ckpt(path: Path, model, optim=None, sched=None, device="cpu") -> Dict:
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])          # strict
    if optim is not None:
        optim.load_state_dict(ck["optim"])
    if sched is not None:
        sched.load_state_dict(ck["sched"])
    if optim is not None and "rng" in ck:
        set_rng_state(ck["rng"])
    return ck


# ----------------------------------------------------------------------------- validation
@torch.no_grad()
def validate(model, loader, device, num_loops: int) -> Dict:
    """fastMRI-convention SSIM / PSNR / NMSE of the centre-slice output after `num_loops` loops, overall and per acceleration."""
    model.eval()
    metrics = VolumeMetrics()
    t0 = time.time()
    for batch in loader:
        b = batch_to_device(batch, device)
        img = model(b["masked_kspace"], b["mask"], b["num_low_frequencies"], num_loops=num_loops, loss_window=1)["images"][-1]
        target, output = transforms.center_crop_to_smallest(b["target"], img)
        for i in range(output.shape[0]):
            metrics.update(b["fname"][i], b["slice_num"][i], output[i].cpu().numpy(), target[i].cpu().numpy(),
                           float(b["max_value"][i]), b["accel"])
    res = metrics.compute()
    res["seconds"] = time.time() - t0
    model.train()
    return res


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action="extend", default=[], help="dotted.key=value overrides (may repeat)")
    ap.add_argument("--run_dir", default=None, help="default: <runs_root>/<run_name>")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    run_dir = Path(args.run_dir or repo_path(cfg["runs_root"]) / cfg["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"))

    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg, ocfg, lcfg = cfg["train"], cfg["optim"], cfg["loops"]

    model = build_model(cfg).to(device)
    if cfg["model"].get("sens_init"):   # a fresh run starts the estimator from a pre-fitted one (a resume below overwrites it)
        print(f"[train] sens_init: {load_sens_init(model.sens_net, str(weights_path(cfg['model']['sens_init'])))}")
    for blk in model.blocks:                # the shared block (or the untied ones)
        if tcfg.get("compile_block"):   # torch.compile the block in place: same weights and state_dict keys (only bf16 rounding order differs)
            blk.inner.compile(mode=str(tcfg["compile_block"]), dynamic=False)
        if tcfg.get("checkpoint_block"):  # recompute the block in backward: same gradients, activation memory of one block call
            blk.checkpoint_inner = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params {n_params / 1e6:.2f}M (sensitivity estimator {sum(p.numel() for p in model.sens_net.parameters()) / 1e6:.2f}M, "
          f"block(s) {sum(p.numel() for b in model.blocks for p in b.parameters()) / 1e6:.2f}M)")

    loss_fn = WindowSSIMLoss(cfg["loss"]["loops"]).to(device)
    optim = build_optimizer(model, ocfg)
    sched = build_scheduler(optim, ocfg)
    max_steps = int(tcfg["max_steps"])

    step, best = 0, -1.0
    if (run_dir / "ckpt_last.pt").exists():
        ck = load_ckpt(run_dir / "ckpt_last.pt", model, optim, sched, device)
        step, best = int(ck["step"]), float(ck["best"])
        print(f"[train] resumed from {run_dir / 'ckpt_last.pt'} at step {step} (best {best:.4f})")

    dcfg = cfg["data"]
    train_ds = build_dataset(dcfg, "train", train=True)
    val_ds = build_dataset(dcfg, "val", train=False)
    train_loader = build_loader(train_ds, True, int(dcfg["num_workers"]))
    val_loader = build_loader(val_ds, False, int(dcfg["num_workers"]))
    print(f"[train] train {len(train_ds)} slices / val {len(val_ds)} slices; device {device}")

    stop = {"requeue": False}
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__("requeue", True))

    log_path = run_dir / "log.csv"
    new_log = not log_path.exists() or step == 0
    log_f = open(log_path, "a", newline="")
    log_w = csv.writer(log_f)
    if new_log:
        log_w.writerow(["step", "loss", "lr", "loops", "gnorm", "steps_per_s", "elapsed", "alpha", "dc_weight"])
    events = open(run_dir / "events.jsonl", "a")

    def log_event(kind, payload):
        events.write(json.dumps({"kind": kind, "step": step, "time": time.time(), **payload}) + "\n")
        events.flush()

    rng = np.random.default_rng(int(cfg["seed"]) + step)       # loop counts; offset by the step so a resumed run does not repeat them
    sens_hold = int(ocfg.get("sens_hold_steps", 0) or 0)
    loss_window = int(lcfg["loss_window"])
    eval_loops = int(lcfg["eval_loops"])
    keep_steps = {int(s) for s in tcfg.get("keep_steps") or []}
    log_every = int(tcfg["log_every"])
    model.train()
    t_start = t_log = time.time()
    n_log, loss_acc, nonfinite = 0, 0.0, 0
    done = step >= max_steps
    while not done:
        for batch in train_loader:
            b = batch_to_device(batch, device)
            r = sample_num_loops(lcfg["train"], rng)
            try:
                out = model(b["masked_kspace"], b["mask"], b["num_low_frequencies"], num_loops=r, loss_window=loss_window)
                loss = loss_fn(out["images"], b["target"], b["max_value"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if hold_sens(model, step, sens_hold) and step == 0:
                    print(f"[train] sens_hold_steps: the sensitivity estimator is held fixed until step {sens_hold}")
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(ocfg["grad_clip"]))
                optim.step()
            except FloatingPointError as err:
                nonfinite += 1
                optim.zero_grad(set_to_none=True)
                print(f"[train] step {step}: {err} (skipped, total {nonfinite})")
                if nonfinite > int(tcfg["max_nonfinite"]):
                    raise
                continue
            sched.step()
            step += 1
            loss_acc += loss.item()
            n_log += 1

            if step % log_every == 0:
                now = time.time()
                log_w.writerow([step, f"{loss_acc / n_log:.5f}", f"{sched.get_last_lr()[0]:.2e}", r, f"{float(gnorm):.3e}",
                                f"{n_log / (now - t_log):.2f}", f"{now - t_start:.0f}", f"{model.alpha.item():.4f}",
                                f"{model.dc_weight.item():.4f}"])
                log_f.flush()
                print(f"[train] step {step}/{max_steps} loss {loss_acc / n_log:.4f} lr {sched.get_last_lr()[0]:.1e} r={r} "
                      f"{n_log / (now - t_log):.1f} it/s")
                t_log, n_log, loss_acc = now, 0, 0.0

            if step % int(tcfg["val_every"]) == 0 or step >= max_steps:
                res = validate(model, val_loader, device, eval_loops)
                log_event("val", {"loops": eval_loops, **res})
                print(f"[val] step {step} loops {eval_loops} ssim {res['all']['ssim']:.4f} psnr {res['all']['psnr']:.2f} "
                      + " ".join(f"{g}:{v['ssim']:.4f}" for g, v in res.items() if g.startswith("x")))
                if res["all"]["ssim"] > best:
                    best = res["all"]["ssim"]
                    save_ckpt(run_dir / "ckpt_best.pt", model, optim, sched, step, best, cfg)
                if step >= max_steps:          # final model: validation at other loop counts too
                    for n in lcfg.get("eval_loop_sweep", []):
                        r2 = validate(model, val_loader, device, int(n))
                        log_event("val_sweep", {"loops": int(n), **r2})
                        print(f"[val] loops {n} ssim {r2['all']['ssim']:.4f} psnr {r2['all']['psnr']:.2f}")

            if step in keep_steps:
                save_ckpt(run_dir / f"ckpt_step{step}.pt", model, optim, sched, step, best, cfg)
            if step % int(tcfg["ckpt_every"]) == 0 or step >= max_steps or stop["requeue"]:
                save_ckpt(run_dir / "ckpt_last.pt", model, optim, sched, step, best, cfg)
            if stop["requeue"]:
                print(f"[train] SIGUSR1: checkpoint saved at step {step}; exiting for requeue")
                sys.exit(REQUEUE_EXIT_CODE)
            if step >= max_steps:
                done = True
                break
    log_event("done", {"best_ssim": best, "elapsed": time.time() - t_start, "nonfinite_steps": nonfinite})
    print(f"[train] done. best val ssim {best:.4f}")


if __name__ == "__main__":
    main()
