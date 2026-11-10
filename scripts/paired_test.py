#!/usr/bin/env python
"""Paired significance tests, as in the paper: two-sided Wilcoxon signed-rank test over test units (no multiple-comparison correction),
with the mean difference, the number of units won and the two-sided sign test.

  python scripts/paired_test.py --ours runs/eval/cc359_fd_test.json            # LoopMR vs the baselines in results/
  python scripts/paired_test.py --a runs/<run A>/eval.json --b runs/<run B>/eval.json [--names A B]

--ours: a loopmri.evaluate result (any of the three datasets) against E2E-VarNet / PromptMR / PromptMR+ from
  results/baselines_per_volume.json, per acceleration. A PromptMR+ result (weights/promptmr_plus/) here checks the stored PromptMR+
  numbers: its row against PromptMR+ shows the difference.
--a / --b: any two loopmri.evaluate results on the same data (e.g. two training runs); difference = a - b.
Units (the paper's): CC359 test volumes, every volume at R5 and at R10; knee held-out volumes at their one R; CMRxRecon patients, every
patient at R4, R8 and R10 (the mean of that patient's files scored at that R).
"""
import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATASET = {"cc_test": "cc359", "heldout": "knee", "cmr_challenge": "cmr23"}


def units(path: str, loops: int = None) -> (str, dict):
    """(dataset, {"x<R>": {unit: {"ssim", "psnr"}}}) from a loopmri.evaluate json, at its evaluation loop count (or `loops`)."""
    res = json.loads(Path(path).read_text())
    ds = DATASET[res["protocol"]]
    if loops is None and len(res["loops"]) > 1:
        raise SystemExit(f"{path} holds several loop counts {res['loops']}: pick one with --loops")
    r = res["results"][f"loops{loops or res['loops'][0]}"]
    out = defaultdict(dict)
    if ds == "cc359":
        for R, v in r.items():
            out[R] = {f: dict(p["challenge"]) for f, p in v["volumes"].items()}
        return ds, dict(out)
    if ds == "knee":
        if "fixed_R_per_volume" not in r:
            raise SystemExit(f"{path}: needs the default knee evaluation (one fixed R per volume), not --all-accels")
        for f, v in r["fixed_R_per_volume"]["volumes"].items():
            out[f"x{v['accel']}"][f] = {"ssim": v["ssim"], "psnr": v["psnr"]}
        return ds, dict(out)
    if "fixed_R_per_volume" in r:
        raise SystemExit(f"{path}: an old CMRxRecon result (one R per file); rerun loopmri.evaluate (every file at every R)")
    per_patient = defaultdict(list)
    for R, res_R in r.items():
        for f, v in res_R["volumes"].items():
            per_patient[(R, f.split("_")[0])].append(v)
    for (R, p), vs in per_patient.items():
        out[R][p] = {m: float(np.mean([v[m] for v in vs])) for m in ("ssim", "psnr")}
    return ds, dict(out)


def sign_p(wins: int, n: int) -> float:
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a: dict, b: dict) -> dict:
    """a, b: {unit: {ssim, psnr}} -> per metric: means over the units of a and b, mean difference a - b, wins of a, n, Wilcoxon p,
    sign-test p."""
    if set(a) != set(b):
        raise SystemExit(f"the two results have different units: only in a {sorted(set(a) - set(b))}, only in b {sorted(set(b) - set(a))}")
    shared, out = sorted(a), {}
    for m in ("ssim", "psnr"):
        d = np.array([a[u][m] - b[u][m] for u in shared])
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        out[m] = {"a": float(np.mean([a[u][m] for u in shared])), "b": float(np.mean([b[u][m] for u in shared])),
                  "diff": float(d.mean()), "wins": wins, "n": len(shared),
                  "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, "sign_p": sign_p(wins, wins + losses) if wins + losses else 1.0}  # ties dropped
    return out


def show(title: str, rows: list) -> None:
    print(f"\n{title}")
    print(f"  {'R':>4s}  {'comparison':34s} {'metric':6s} {'mean A':>8s} {'mean B':>8s} {'A - B':>9s} {'wins':>6s} {'Wilcoxon p':>11s} {'sign p':>8s}")
    for R, name, res in rows:
        for m, v in res.items():
            print(f"  {R:>4s}  {name:34s} {m:6s} {v['a']:8.4f} {v['b']:8.4f} {v['diff']:+9.4f} {v['wins']:>2d}/{v['n']:<3d} "
                  f"{v['wilcoxon_p']:11.2e} {v['sign_p']:8.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", help="loopmri.evaluate result of LoopMR (or of PromptMR+, to check it against the stored baseline)")
    ap.add_argument("--baselines", default=str(ROOT / "results/baselines_per_volume.json"))
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--loops", type=int, default=None, help="loop count to read (default: the result's evaluation loop count)")
    ap.add_argument("--out", default=None, help="also write the table as json")
    args = ap.parse_args()

    rows = []
    if args.ours:
        ds, ours = units(args.ours, args.loops)
        who = json.loads(Path(args.ours).read_text()).get("model", "LoopMR")
        base = json.loads(Path(args.baselines).read_text())[ds]
        for R in sorted(ours, key=lambda r: int(r[1:])):
            for name, b in base.get(R, {}).items():
                rows.append((R, f"{who} vs {name}", compare(ours[R], b["units"])))
        title = f"{ds}: A = {who} ({args.ours}), B = baseline; unit = " + json.loads(Path(args.baselines).read_text())["_about"]["units"][ds]
    elif args.a and args.b:
        (da, ua), (db, ub) = units(args.a, args.loops), units(args.b, args.loops)
        if da != db or set(ua) != set(ub):
            raise SystemExit("the two results are not on the same data / accelerations")
        for R in sorted(ua, key=lambda r: int(r[1:])):
            rows.append((R, f"{args.names[0]} vs {args.names[1]}", compare(ua[R], ub[R])))
        title = f"{da}: A = {args.names[0]} ({args.a}), B = {args.names[1]} ({args.b})"
    else:
        sys.exit("give --ours, or --a and --b")
    show(title, rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps([{"R": R, "comparison": n, **res} for R, n, res in rows], indent=1))


if __name__ == "__main__":
    main()
