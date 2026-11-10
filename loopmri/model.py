"""The looped (weight-tied) unrolled reconstruction: ONE denoiser block applied r times, E2E-VarNet's k-space state and soft data consistency.

Inputs: y = masked multicoil k-space of A adjacent slices, slice-major [B, A*C, H, W, 2] (A = 5: the centre slice +- 2; CMRxRecon: the
centre time frame +- 2), the mask, and the size of the fully sampled calibration region (CC359: disc radius; knee / CMRxRecon: number of
central lines, `model.sens_calibration`). Only the centre slice is scored and returned; the neighbours are reconstructed along with it.

Once per call:
    S      = sens_net(y)                                 PromptMR+'s sensitivity estimator, maps [B, A*C, H, W, 2]
    e      = A^H y = sens_reduce(y, S)                   zero-filled images [B, A, H, W, 2]
    k_0    = y,   L_0 = K copies of e's centre slice     K carried latent images [B, K, H, W, 2]

One loop (the same block and the same two scalars at every t):
    x_t          = sens_reduce(k_t, S)                    current images            [B, A, ...]
    g_t          = A^H A x_t                              normal operator applied   [B, A, ...]
    dx, L_{t+1}  = block(x_t, g_t, L_t, e, g_t - e)       PromptMR+'s cascade inputs, in its order; outputs A updates + K new latents
    k_{t+1}      = k_t - alpha * (dc_weight * M (k_t - y) + sens_expand(dx, S))
The block sees 2 * (4A + K) real channels and returns 2 * (A + K). The latents are free memory: never scored, never data-consistent,
replaced by the block's output at every loop (PromptMR+'s "latent" image, K = 4 of them instead of 1).

Training with r loops and gradient window T: loops 1 .. r - T run without gradient, the last T loops carry gradient (truncated
backprop). `forward` returns the centre slice's RSS image after each of those T loops; the loss decides which of them it scores.
  * sampled depth (supported, no config here):  r ~ loguniform{4..32} per step, T = 8, every iterate in the window scored (linear weights), evaluated at 8 loops;
  * fixed:   r = T = 12 (full backprop through all loops), only the final output scored, evaluated at 12 loops.

Untied control (`model.untied: n`, the distinct-weight counterpart, not used by the released configs): n blocks of the same architecture with SEPARATE weights, loop t
uses block t (a standard unrolled network with n cascades); everything else (inputs, latents, soft DC, the shared alpha and dc_weight, the
sensitivity estimator, loss) is unchanged. At most n loops exist, so train it with a fixed n loops and a gradient window of n.
"""
import math
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .block import NormWrapper, SDUMRestormer
from .mri import centre_chunk, kspace_to_rss, normal_op, sens_expand, sens_reduce, soft_dc
from .sens import PromptMRSensNet


def sample_num_loops(train_loops: Dict, rng: np.random.Generator) -> int:
    """Loop count of one training step. {dist: fixed, value: r} or {dist: loguniform, min: a, max: b}."""
    dist = train_loops["dist"]
    if dist == "fixed":
        return int(train_loops["value"])
    if dist == "loguniform":
        lo, hi = int(train_loops["min"]), int(train_loops["max"])
        r = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
        return min(max(r, lo), hi)
    raise ValueError(f"loops.train.dist must be fixed or loguniform, got {dist}")


class LoopedRecon(nn.Module):
    def __init__(self, num_adj_slices: int = 5, n_latent: int = 4, rsdum: Optional[Dict] = None, alpha_init: float = 0.5,
                 dc_weight_init: float = 1.0, eval_loops: int = 8, amp: bool = True, sens_calibration: str = "disc", untied: int = 0):
        super().__init__()
        self.A = int(num_adj_slices)
        self.K = int(n_latent)
        self.amp = bool(amp)                   # bf16 autocast around the block only; FFTs, maps and data consistency stay fp32
        self.eval_loops = int(eval_loops)
        self.sens_net = PromptMRSensNet(adj=self.A, calibration=sens_calibration)
        self.untied = int(untied)

        def make_block():
            inner = SDUMRestormer(in_chans=2 * (4 * self.A + self.K), out_chans=2 * (self.A + self.K), **(rsdum or {}))
            return NormWrapper(inner, pad_multiple=inner.pad_multiple)
        if self.untied:
            self.denoisers = nn.ModuleList([make_block() for _ in range(self.untied)])
        else:
            self.denoiser = make_block()
        self.dc_weight = nn.Parameter(torch.full((1,), float(dc_weight_init)))
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))

    @property
    def blocks(self):
        """The block(s): [the shared one], or the n untied ones."""
        return list(self.denoisers) if self.untied else [self.denoiser]

    # ------------------------------------------------------------------ pieces
    def init_state(self, masked_kspace, mask, num_low_frequencies) -> Tuple[torch.Tensor, ...]:
        """Maps, zero-filled images, k_0 and L_0 (see the module docstring)."""
        y = masked_kspace
        S = self.sens_net(y, mask, num_low_frequencies)
        e = sens_reduce(y, S, self.A)
        lat = centre_chunk(e, self.A).repeat(1, self.K, 1, 1, 1)
        return y, S, e, y.clone(), lat

    def step(self, k, lat, y, mask, S, e, t: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """One loop: (k_t, L_t) -> (k_{t+1}, L_{t+1}). `t` (0-based) selects the block of an untied model; the tied model ignores it."""
        net = self.denoisers[t] if self.untied else self.denoiser
        x = sens_reduce(k, S, self.A)
        g = normal_op(x, mask, S, self.A)
        extras = [g, lat, e, g - e]
        if self.amp and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x, extras)
            out = out.float()
        else:
            out = net(x, extras)
        dx, lat = out[:, :self.A], out[:, self.A:]
        k = k - self.alpha * (soft_dc(k, y, mask, self.dc_weight) + sens_expand(dx, S, self.A))
        return k, lat

    def centre_image(self, k: torch.Tensor) -> torch.Tensor:
        """RSS image of the centre slice [B, H, W]: what is scored and returned."""
        return kspace_to_rss(centre_chunk(k, self.A))

    # ------------------------------------------------------------------ training / validation
    def forward(self, masked_kspace, mask, num_low_frequencies, num_loops: Optional[int] = None,
                loss_window: Optional[int] = None) -> Dict:
        """r = num_loops (default eval_loops) loops; the last min(loss_window, r) carry gradient and their images are returned."""
        y, S, e, k, lat = self.init_state(masked_kspace, mask, num_low_frequencies)
        r = int(num_loops or self.eval_loops)
        if self.untied:
            r = min(r, self.untied)                     # an untied model has no loop beyond its last block
        T = min(int(loss_window or r), r)
        with torch.no_grad():
            for t in range(r - T):
                k, lat = self.step(k, lat, y, mask, S, e, t)
        # Only k is cut here. When r == T no loop ran above and L_0 = copies of e = A^H y still carries the gradient path into the
        # sensitivity estimator; when r > T, k and L come out of no_grad loops and carry no gradient anyway.
        k = k.detach()
        images = []
        for t in range(r - T, r):
            k, lat = self.step(k, lat, y, mask, S, e, t)
            if not torch.isfinite(k).all():
                raise FloatingPointError("non-finite k-space state inside the loop")
            images.append(self.centre_image(k))
        return {"images": images, "num_loops": r}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def iterate(self, masked_kspace, mask, num_low_frequencies, max_loops: int) -> Iterator[Tuple[int, torch.Tensor]]:
        """Yield (t, centre slice's k-space k_t) for t = 0 (the zero-filled input) .. max_loops."""
        y, S, e, k, lat = self.init_state(masked_kspace, mask, num_low_frequencies)
        yield 0, centre_chunk(k, self.A)
        if self.untied:
            max_loops = min(int(max_loops), self.untied)
        for t in range(1, int(max_loops) + 1):
            k, lat = self.step(k, lat, y, mask, S, e, t - 1)
            yield t, centre_chunk(k, self.A)

    @torch.no_grad()
    def reconstruct(self, masked_kspace, mask, num_low_frequencies, loops: Sequence[int] = ()) -> Dict[int, torch.Tensor]:
        """Centre-slice RSS images {t: [B, H, W]} after each loop count in `loops` (default: eval_loops), one pass."""
        loops = sorted(set(int(t) for t in loops)) or [self.eval_loops]
        out = {}
        for t, k in self.iterate(masked_kspace, mask, num_low_frequencies, max_loops=loops[-1]):
            if t in loops:
                out[t] = kspace_to_rss(k)
        return out


def build_model(cfg: Dict) -> LoopedRecon:
    m = cfg["model"]
    if int(m["num_adj_slices"]) != int(cfg["data"]["num_adj_slices"]):
        raise ValueError("model.num_adj_slices and data.num_adj_slices must be equal")
    return LoopedRecon(num_adj_slices=m["num_adj_slices"], n_latent=m["n_latent"], rsdum=m.get("rsdum"), alpha_init=m["alpha_init"],
                       dc_weight_init=m["dc_weight_init"], eval_loops=cfg["loops"]["eval_loops"], amp=cfg["train"]["amp"],
                       sens_calibration=m.get("sens_calibration", "disc"), untied=int(m.get("untied", 0)))
