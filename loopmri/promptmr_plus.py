"""PromptMR+ (Xin et al., ECCV 2024), the baseline of the paper's tables, for evaluation with loopmri.evaluate.

The network is the authors' code, vendored unchanged in `vendor/promptmr_plus` (Rutgers non-commercial research licence). This wrapper
only re-runs its cascade loop so that, as in our research code, the cascades' denoisers run under bf16 autocast while the FFTs and data
consistency stay fp32, and so that the evaluation can read the image after every cascade. With the same weights it gives the images the
paper's PromptMR+ numbers were computed from (`iterate_images`, cascade 12 = the network's output).

Weights (configs/promptmr_plus/*.yaml hold the configs, the weights files store them):
  * CC359 brain: the AUTHORS' released weights (huggingface.co/hellopipu/PromptMR, weights/cc-brain/promptmr-plus-epoch=19-step=120320.ckpt).
    `weights/promptmr_plus/cc359_promptmr_plus_released.pt` is not distributed: on first use `convert_released` downloads the authors'
    Lightning checkpoint and converts it (hyper-parameters checked against the config, strict load).
  * fastMRI knee, CMRxRecon2023: trained by us with the authors' settings on the same data and schedule as LoopMR (the authors' released
    weights were trained on other data: fastMRI's full knee training set, and CMRxRecon2023 P001-P120, which includes the test patients),
    downloaded from huggingface.co/hym97/loopmri like LoopMR's weights.
"""
import hashlib
from pathlib import Path
from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
import yaml

import fastmri

from .vendor.promptmr_plus.coil_combine import sens_expand, sens_reduce
from .vendor.promptmr_plus.promptmr_v2 import PromptMR, PromptUnet

# The authors' CC359 PromptMR+ checkpoint, pinned to a revision of their Hugging Face repository.
RELEASED_CC359 = {"repo": "hellopipu/PromptMR", "revision": "fd1642e375e29ee515696d802179c099ee08d737",
                  "file": "weights/cc-brain/promptmr-plus-epoch=19-step=120320.ckpt",
                  "sha256": "acaa4b6c1ec79fb07189e7450624f50f57fe82c997645e265e5ab32f6db45288",
                  "config": "configs/promptmr_plus/cc359_promptmr_plus.yaml",
                  "weights": "promptmr_plus/cc359_promptmr_plus_released.pt"}
# Network hyper-parameters the Lightning checkpoint stores (its training-only ones are not compared).
NET_KEYS = ["num_cascades", "num_adj_slices", "n_feat0", "feature_dim", "prompt_dim", "sens_n_feat0", "sens_feature_dim",
            "sens_prompt_dim", "len_prompt", "prompt_size", "n_enc_cab", "n_dec_cab", "n_skip_cab", "n_bottleneck_cab", "no_use_ca",
            "learnable_prompt", "adaptive_input", "n_buffer", "n_history", "use_sens_adj"]


class PromptMRPlus(nn.Module):
    def __init__(self, amp: bool = True, mask_type: str = "cartesian", **net_kwargs):
        super().__init__()
        self.net = PromptMR(**net_kwargs)
        # Upstream's PromptUnet has `act=nn.PReLU()` as a default argument: one PReLU (one learnable slope) shared by every PromptUnet
        # built in the process. Within one PromptMR+ that ties the slope across its estimator and cascades (upstream's behaviour, kept);
        # across models it would tie this network to any other one built in the same process (loopmri/sens.py does the same).
        shared = next(d for d in PromptUnet.__init__.__defaults__ if isinstance(d, nn.PReLU))
        private = nn.PReLU()
        private.load_state_dict(shared.state_dict())
        for mod in self.net.modules():
            for name, child in list(mod._modules.items()):
                if child is shared:
                    mod._modules[name] = private
        self.adj = int(self.net.num_adj_slices)
        self.num_cascades = int(self.net.num_cascades)
        self.amp = amp
        self.mask_type = (mask_type,)

    def _cascade(self, block, img, img_zf, latent, mask, S, history):
        """PromptMRBlock.forward with autocast around the denoiser only."""
        zero = torch.zeros(1, 1, 1, 1, 1, dtype=img.dtype, device=img.device)
        ffx = sens_reduce(torch.where(mask, sens_expand(img, S, self.adj), zero), S, self.adj)
        n_buffer = block.model.n_buffer
        buffer = torch.cat([ffx] + [latent] * (n_buffer - 3) + [img_zf, ffx - img_zf], dim=1) if n_buffer > 0 else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and img.is_cuda):
            term, latent, history = block.model(img, history, buffer)
        latent = latent.float() if latent is not None else None
        return img - (ffx - img_zf) * block.dc_weight - term.float(), latent, history

    def _centre(self, x):
        """Centre-slice part of a tensor stacked slice-major along dim 1."""
        return x if self.adj == 1 else torch.chunk(x, self.adj, dim=1)[self.adj // 2]

    @torch.no_grad()
    def iterate_images(self, masked_kspace, mask, num_low_frequencies, max_loops: int = None) -> Iterator[Tuple[int, torch.Tensor]]:
        """Yield (t, centre slice's RSS image [B, H, W]) for t = 0 (zero-filled) .. min(max_loops, num_cascades); t >= 1 is the image
        after cascade t, |S x_t| combined over the coils, as the network's own output."""
        S = self.net.sens_net(masked_kspace, mask, num_low_frequencies, self.mask_type)
        img_zf = sens_reduce(masked_kspace, S, self.adj)
        img, latent, history = img_zf.clone(), img_zf.clone(), None
        Sc = self._centre(S)
        yield 0, fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(self._centre(masked_kspace))), dim=1)
        for t, block in enumerate(self.net.cascades[:min(int(max_loops or self.num_cascades), self.num_cascades)], 1):
            img, latent, history = self._cascade(block, img, img_zf, latent, mask, S, history)
            yield t, fastmri.rss(fastmri.complex_abs(fastmri.complex_mul(self._centre(img), Sc)), dim=1)


def is_promptmr_plus(cfg: Dict) -> bool:
    return cfg["model"].get("name") == "promptmr_plus"


def build_promptmr_plus(cfg: Dict) -> PromptMRPlus:
    m = dict(cfg["model"])
    m.pop("name")
    if int(m["num_adj_slices"]) != int(cfg["data"]["num_adj_slices"]):
        raise ValueError("model.num_adj_slices and data.num_adj_slices must be equal")
    if int(cfg["loops"]["eval_loops"]) != int(m["num_cascades"]):
        raise ValueError("loops.eval_loops must equal model.num_cascades (the network's output)")
    return PromptMRPlus(amp=cfg["train"]["amp"], **m)


def convert_released(out: Path) -> Path:
    """Download the authors' CC359 PromptMR+ checkpoint (RELEASED_CC359) and write it to `out` as a loopmri weights file
    {cfg, model, step}. Aborts, writing nothing, if the file's sha256, a stored network hyper-parameter or a strict load disagrees."""
    from huggingface_hub import hf_hub_download

    from .data import repo_path
    r = RELEASED_CC359
    print(f"[weights] converting the authors' PromptMR+ CC359 checkpoint: huggingface.co/{r['repo']} {r['file']} @ {r['revision'][:7]}")
    src = hf_hub_download(r["repo"], r["file"], revision=r["revision"])
    h = hashlib.sha256()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != r["sha256"]:
        raise RuntimeError(f"{src}: sha256 {h.hexdigest()} is not the expected {r['sha256']}")
    cfg = yaml.safe_load(repo_path(r["config"]).read_text())
    ck = torch.load(src, map_location="cpu", weights_only=True)
    hp = ck.get("hyper_parameters", {})
    bad = [f"{k}: checkpoint {hp[k]!r} vs config {cfg['model'].get(k)!r}" for k in NET_KEYS if k in hp and hp[k] != cfg["model"].get(k)]
    if bad:
        raise RuntimeError("the checkpoint's hyper-parameters differ from " + r["config"] + ": " + "; ".join(bad))
    # Lightning's 'promptmr.*' -> our 'net.*'; the module's other entries (loss buffers) are not part of the network.
    sd = {"net." + k[len("promptmr."):]: v for k, v in ck["state_dict"].items() if k.startswith("promptmr.")}
    model = build_promptmr_plus(cfg)
    model.load_state_dict(sd, strict=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg, "model": model.state_dict(), "step": int(ck["global_step"]),
                "source": f"huggingface.co/{r['repo']}@{r['revision']} {r['file']} (sha256 {r['sha256']})"}, out)
    print(f"[weights] wrote {out}")
    return out
