"""Coil-sensitivity estimation: PromptMR+'s estimator (vendor/promptmr_plus, its SensitivityModel) at upstream's default sizes.

The estimator sees only the fully sampled CALIBRATION region of k-space (`model.sens_calibration`):
  * disc  (CC359, 2-D Poisson-disc masks): the central disc, radius = the loader's num_low_frequencies (16); upstream's `poisson_disc`;
  * lines (fastMRI knee, CMRxRecon; line masks): the central band of fully sampled columns, num_low_frequencies wide; upstream's
    `cartesian`.
With A adjacent slices (A > 1) it is upstream's adjacent estimation (use_sens_adj): one coil's A slices enter the PromptUnet together as
2A channels and all A maps of that coil come out at once. The maps have unit root-sum-of-squares over the coils of each slice.
1.87M parameters.

`plain_maps` are the parameter-free maps of the same calibration region (inverse FFT, RSS normalisation per slice): the target of
scripts/prefit_sens.py, which pre-fits the estimator before knee training (model.sens_init loads the result, `load_sens_init`).
"""
import torch
import torch.nn as nn

import fastmri

from .vendor.promptmr_plus.promptmr_v2 import PromptUnet
from .vendor.promptmr_plus.promptmr_v2 import SensitivityModel as PromptMRSensitivityModel

MASK_TYPE = {"disc": "poisson_disc", "lines": "cartesian"}


class PromptMRSensNet(nn.Module):
    def __init__(self, adj: int = 1, calibration: str = "disc"):
        super().__init__()
        if calibration not in MASK_TYPE:
            raise ValueError(f"model.sens_calibration must be disc or lines, got {calibration}")
        self.net = PromptMRSensitivityModel(
            num_adj_slices=int(adj), n_feat0=24, feature_dim=[36, 48, 60], prompt_dim=[12, 24, 36], len_prompt=[5, 5, 5],
            prompt_size=[64, 32, 16], n_enc_cab=[2, 3, 3], n_dec_cab=[2, 2, 3], n_skip_cab=[1, 1, 1], n_bottleneck_cab=3,
            no_use_ca=False, mask_center=True, learnable_prompt=False, use_sens_adj=True)
        # Upstream's PromptUnet has `act=nn.PReLU()` as a DEFAULT ARGUMENT, i.e. one PReLU module (one learnable slope) shared by every
        # PromptUnet built in the process. Inside one PromptMR+ model that ties the slope across its estimator and cascades (upstream's
        # behaviour); here it would tie this estimator to any other PromptMR+ net built in the same process (e.g. two models loaded in one
        # notebook: the second load would overwrite the first one's slope). Give the estimator its own PReLU, still shared by all of ITS
        # blocks as upstream. Parameter names are unchanged.
        shared = next(d for d in PromptUnet.__init__.__defaults__ if isinstance(d, nn.PReLU))
        private = nn.PReLU()
        for mod in self.net.modules():
            for name, child in list(mod._modules.items()):
                if child is shared:
                    mod._modules[name] = private
        self.mask_type = (MASK_TYPE[calibration],)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, num_low_frequencies) -> torch.Tensor:
        """[B, A*C, H, W, 2] masked k-space -> maps of the same shape."""
        return self.net(masked_kspace, mask, num_low_frequencies, self.mask_type)


# ----------------------------------------------------------------------------- plain calibration maps (prefit target)
def center_disc(k: torch.Tensor, radius: int) -> torch.Tensor:
    """Keep the central disc (distance from [H//2, W//2] <= radius) of k-space [..., H, W, 2], zero the rest."""
    h, w = k.shape[-3], k.shape[-2]
    yy = torch.arange(h, device=k.device).view(h, 1) - h // 2
    xx = torch.arange(w, device=k.device).view(1, w) - w // 2
    return k * ((yy ** 2 + xx ** 2) <= radius ** 2).to(k.dtype).unsqueeze(-1)


def center_lines(k: torch.Tensor, mask: torch.Tensor, num_low_frequencies) -> torch.Tensor:
    """Keep the fully sampled central band of k-space COLUMNS [..., H, W, 2], zero the rest (fastMRI's SensitivityModel calibration).
    Band width = num_low_frequencies, else read from the mask [B, 1, 1, W, 1] (the symmetric fully sampled run around W // 2)."""
    b, w = k.shape[0], k.shape[-2]
    if num_low_frequencies is None or (torch.as_tensor(num_low_frequencies) == 0).all():
        sq = mask[:, 0, 0, :, 0].to(torch.int8)
        cent = sq.shape[1] // 2
        left = torch.argmin(sq[:, :cent].flip(1), dim=1)
        right = torch.argmin(sq[:, cent:], dim=1)
        n = torch.clamp(2 * torch.min(left, right), min=1)
    else:
        n = torch.as_tensor(num_low_frequencies, device=k.device).reshape(-1).long()
        n = n.expand(b) if n.numel() == 1 else n
    pad = (w - n + 1) // 2
    out = torch.zeros_like(k)
    for i in range(b):
        p0, p1 = int(pad[i]), int(pad[i] + n[i])
        out[i, ..., p0:p1, :] = k[i, ..., p0:p1, :]
    return out


def calibration_region(y: torch.Tensor, mask, num_low_frequencies, calibration: str) -> torch.Tensor:
    if calibration == "disc":
        return center_disc(y.float(), int(torch.as_tensor(num_low_frequencies).reshape(-1)[0]))
    return center_lines(y.float(), mask, num_low_frequencies)


def plain_maps(y: torch.Tensor, mask, num_low_frequencies, adj: int, calibration: str) -> torch.Tensor:
    """Parameter-free maps: inverse FFT of the calibration region, divided by its RSS over each slice's coils (+ 1e-6 of the max)."""
    x = fastmri.ifft2c(calibration_region(y, mask, num_low_frequencies, calibration))
    b, ac = x.shape[:2]
    xs = x.reshape(b * adj, ac // adj, *x.shape[2:])
    r = fastmri.rss_complex(xs, dim=1)                                    # [B*A, H, W]
    eps = 1e-6 * r.flatten(1).amax(1).view(-1, 1, 1) + 1e-12
    return (xs / (r + eps).unsqueeze(1).unsqueeze(-1)).reshape(x.shape)


def load_sens_init(sens_net: nn.Module, path: str) -> str:
    """model.sens_init: copy a pre-fitted estimator (keys `sens_net.*`, e.g. scripts/prefit_sens.py's output) into `sens_net`, strictly."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    donor = {k[len("sens_net."):]: v for k, v in sd.items() if k.startswith("sens_net.")}
    if not donor:
        raise ValueError(f"{path} holds no sens_net.* weights")
    sens_net.load_state_dict(donor, strict=True)
    return f"{len(donor)} tensors from {path}"
