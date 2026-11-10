"""The released weights live on the Hugging Face Hub (HF_REPO); a file below weights/ that is not on disk yet is downloaded from there on
first use, to the same path. The one exception is PromptMR+'s CC359 weights, which are the authors' own: they are downloaded from the
authors' repository and converted (loopmri.promptmr_plus.convert_released)."""
from pathlib import Path

from .data import REPO_ROOT, repo_path

HF_REPO = "hym97/loopmri"
WEIGHTS = REPO_ROOT / "weights"


def weights_path(p) -> Path:
    """Path of a checkpoint or weights file (as given, else relative to the repo); a released file below weights/ that is missing is
    downloaded from HF_REPO first."""
    path = Path(p) if Path(p).exists() else repo_path(p)
    if path.exists() or not path.resolve().is_relative_to(WEIGHTS):
        return path
    rel = path.resolve().relative_to(WEIGHTS).as_posix()
    from .promptmr_plus import RELEASED_CC359, convert_released
    if rel == RELEASED_CC359["weights"]:
        return convert_released(path.resolve())
    from huggingface_hub import hf_hub_download
    print(f"[weights] {rel} not found locally: downloading it from huggingface.co/{HF_REPO}")
    return Path(hf_hub_download(HF_REPO, rel, local_dir=WEIGHTS))
