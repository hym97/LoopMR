# Sourced by the batch scripts: the Python environment (torch 2.8 / CUDA 12.8, fastmri 0.3.0, einops, timm, h5py, scikit-image, scipy).
# Set CONDA_SH / CONDA_ENV, or edit the line below for your machine.
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV:-loopmri}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job ${SLURM_JOB_ID:-local} on $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
