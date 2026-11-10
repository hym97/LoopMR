#!/bin/bash
#SBATCH --job-name=loopmri_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-eval-%j.out
# Usage (from the repo root): CKPT=weights/cc359_rs5_fd_146k.pt OUT=runs/eval/cc359_fd_test.json [LOOPS="4 8 12"] sbatch --gres=gpu:1 slurm/eval.sbatch
# Any of the three datasets: the protocol and metric follow from the checkpoint (loopmri/evaluate.py).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" && source slurm/env.sh
python -u -m loopmri.evaluate --ckpt "$CKPT" --out "$OUT" ${LOOPS:+--loops $LOOPS}
