#!/bin/bash
#SBATCH --job-name=loopmri_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/slurm-train-%j.out
# Usage (from the repo root; adapt the partition / GPU to your cluster): mkdir -p runs &&
#   CONFIG=configs/cc359_rs5_fd.yaml [EXTRA="--set run_name=x"] sbatch --gres=gpu:1 slurm/train.sbatch
# The job checkpoints and requeues itself at the 1-day limit (USR1 -> python saves ckpt_last.pt, exits 3 -> scontrol requeue) and
# resumes from ckpt_last.pt. NOTE: a plain `scancel` is caught as preemption and requeues too; to stop a run, `scontrol hold <id>` first.
set -uo pipefail
set -f
cd "$SLURM_SUBMIT_DIR" && source slurm/env.sh
python -u -m loopmri.train --config "$CONFIG" ${EXTRA:-} &
PID=$!
trap 'echo "forwarding USR1"; kill -USR1 $PID' USR1
trap 'echo "TERM received (preemption): checkpointing"; kill -USR1 $PID' TERM
wait $PID; CODE=$?
while kill -0 $PID 2>/dev/null; do wait $PID; CODE=$?; done
if [ "$CODE" -eq 3 ]; then
  echo "requeueing $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
fi
exit $CODE
