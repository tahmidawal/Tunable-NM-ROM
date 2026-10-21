#!/bin/bash
# imp-fronts Phase B smoke SM-b (DESIGN Amendment 11: ceiling 0.4 GPU-h (Amendment 14), wall 24 min, self-timer 90 %): arm cov. C-B1 guard rejection; full 576 + 4,032 low-nu upwind generation timed per population with Newton residual stats and C-B4 (smoke_datagen.py); chaining slice = run_seed.sh TRAIN2D_DRYRUN mode with the time left; phase_times.py -> training projection. (SM-r is a separate job, not this template.)
# Staged by stage_job.sh into /cluster/tufts/paralab/tawal01/imp-fronts/smB/ and submitted from there (one job per
# directory). Body: train_body.sh. gpu partition only (never preempt), one A100-80G (the K1 control's GPU type).
#SBATCH --job-name=if_smB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:24:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/smB/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/smB/logs/%j.err
set -euo pipefail
export JOB=smB ARM=cov MODE=smoke WALL_S=1440 SELF_TIMER_FRAC=0.9
cd /cluster/tufts/paralab/tawal01/imp-fronts/smB
source ./train_body.sh
