#!/bin/bash
# imp-fronts Phase B production TR-fid: arm (a) fidelity retrain, seed 1, full recipe (run_seed.sh production). Amendment 11: ceiling 4.6 GPU-h = wall 4.6 h, no self-timer (SELF_TIMER_FRAC=1); exhaustion -> arm NOT RUN.
# Staged by stage_job.sh into /cluster/tufts/paralab/tawal01/imp-fronts/trfid/ and submitted from there (one job per
# directory). Body: train_body.sh. gpu partition only (never preempt), one A100-80G (the K1 control's GPU type).
#SBATCH --job-name=if_trfid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:36:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/trfid/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/trfid/logs/%j.err
set -euo pipefail
export JOB=trfid ARM=fid MODE=production WALL_S=16560 SELF_TIMER_FRAC=1
cd /cluster/tufts/paralab/tawal01/imp-fronts/trfid
source ./train_body.sh
