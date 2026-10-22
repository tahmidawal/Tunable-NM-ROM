#!/bin/bash
# imp-fronts Phase B production TR-bnd: arm (c) boundary-compatible bank (sine_even, DESIGN Amendments 19-21), seed 1, full recipe (run_seed.sh forced production). Ceiling 4.6 GPU-h = wall 4.6 h, no self-timer; run only if SM-c certified the projection (<= 16,440 s); exhaustion or failure -> arm NOT RUN.
# Staged by stage_job.sh into /cluster/tufts/paralab/tawal01/imp-fronts/trbnd/ and submitted from there (one job per
# directory). Body: train_body.sh. gpu partition only (never preempt), one A100-80G (the K1 control's GPU type).
#SBATCH --job-name=if_trbnd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:36:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/trbnd/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/trbnd/logs/%j.err
set -euo pipefail
export JOB=trbnd ARM=bnd MODE=production WALL_S=16560 SELF_TIMER_FRAC=1
cd /cluster/tufts/paralab/tawal01/imp-fronts/trbnd
source ./train_body.sh
