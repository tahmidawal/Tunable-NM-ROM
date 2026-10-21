#!/bin/bash
# imp-fronts Phase B production TR-cov: arm (b) coverage retrain, seed 1, full recipe (run_seed.sh production), wall limit = the 4.4 GPU-h ceiling (Amendment 3 B8); exhaustion -> arm NOT RUN.
# Staged by stage_job.sh into /cluster/tufts/paralab/tawal01/imp-fronts/trcov/ and submitted from there (one job per
# directory). Body: train_body.sh. gpu partition only (never preempt), one A100-80G (the K1 control's GPU type).
#SBATCH --job-name=if_trcov
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:24:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/trcov/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/trcov/logs/%j.err
set -euo pipefail
export JOB=trcov ARM=cov MODE=production WALL_S=15840
cd /cluster/tufts/paralab/tawal01/imp-fronts/trcov
source ./train_body.sh
