#!/bin/bash
# imp-fronts Phase B smoke SM-c (DESIGN Amendments 19-21: ceiling 0.45 GPU-h, wall 27 min, self-timer 90 %): arm (c) bnd. C-B1 guard rejection; tests/test_arm_c.py identity tests on the GPU; paired incumbent / arm-(c) timing after burn-in (Stage A sub and full-grid steps, Stage B extraction kernel; repetitions persisted); TRAIN2D_DRYRUN chaining slice of the bnd tree; phase_times.py smokec projection vs 16,440 s.
# Staged by stage_job.sh into /cluster/tufts/paralab/tawal01/imp-fronts/smC/ and submitted from there (one job per
# directory). Body: train_body.sh. gpu partition only (never preempt), one A100-80G (the K1 control's GPU type).
#SBATCH --job-name=if_smC
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:27:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/smC/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/smC/logs/%j.err
set -euo pipefail
export JOB=smC ARM=bnd MODE=smokec WALL_S=1620 SELF_TIMER_FRAC=0.9
cd /cluster/tufts/paralab/tawal01/imp-fronts/smC
source ./train_body.sh
