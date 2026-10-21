#!/bin/bash
# imp-fronts Phase B smoke SM-a (DESIGN Amendment 11: ceiling 0.2 GPU-h, wall 12 min, self-timer 90 %): arm fid. C-B1 guard rejection; full 576 + 4,032 PS data generation timed per population with C-B3 on dev6 + PS self-convergence (smoke_datagen.py); chaining slice = run_seed.sh TRAIN2D_DRYRUN mode with the time left; phase_times.py -> training projection (K1 max 4.17 h + measured generation increment).
# Staged by stage_job.sh into /cluster/tufts/paralab/tawal01/imp-fronts/smA/ and submitted from there (one job per
# directory). Body: train_body.sh. gpu partition only (never preempt), one A100-80G (the K1 control's GPU type).
#SBATCH --job-name=if_smA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:12:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/smA/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/smA/logs/%j.err
set -euo pipefail
export JOB=smA ARM=fid MODE=smoke WALL_S=720 SELF_TIMER_FRAC=0.9
cd /cluster/tufts/paralab/tawal01/imp-fronts/smA
source ./train_body.sh
