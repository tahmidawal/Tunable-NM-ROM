#!/bin/bash
# imp-fronts Phase B smoke SM-b (training part; DESIGN B9 as amended; Amendment 4: 0.45 GPU-h): arm cov, low-nu upwind data for 576 + 4,032 (+8 test) trajectories with the Newton <= 1e-8 gates, in-job R0 replacement gate (C-B4), C-B1 guard rejection, run_seed.sh smoke mode, per-phase timing JSON. The r2acc policy round, C-B6 fixture and refjob steps of SM-b are not part of this training template.
# Staged by stage_job.sh into /cluster/tufts/paralab/tawal01/imp-fronts/smB/ and submitted from there (one job per
# directory). Body: train_body.sh. gpu partition only (never preempt), one A100-80G (the K1 control's GPU type).
#SBATCH --job-name=if_smB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:27:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/smB/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/smB/logs/%j.err
set -euo pipefail
export JOB=smB ARM=cov MODE=smoke WALL_S=1620
cd /cluster/tufts/paralab/tawal01/imp-fronts/smB
source ./train_body.sh
