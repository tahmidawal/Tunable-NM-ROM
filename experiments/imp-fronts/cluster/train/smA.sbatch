#!/bin/bash
# imp-fronts Phase B smoke SM-a (DESIGN B9 as amended; Amendment 4: 0.45 GPU-h): arm fid, PS data for 576 + 4,032 (+8 test) trajectories, in-job R0 replacement gate (C-B3 on dev6 + PS self-convergence), C-B1 guard rejection, run_seed.sh smoke mode (Stage A 3,000 steps incl. 500 full-spatial, Stage B full data with head fit 2,000 steps, rotation), per-phase timing JSON.
# Staged by stage_job.sh into /cluster/tufts/paralab/tawal01/imp-fronts/smA/ and submitted from there (one job per
# directory). Body: train_body.sh. gpu partition only (never preempt), one A100-80G (the K1 control's GPU type).
#SBATCH --job-name=if_smA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:27:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/smA/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/smA/logs/%j.err
set -euo pipefail
export JOB=smA ARM=fid MODE=smoke WALL_S=1620
cd /cluster/tufts/paralab/tawal01/imp-fronts/smA
source ./train_body.sh
