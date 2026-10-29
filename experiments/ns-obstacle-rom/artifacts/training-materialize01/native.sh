#!/usr/bin/env bash
#SBATCH --job-name=ctol_nsobsrom_materialize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:07:30
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
exec /usr/bin/timeout --signal=TERM --kill-after=10s 435s /usr/bin/bash ./native-body.sh
