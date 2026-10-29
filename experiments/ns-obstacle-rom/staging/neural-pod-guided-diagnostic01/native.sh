#!/usr/bin/env bash
#SBATCH --job-name=ctol_nsobsrom_teacherdiag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
exec /usr/bin/timeout --signal=TERM --kill-after=10s 885s /usr/bin/bash ./native-body.sh
