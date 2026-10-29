#!/usr/bin/env bash
#SBATCH --job-name=ctol_nsobsrom_podrecover
#SBATCH --partition=gpu
#SBATCH --nodelist=pax023
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
exec /usr/bin/timeout --signal=TERM --kill-after=10s 1185s /usr/bin/bash ./native-body.sh
