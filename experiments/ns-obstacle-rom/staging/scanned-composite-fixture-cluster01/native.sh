#!/bin/bash
#SBATCH --job-name=ctol_nsobsrom_scanfixture
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
exec /usr/bin/timeout --signal=TERM --kill-after=5s 105s /bin/bash ./native-body.sh
