#!/bin/bash
# Stage + submit the three validation panels and three certification jobs for model R (one dir per job).
set -euo pipefail
R="$1"; TAG="$2"   # e.g. 512 v1
cd "$(dirname "$0")/.."
for n in 33 65 129; do
  for kind in val cert; do
    A="${kind}${n}${TAG}"
    GPU=a100-80G
    /home/tahmid/Dev/.venv/bin/python cluster/stage.py "$A" --script panel.py \
      --args "--config experiments/burgers3d-span/configs/${kind}_R${R}_n${n}.json --model experiments/burgers3d-span/inputs/model_R${R} --out output" \
      --gpu $GPU --mem 200G --hours 8 \
      --extra panel.py configs/${kind}_R${R}_n${n}.json inputs/model_R${R}/bank.pkl inputs/model_R${R}/head_K32.pkl inputs/model_R${R}/head_K64.pkl | tail -1
    ./cluster/submit.sh "$A" 2>&1 | grep -E "Submitted|refus"
  done
done
ssh tufts-login 'squeue -u $USER -o "%.10i %.20j %.8T %R" | grep b3s'
