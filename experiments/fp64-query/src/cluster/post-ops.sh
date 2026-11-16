#!/bin/bash
# Run after the ROM driver, in the same job on the same GPU: re-time and evaluate the frozen FNO/U-Net checkpoints.
#   post-ops.sh test64            (reference fields: rebuttal-burgers-ops/data/test, read only)
#   post-ops.sh <name> <seed> [count=64]   (reference fields generated here by the same solver)
set -euo pipefail
C="$1"; O=experiments/b2d-query-speed/ops
export PYTHONPATH="$PWD/$O" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$PY" $O/pick_manifest.py $O/selection_manifest.json output/ops_manifest.json
if [ "$C" = test64 ]; then
  DIR=/cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/data/test
else
  DIR="$PWD/cohort"; "$PY" $O/gen_cohort.py --seed "$2" --count "${3:-64}" --name "$C" --out "$DIR"
  cp "$DIR/$C-L256.json" output/
fi
"$PY" $O/evaluate.py --manifest output/ops_manifest.json --cohort-dir "$DIR" --cohort "$C" --out-dir "output/ops-$C" --skip-1024
rm -rf "$PWD/cohort"
