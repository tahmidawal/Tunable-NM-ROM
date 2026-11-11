# Final evaluation of frozen arms: dev Re 95 + sealed held-out Re 105, timing on one GPU.
# Usage: tasks/final_eval.sh [--no-heldout] label=<attempt>/results/<arm> [...]   (paths relative to the lane namespace)
# --no-heldout is the pipeline dry run (development only; the sealed directory is not read).
set -euo pipefail
ARGS=()
for s in "$@"; do
  if [ "$s" = "--no-heldout" ]; then ARGS+=(--no-heldout); else ARGS+=(--arm "${s%%=*}=$NS/${s#*=}"); fi
done
"$PY" final_eval.py --data "$NS/prep01/results/data" --sealed "$NS/prep01/results/sealed" --out results/final "${ARGS[@]}"
