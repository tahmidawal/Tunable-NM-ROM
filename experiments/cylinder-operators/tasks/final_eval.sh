# Final evaluation of frozen arms: dev Re 95 + sealed held-out Re 105, timing on one GPU.
# Usage: tasks/final_eval.sh label=<attempt>/results/<arm> [...]   (paths relative to the lane namespace)
set -euo pipefail
ARGS=()
for s in "$@"; do ARGS+=(--arm "${s%%=*}=$NS/${s#*=}"); done
"$PY" final_eval.py --data "$NS/prep01/results/data" --sealed "$NS/prep01/results/sealed" --out results/final "${ARGS[@]}"
