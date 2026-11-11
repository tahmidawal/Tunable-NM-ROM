# Tune + final-train + development-score one arm.  Usage: tasks/train_arm.sh <node|grid|don> [extra ops_train args]
set -euo pipefail
ARM=$1; shift
"$PY" ops_train.py --arm "$ARM" --data "$NS/prep01/results/data" --out "results/$ARM" "$@"
