#!/usr/bin/env bash
set -euo pipefail
cfg=$1
base=$2
output=$3
src=$(cd "$(dirname "$0")" && pwd)
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$PY" "$src/witness_audit.py" --base "$base" --config "$cfg" --output "$output" --receipt "$output/witness-audit.json"
