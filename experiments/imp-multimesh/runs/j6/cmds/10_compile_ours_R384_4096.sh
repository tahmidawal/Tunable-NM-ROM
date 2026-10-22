set -uo pipefail
$PY mmcompile.py ours 4096 384 gauss128 CNR $OUT/compile/ours_4096_R384.json
