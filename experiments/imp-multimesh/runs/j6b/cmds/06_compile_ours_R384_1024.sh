set -uo pipefail
$PY mmcompile.py ours 1024 384 gauss128 CNR $OUT/compile/ours_1024_R384.json
