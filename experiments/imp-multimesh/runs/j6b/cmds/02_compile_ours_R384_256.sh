set -uo pipefail
$PY mmcompile.py ours 256 384 gauss128 CNR $OUT/compile/ours_256_R384.json
