set -uo pipefail
mkdir -p $OUT/compile && $PY mmcompile.py ours 256 128 fib4181 BE $OUT/compile/ours_256_R128.json
