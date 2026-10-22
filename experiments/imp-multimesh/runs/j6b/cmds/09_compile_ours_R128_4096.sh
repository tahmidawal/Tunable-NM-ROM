set -uo pipefail
mkdir -p $OUT/compile && $PY mmcompile.py ours 4096 128 fib4181 BE $OUT/compile/ours_4096_R128.json
