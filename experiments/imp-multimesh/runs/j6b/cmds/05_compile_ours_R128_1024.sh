set -uo pipefail
mkdir -p $OUT/compile && $PY mmcompile.py ours 1024 128 fib4181 BE $OUT/compile/ours_1024_R128.json
