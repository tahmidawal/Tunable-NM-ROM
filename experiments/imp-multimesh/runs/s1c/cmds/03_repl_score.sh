set -uo pipefail
env JAX_PLATFORMS=cpu $PY oursq.py $OUT/repl 1024 $OUT/repl/oursq.json
