set -uo pipefail
env JAX_PLATFORMS=cpu $PY oursq.py $OUT/ours1024 1024 $OUT/ours1024/oursq.json
