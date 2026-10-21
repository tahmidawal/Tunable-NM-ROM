set -uo pipefail
env JAX_PLATFORMS=cpu $PY oursq.py $OUT/ours4096 4096 $OUT/ours4096/oursq.json
