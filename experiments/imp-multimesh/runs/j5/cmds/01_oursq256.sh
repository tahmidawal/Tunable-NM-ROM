set -uo pipefail
env JAX_PLATFORMS=cpu $PY oursq.py $OUT/ours256 256 $OUT/ours256/oursq.json
