#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?batch allocation required}"
[ "$(pwd -P)" = /cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/scanned-composite-fixture-cluster01 ] || exit 64
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_CACHE_MAXSIZE=16777216 JAX_ENABLE_COMPILATION_CACHE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/cache" CUDA_CACHE_PATH="$PWD/cache/cuda"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" results
sha256sum -c SOURCE-SHA256SUMS
"$PY" -c 'import jax,json,sys; b=jax.default_backend(); print("jax_backend="+b); r={"backend":b,"x64":jax.config.x64_enabled,"matmul_precision":str(jax.config.jax_default_matmul_precision),"devices":[d.device_kind for d in jax.devices()]};open("results/runtime.json","w").write(json.dumps(r));sys.exit(0 if b=="gpu" and r["x64"] and r["matmul_precision"]=="highest" else 42)'
exec "$PY" -m accuracy_v2.residual_composite_fixture --model model.npz --model-sha256 645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60 --points-fixture points.npz --points-sha256 d29ac3b9181ec6bbffee296328b135bb0ec687edbae4dd89589a73f252403e46 --output results/fixture
