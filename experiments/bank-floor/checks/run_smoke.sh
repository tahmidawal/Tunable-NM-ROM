#!/bin/bash
# usage: run_smoke.sh poisson|burgers OUTDIR   (local GB10, tiny)
set -euo pipefail
E=$(cd "$(dirname "$0")/../.." && pwd)
source /etc/profile.d/jax-mem.sh
export PYTHONPATH=$E/mr-burgers2d:$E/head-ablation:$E/separable-decoder:$E/b-head-train:$E/multiresolution-poisson:$E/cost-to-tolerance:$E/wave2d-rom-latent-stepping/deps/multistage-precision:$E/bank-floor
export JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONUNBUFFERED=1
if [ "$1" = poisson ]; then INC=$E/bank-floor/incumbents/poisson_primary_K32.pkl; else INC=$E/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl; fi
jaxrun /home/tahmid/Dev/.venv/bin/python $E/bank-floor/bf_rep.py --config $E/bank-floor/checks/config-smoke-$1.json --incumbent $INC --out $2
