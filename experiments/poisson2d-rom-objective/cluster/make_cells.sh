#!/usr/bin/env bash
# Build the Tufts cell directories for the Poisson-2D ROM-objective study.
#   ./make_cells.sh <staging_root>      -> <staging_root>/<cell>/{code,ckpt,out,logs,run.sbatch}
# Then: rsync/scp <staging_root>/<cell> to /cluster/tufts/paralab/tawal01/pobj/<cell>
# and sbatch run.sbatch from inside it (one job per cell dir).
set -euo pipefail
ROOT="${1:?staging root}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MSP="$HERE/../../../2026-08-14-multistage-precision/experiments/multistage-precision"
REMOTE=/cluster/tufts/paralab/tawal01/pobj
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
COMMIT="$(git -C "$HERE" rev-parse --short HEAD)"

mk() {  # mk <cell> <time> <body...>
  local cell="$1" tlim="$2"; shift 2
  local d="$ROOT/$cell"
  rm -rf "$d"; mkdir -p "$d/code/deps" "$d/ckpt" "$d/out" "$d/logs"
  cp "$HERE"/pro_common.py "$HERE"/pro_objective.py "$HERE"/pro_colloc.py "$HERE"/pro_train.py "$d/code/"
  cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py "$d/code/deps/"
  cp "$MSP"/runs/autodec/ad_K8/ms_autodecoder_K8_stages.pkl "$d/ckpt/"
  cp "$MSP"/runs/autodec/ad_K4/ms_autodecoder_K4_stages.pkl "$d/ckpt/"
  cat > "$d/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J pobj_$cell
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t $tlim
#SBATCH -o $REMOTE/$cell/logs/%j.out
#SBATCH -e $REMOTE/$cell/logs/%j.err
set -euo pipefail
cd "$REMOTE/$cell"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=$PY
echo "host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit=$COMMIT  cell=$cell"
\$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
cd code
$*
echo "ALL-DONE"
EOF
  echo "$d"
}

K8=../ckpt/ms_autodecoder_K8_stages.pkl
K4=../ckpt/ms_autodecoder_K4_stages.pkl
COMPACT="fd,spec_a0_M64,spec_a0_M256,spec_a0.5_M256,spec_a0.5_Mall,spec_a1_M64,spec_a1_M256,spec_a1_Mall,ritz,cg5,cg20,lowpass2,lowpass4,lowpass8"

# cell 1: full objective sweep, K=8, stage-0 decoder, 3 inits
mk obj_K8_S1 06:00:00 "PKL=$K8 NS=1 GN_ITERS=60 \$PY pro_objective.py ../out/obj_K8_S1.json"
# cell 1b: same on the 3-stage decoder (comparability with the multistage S=3 rows)
mk obj_K8_S3 06:00:00 "PKL=$K8 NS=3 GN_ITERS=60 OBJECTIVES=$COMPACT \$PY pro_objective.py ../out/obj_K8_S3.json"
# cell 1c: K=4 decoder
mk obj_K4_S1 06:00:00 "PKL=$K4 NS=1 GN_ITERS=60 \$PY pro_objective.py ../out/obj_K4_S1.json"
# cell 1d: 5x budget on K=8 (solver- vs objective-floor check for the new objectives)
mk obj_K8_S1_b300 08:00:00 "PKL=$K8 NS=1 GN_ITERS=300 INITS=nearest,mean OBJECTIVES=$COMPACT \$PY pro_objective.py ../out/obj_K8_S1_b300.json"
# cell 2: collocation study, K=8
mk colloc_K8 12:00:00 "PKL=$K8 NS=1 GN_ITERS=60 OBJECTIVES=fd,spec_a0_M64,spec_a1_M64,spec_a1_M256,spec_a0.5_M256 MS=128,256,512,1024 SCHEMES=uniform,biased,nnls,offgrid INITS=nearest,mean \$PY pro_colloc.py ../out/colloc_K8.json"
# cell 4: hard-BC retrain (K=8, N=64) then the objective sweep on it
mk hbc_K8 10:00:00 "K_LAT=8 HARD_BC=1 \$PY pro_train.py ../out && PKL=../out/autodec_K8_N64_hbc_stages.pkl NS=1 GN_ITERS=60 \$PY pro_objective.py ../out/obj_hbc_K8.json && PKL=../out/autodec_K8_N64_hbc_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=fd,spec_a1_M64,spec_a1_M256 MS=128,256,512,1024 SCHEMES=uniform,offgrid INITS=nearest \$PY pro_colloc.py ../out/colloc_hbc_K8.json"
# cell 5: N ladder — retrain plain K=8 stage-0 at N=32/64/128 with THIS trainer, then the compact sweep
for NN in 32 64 128; do
  mk nlad_N$NN 08:00:00 "N=$NN K_LAT=8 HARD_BC=0 \$PY pro_train.py ../out && N=$NN PKL=../out/autodec_K8_N${NN}_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=$COMPACT \$PY pro_objective.py ../out/obj_nlad_N$NN.json"
done
