#!/usr/bin/env bash
# Follow-up cells for the Poisson-2D study (namespace pobj2/).
#   ./fu_cells.sh <staging_root>  -> <staging_root>/<cell>/{code,ckpt,out,logs,run.sbatch}
set -euo pipefail
ROOT="${1:?staging root}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # experiments/poisson2d-rom-objective
MSP="$HERE/../../../2026-08-14-multistage-precision/experiments/multistage-precision"
REMOTE=/cluster/tufts/paralab/tawal01/pobj2
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
COMMIT="$(git -C "$HERE" rev-parse --short HEAD)"
HBC8="$HERE/runs/hbc_K8/autodec_K8_N64_hbc_stages.pkl"

mk() {  # mk <cell> <time> <mem> <body...>
  local cell="$1" tlim="$2" mem="$3"; shift 3
  local d="$ROOT/$cell"
  rm -rf "$d"; mkdir -p "$d/code/deps" "$d/code/followup" "$d/ckpt" "$d/out" "$d/logs"
  cp "$HERE"/pro_common.py "$HERE"/pro_objective.py "$HERE"/pro_colloc.py "$HERE"/pro_train.py "$d/code/"
  cp "$HERE"/followup/fu_*.py "$d/code/followup/"
  cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py "$d/code/deps/"
  cp "$HBC8" "$d/ckpt/"
  cat > "$d/run.sbatch" <<EOS
#!/bin/bash
#SBATCH -J pobj2_$cell
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=$mem
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
EOS
  (cd "$d" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
  echo "$d"
}

ROM="NS=1 GN_ITERS=60 OBJECTIVES=weak_a1_M64 MS=256 SCHEMES=full,nnls,nnlsoff INITS=nearest,mean EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072"
FDC="NS=1 GN_ITERS=60 OBJECTIVES=fd MS=256 SCHEMES=full INITS=nearest"
# k-ladder: hard-BC K ladder, seed 0 (K=8 = reproduction of hbc_K8)
for K in 2 4 6 8 12 16 24 32; do
  mk pk_K$K 08:00:00 64G "K_LAT=$K HARD_BC=1 \$PY followup/fu_train.py ../out && PKL=../out/autodec_K${K}_N64_hbc_stages.pkl $ROM \$PY pro_colloc.py ../out/rom_K$K.json && PKL=../out/autodec_K${K}_N64_hbc_stages.pkl $FDC \$PY pro_colloc.py ../out/fd_K$K.json"
done
# multi-seed: K=8 hard-BC, training seeds 1,2 (seed 0 = pk_K8 / hbc_K8)
for S in 1 2; do
  mk ps_S$S 08:00:00 64G "K_LAT=8 HARD_BC=1 TRAIN_SEED=$S \$PY followup/fu_train.py ../out && PKL=../out/autodec_K8_N64_hbc_S${S}_stages.pkl $ROM \$PY pro_colloc.py ../out/rom_S$S.json && PKL=../out/autodec_K8_N64_hbc_S${S}_stages.pkl $FDC \$PY pro_colloc.py ../out/fd_S$S.json"
done
# m ladder (M=64) and M ladder (m=4M) + full grid, on the existing hard-BC K=8 decoder
mk pm_K8 10:00:00 64G "PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=weak_a1_M64 MS=64,128,256,512,1024 SCHEMES=full,nnls,nnlsoff INITS=nearest EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 \$PY pro_colloc.py ../out/mlad_K8.json && PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=weak_a1_M16 MS=64 SCHEMES=full,nnls,nnlsoff INITS=nearest EQ_ROWS=3072 \$PY pro_colloc.py ../out/Mlad_M16.json && PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=weak_a1_M32 MS=128 SCHEMES=full,nnls,nnlsoff INITS=nearest EQ_ROWS=3072 \$PY pro_colloc.py ../out/Mlad_M32.json && PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=weak_a1_M128 MS=512 SCHEMES=full,nnls,nnlsoff INITS=nearest EQ_ROWS=3072 \$PY pro_colloc.py ../out/Mlad_M128.json && PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=weak_a1_M256 MS=1024 SCHEMES=full,nnls,nnlsoff INITS=nearest EQ_ROWS=3072 \$PY pro_colloc.py ../out/Mlad_M256.json"
# POD ladder (linear control) -- CPU/GPU light
mk pp_pod 02:00:00 32G "KS=2,4,6,8,12,16,24,32,48,64 MS=64,256 \$PY followup/fu_pod.py ../out/pod_ladder.json"
# timing vs N on ONE GPU (hard-BC K=8 decoder trained at N=64, meshfree EQ M=64 m=256)
mk pt_n 06:00:00 96G "PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl NS=32,64,128,256,512 M=64 MQ=256 GN_ITERS=60 TIME_REPS=7 \$PY followup/fu_timing.py ../out/timing_n.json"
