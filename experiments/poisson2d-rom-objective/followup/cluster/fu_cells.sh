#!/usr/bin/env bash
# Follow-up cells for the Poisson-2D ROM-objective study (cluster namespace pobj2/).
#   ./fu_cells.sh            -> builds followup/cluster/stage/<cell>/{code,ckpt,out,logs,run.sbatch}
# One cell = one job dir = one job.  Then: launch.sh <cell>.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/../.." && pwd)"                                # experiments/poisson2d-rom-objective
MSP="$EXP/../../../2026-08-14-multistage-precision/experiments/multistage-precision"
WAVE="${1:-wave1}"                                  # wave1 | wave2 (wave2 needs the k cells pulled)
ROOT="${2:-$HERE/stage}"
REMOTE=/cluster/tufts/paralab/tawal01/pobj2
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
COMMIT="$(git -C "$EXP" rev-parse --short HEAD)"
DIRTY="$(git -C "$EXP" status --porcelain -- . | sha256sum | cut -c1-12)"   # e3b0c442 == clean
HBC8="$EXP/runs/hbc_K8/autodec_K8_N64_hbc_stages.pkl"
[[ -f "$HBC8" ]] || { echo "missing $HBC8" >&2; exit 1; }

mk() {  # mk <cell> <time> <mem> <body...>
  local cell="$1" tlim="$2" mem="$3"; shift 3
  local d="$ROOT/$cell"
  rm -rf "$d"; mkdir -p "$d/code/deps" "$d/code/followup" "$d/ckpt" "$d/out" "$d/logs"
  cp "$EXP"/pro_common.py "$EXP"/pro_objective.py "$EXP"/pro_colloc.py "$EXP"/pro_train.py "$d/code/"
  cp "$EXP"/followup/fu_*.py "$d/code/followup/"
  cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py "$d/code/deps/"
  cp "$HBC8" "$d/ckpt/"
  cat > "$d/run.sbatch" <<EOS
#!/bin/bash
#SBATCH -J $cell
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
echo "host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit=$COMMIT  dirty=$DIRTY  cell=$cell"
\$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
cd code
$*
rc=\$?
[[ \$rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=\$rc"; exit \$rc; }
EOS
  (cd "$d" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
  echo "$d"
}

# the headline recipe: weak-form Galerkin, Lambda^-1 weighting, 64 test modes, NNLS-EQ
# quadrature at m=256 (~4M) on grid nodes and on the meshfree pool, plus the full grid
ROM="NS=1 GN_ITERS=60 OBJECTIVES=weak_a1_M64 MS=256 SCHEMES=full,nnls,nnlsoff INITS=nearest,mean EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_FIXED_SNAPS=1"
FDC="NS=1 GN_ITERS=60 OBJECTIVES=fd MS=256 SCHEMES=full INITS=nearest"

if [[ "$WAVE" == "wave2" ]]; then
  # ---- timing vs k on ONE GPU, all k sequential in ONE process (+ the linear POD control,
  # whose online solve is one precomputed pseudo-inverse matvec).  Needs the wave-1
  # checkpoints pulled into runs/followup/pk_K<k>/.
  PK=""
  for K in 2 4 6 8 12 16 24 32; do
    src="$EXP/runs/followup/pk_K$K/autodec_K${K}_N64_hbc_stages.pkl"
    [[ -f "$src" ]] || { echo "missing $src (pull wave 1 first)" >&2; exit 1; }
    PK="$PK,../ckpt/autodec_K${K}_N64_hbc_stages.pkl"
  done
  d=$(mk pt_k 06:00:00 64G "MODE=k PKLS=${PK#,} PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl M=64 MQ=256 GN_ITERS=60 TIME_REPS=7 POD_KS=2,4,6,8,12,16,24,32,48,64 \$PY -u followup/fu_timing.py ../out/timing_k.json")
  for K in 2 4 6 8 12 16 24 32; do cp "$EXP/runs/followup/pk_K$K/autodec_K${K}_N64_hbc_stages.pkl" "$d/ckpt/"; done
  (cd "$d" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
  echo "cells built under $ROOT"
  exit 0
fi

# ---- k ladder: hard-BC auto-decoder retrained per K at equal budget, seed 0.
# (K=8 reproduces runs/hbc_K8 with the identical pipeline -> it is the ladder's seed-0 point.)
for K in 2 4 6 8 12 16 24 32; do
  mk pk_K$K 08:00:00 64G "K_LAT=$K HARD_BC=1 \$PY -u followup/fu_train.py ../out && PKL=../out/autodec_K${K}_N64_hbc_stages.pkl $ROM \$PY -u pro_colloc.py ../out/rom_K$K.json && PKL=../out/autodec_K${K}_N64_hbc_stages.pkl $FDC \$PY -u pro_colloc.py ../out/fd_K$K.json"
done
# ---- multi-seed: K=8 hard-BC, TRAIN_SEED 1 and 2 (seed 0 = pk_K8).  TRAIN_SEED changes
# the net init / latent init / batch order only; the data draw and the test split are fixed.
for S in 1 2; do
  mk ps_S$S 08:00:00 64G "K_LAT=8 HARD_BC=1 TRAIN_SEED=$S \$PY -u followup/fu_train.py ../out && PKL=../out/autodec_K8_N64_hbc_S${S}_stages.pkl $ROM \$PY -u pro_colloc.py ../out/rom_S$S.json && PKL=../out/autodec_K8_N64_hbc_S${S}_stages.pkl $FDC \$PY -u pro_colloc.py ../out/fd_S$S.json"
done
# ---- m ladder at fixed (K=8, M=64): grid-node EQ vs meshfree-pool EQ vs the full grid
mk pm_m 10:00:00 64G "PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=weak_a1_M64 MS=64,128,256,512,1024 SCHEMES=full,nnls,nnlsoff INITS=nearest,mean EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_FIXED_SNAPS=1 \$PY -u pro_colloc.py ../out/mlad_K8.json"
# ---- M ladder at m ~ 4M (one invocation per M: the EQ weights must be refit for each M)
for MM in 16 32 64 128 256; do
  m=$((4 * MM))
  mk pM_M$MM 06:00:00 64G "PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=weak_a1_M$MM MS=$m SCHEMES=full,nnls,nnlsoff INITS=nearest EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_FIXED_SNAPS=1 \$PY -u pro_colloc.py ../out/Mlad_M$MM.json"
done
# ---- POD (linear subspace) k ladder: the same TRAIN snapshots, the same objectives
mk pp_pod 02:00:00 32G "KS=2,4,6,8,12,16,24,32,48,64 MS=64,256 \$PY -u followup/fu_pod.py ../out/pod_ladder.json"
# ---- timing vs N on ONE GPU, all N sequential in ONE process (EQ refit per N)
mk pt_n 06:00:00 96G "MODE=n PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl NS=32,64,128,256,512 M=64 MQ=256 GN_ITERS=60 TIME_REPS=7 \$PY -u followup/fu_timing.py ../out/timing_n.json"
# ---- per-solve / per-iteration cost across the m and M ladders on ONE GPU (accuracy for
# those ladders is measured by pro_colloc in pm_m / pM_M*; this cell is the cost column)
mk pt_m 06:00:00 64G "MODE=m PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl M=64 MQ=256 MS_LADDER=64,128,256,512,1024 M_LADDER=16,32,64,128,256 POOLS=offgrid,grid GN_ITERS=60 TIME_REPS=7 \$PY -u followup/fu_timing.py ../out/timing_m.json"
echo "cells built under $ROOT"
