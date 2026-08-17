#!/usr/bin/env bash
# make_cells.sh [panels|consolidate] [staging_root]
#
# Builds cluster/stage/<cell>/{code,ckpt,out,logs,run.sbatch} for
# /cluster/tufts/paralab/tawal01/ctol/<cell>/.  ONE CELL = ONE JOB DIR = ONE JOB.
#
#   panels       (default)  one job per (PDE, mesh) PANEL, all submitted at once:
#                  ctol_p_n{32,64,128,256,512}   Poisson
#                  ctol_b_n{32,64,128,256}       Burgers
#                Each panel runs its ENTIRE k x method x tau grid sequentially in one
#                process on one GPU, so every timing inside a panel is comparable and
#                the per-(PDE, N) Pareto frontier -- which is computed WITHIN a panel --
#                is valid.  All panels request the SAME GPU TYPE (a100).
#
#   consolidate  ONE job, ONE GPU, all meshes: re-times only the per-(method, N) argmin
#                configurations named in cluster/stage/consolidate_configs.json, which
#                `ctol_pick_configs.py` writes from the pulled panel results.  This is
#                the ONLY timing source the cross-N scaling figure may use, because
#                cross-N ratios measured on different GPUs are not comparable.
set -euo pipefail
MODE="${1:-panels}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL_DIR="$(dirname "$HERE")"
EXP="$(dirname "$CELL_DIR")"
WT="$(cd "$EXP/.." && pwd)"
WTS="$(cd "$WT/.." && pwd)"
ROOT="${2:-$HERE/stage}"
REMOTE=/cluster/tufts/paralab/tawal01/ctol
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
COMMIT="$(git -C "$WT" rev-parse --short HEAD)"
DIRTY="$(git -C "$CELL_DIR" status --porcelain -- . | sha256sum | cut -c1-12)"

PSRC="$WTS/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective"
BSRC="$WTS/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping"
MSP="$WTS/2026-08-14-multistage-precision/experiments/multistage-precision"
B2D="$WTS/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom"

# The executed bundle is assembled from FOUR worktrees, so this tree's commit alone
# does not identify it.  Record every source worktree's commit and dirty state; the
# drivers additionally sha256 every module and checkpoint they actually load into
# the result JSON, and the staged MANIFEST.sha256 is pulled back with the results.
src_commit() { git -C "$1" rev-parse --short HEAD 2>/dev/null || echo unknown; }
src_dirty()  { git -C "$1" status --porcelain -- . 2>/dev/null | sha256sum | cut -c1-8; }
SRC_COMMITS="ctol=$COMMIT/$DIRTY"
for d in "$PSRC" "$BSRC" "$MSP" "$B2D"; do
  SRC_COMMITS="$SRC_COMMITS,$(basename "$(dirname "$(dirname "$d")")")=$(src_commit "$d")/$(src_dirty "$d")"
done
echo "source bundle: $SRC_COMMITS"

mk() {  # mk <cell> <hours> <mem> <envs> <cmd>
  local cell="$1" hours="$2" mem="$3" envs="$4" cmd="$5"
  local d="$ROOT/$cell"
  rm -rf "$d"; mkdir -p "$d/code" "$d/ckpt" "$d/out" "$d/logs"
  cp "$CELL_DIR"/ctol_*.py "$d/code/"
  cat > "$d/run.sbatch" <<EOS
#!/bin/bash
#SBATCH -J $cell
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=$mem
#SBATCH -t $hours:00:00
#SBATCH -o $REMOTE/$cell/logs/%j.out
#SBATCH -e $REMOTE/$cell/logs/%j.err
set -euo pipefail
cd "$REMOTE/$cell/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=$PY
export CTOL_COMMIT=$COMMIT
export CTOL_SRC_COMMITS="$SRC_COMMITS"
echo "host=\$(hostname)  node=\$SLURMD_NODENAME  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$COMMIT  dirty=$DIRTY (dirty marker 'e3b0c442' == clean)  cell=$cell  job=\$SLURM_JOB_ID"
echo "src_bundle=$SRC_COMMITS"
cp "$REMOTE/$cell/MANIFEST.sha256" "$REMOTE/$cell/out/MANIFEST.sha256" 2>/dev/null || true
\$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export $envs
$cmd
rc=\$?
[[ \$rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=\$rc"; exit \$rc; }
EOS
  echo "$d"
}

stage_poisson() {  # stage_poisson <dir>
  local d="$1"
  mkdir -p "$d/code/deps/poisson2d-rom-objective/followup" \
           "$d/code/deps/poisson2d-rom-objective/deps"
  cp "$PSRC"/pro_common.py "$d/code/deps/poisson2d-rom-objective/"
  cp "$PSRC"/followup/fu_eq.py "$PSRC"/followup/fu_style.py \
     "$d/code/deps/poisson2d-rom-objective/followup/"
  cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py \
     "$d/code/deps/poisson2d-rom-objective/deps/"
  for K in 2 4 6 8 12 16 24 32; do
    cp "$PSRC/runs/followup/pk_K$K/autodec_K${K}_N64_hbc_stages.pkl" "$d/ckpt/"
  done
}

stage_burgers() {  # stage_burgers <dir>
  local d="$1"
  mkdir -p "$d/code/deps/burgers2d-rom-latent-stepping/followup" \
           "$d/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom" \
           "$d/code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision"
  cp "$BSRC"/blat_common.py "$d/code/deps/burgers2d-rom-latent-stepping/"
  cp "$BSRC"/followup/fu_common.py "$BSRC"/followup/fu_style.py \
     "$d/code/deps/burgers2d-rom-latent-stepping/followup/"
  cp "$B2D"/burgers2d_film.py \
     "$d/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/"
  cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py \
     "$d/code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision/"
  # EXPLICIT checkpoint paths (a `find | head -1` would silently pick a different
  # checkpoint if the source tree ever grew a second file of the same name)
  for spec in "2:runs/followup/bk_K2" "4:runs/ad_n64_k4" "6:runs/followup/bk_K6" \
              "8:runs/ad_n64_k8" "12:runs/followup/bk_K12" "16:runs/ad_n64_k16" \
              "24:runs/followup/bk_K24" "32:runs/followup/bk_K32"; do
    K="${spec%%:*}"; sub="${spec#*:}"
    src="$BSRC/$sub/blat_ad_N64_K${K}.pkl"
    [[ -f "$src" ]] || { echo "missing burgers checkpoint K=$K at $src" >&2; exit 1; }
    cp "$src" "$d/ckpt/"
  done
}

seal() { (cd "$1" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256); echo "built $1"; }

PGRID="KS=2,4,6,8,12,16,24,32 TAUS=1e-1,1e-2,1e-3 M=64 MQ=256 M_BIG=256 K_BIG=32 MQ_4M=1024 DO_SUPP=1 POOL_CONTROL=1 N_TEST=16 GN_ITERS=60 TIME_REPS=7 TIME_WARM=2 PKL_DIR=../ckpt"
BGRID="N=64 KS=2,4,6,8,12,16,24,32 TAUS=1e-1,1e-2,1e-3 M=64 MQ=256 M_BIG=256 K_BIG=32 MQ_4M=1024 DO_SUPP=1 CTOL_N_TEST=16 N_POD_TRAJ=512 POD_SLICE_STRIDE=4 TIME_REPS=7 TIME_WARM=2 GEN_CHUNK=16 PKL_DIR=../ckpt"

if [[ "$MODE" == "panels" ]]; then
  for NN in 32 64 128 256 512; do
    mem=64G; hrs=16
    [[ $NN -ge 256 ]] && { mem=192G; hrs=24; }
    d=$(mk "ctol_p_n$NN" "$hrs" "$mem" "$PGRID NS=$NN" \
        "\$PY -u ctol_poisson.py ../out/ctol_poisson_n$NN.json")
    stage_poisson "$d"; seal "$d"
  done
  for NN in 32 64 128 256; do
    mem=96G; hrs=24
    [[ $NN -ge 256 ]] && { mem=240G; hrs=24; }
    d=$(mk "ctol_b_n$NN" "$hrs" "$mem" "$BGRID NS=$NN" \
        "\$PY -u ctol_burgers.py ../out/ctol_burgers_n$NN.json")
    stage_burgers "$d"; seal "$d"
  done
elif [[ "$MODE" == "fom" ]]; then
  # FOM-ONLY cell: the iso-accuracy ladder at EVERY mesh, sequentially in ONE process
  # on ONE GPU.  Two reasons this is its own job rather than part of the panels:
  #   * the denominator of the headline claim then has cross-N comparable timings,
  #     which the fanned-out panels cannot give;
  #   * it refines the ladder without disturbing ROM panels that are already running.
  d=$(mk ctol_fom_p 4 96G \
      "$PGRID NS=32,64,128,256,512 FOM_ONLY=1 DO_SUPP=0 POOL_CONTROL=0 CAP_CONTROL=0 DO_CEILING=0" \
      "\$PY -u ctol_poisson.py ../out/ctol_poisson_fomladder.json")
  stage_poisson "$d"; seal "$d"
elif [[ "$MODE" == "consolidate" ]]; then
  CFG="$HERE/stage/consolidate_configs.json"
  [[ -f "$CFG" ]] || { echo "missing $CFG -- run ctol_pick_configs.py first" >&2; exit 1; }
  d=$(mk ctol_consol_p 8 192G \
      "$PGRID NS=32,64,128,256,512 DO_SUPP=0 POOL_CONTROL=0 DO_POD_DIRECT=0 CONFIGS=../consolidate_configs.json ARM_TAG=consolidated" \
      "\$PY -u ctol_poisson.py ../out/ctol_poisson_consolidated.json")
  stage_poisson "$d"; cp "$CFG" "$d/consolidate_configs.json"; seal "$d"
  d=$(mk ctol_consol_b 24 240G \
      "$BGRID NS=32,64,128,256 DO_SUPP=0 CONFIGS=../consolidate_configs.json ARM_TAG=consolidated" \
      "\$PY -u ctol_burgers.py ../out/ctol_burgers_consolidated.json")
  stage_burgers "$d"; cp "$CFG" "$d/consolidate_configs.json"; seal "$d"
else
  echo "usage: make_cells.sh [panels|consolidate] [staging_root]" >&2; exit 1
fi

echo "cells built under $ROOT (mode $MODE, commit $COMMIT, dirty $DIRTY)"
