#!/usr/bin/env bash
# cells.sh [stage|launch] -- define + stage (+ launch) every cell of the study.
# ONE job per directory; each cell regenerates its data from the seed on the cluster.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WT="$(cd "$EXP/../../.." && pwd)"
ACT="${1:-stage}"
SWEEP="$WT/2026-08-14-wave2d-coord-rom/experiments/wave2d-coord-rom/sweep"

# ---- common knobs -----------------------------------------------------------
# RS=20 (dt=1e-3, 1000 latent steps): the u-only Newmark FOM at that dt is 1.16e-3
# from the 80-substep FOM at N=64 (wlat_verify V5), i.e. ~10x below the expected
# manifold floor.  Every ROM error is reported against BOTH FOMs.
RS=20
BASE="N_TEST=16 GN_BUDGET=30 IC_BUDGET=100 FLOOR_BUDGET=60 EQ_SNAPS=64 EQ_POOL=4096"
# the recipe (weak Galerkin + NNLS-EQ at m ~ 4M), the M sweep, and the controls
# ordered CHEAP FIRST: wlat_rom flushes its JSON after every variant, so a cell
# that runs out of wall clock still yields the recipe arms
VAR="lspg:eq256:weak64,galerkin:eq256:weak64,lspg:eq576:weak144,lspg:eq1024:weak256"
VAR="$VAR,galerkin:eq1024:weak256,lspg:eqoff256:weak64,lspg:full:weak64,galerkin:full:weak64"
VAR="$VAR,lspg:full:weaku64,lspg:full:weakl64,lspg:full:weak144,lspg:full:weak256"
VAR="$VAR,lspg:full:fd,galerkin:full:fd,lspg:rand512:fd,lspg:offgrid512:fd"
PODV="lspg:full:fd,lspg:full:weak256,lspg:eq1024:weak256"
# the RS (ROM time-step) convergence arm: cheap variants only
RSVAR="lspg:eq256:weak64,galerkin:eq256:weak64,lspg:full:weak64"

mk () { "$HERE/make_cell.sh" "$@" >/dev/null; echo "staged $1"; }

# ---- Stage 1: the (z,t) sweep decoder, space-time LSPG -----------------------
for icw in 1 sqrt50; do
  cell="ws1_n64_icw$icw"
  w=1.0; [[ $icw == sqrt50 ]] && w=7.0710678118654755
  mk "$cell" 64G 4 "N=64 $BASE S1_RS=10 S1_BUDGET=60 IC_W=$w" \
     '$PY -u wlat_stage1.py in/wave2d_film_N64.pkl ../out'
  cp "$SWEEP/wave2d_film_N64.pkl" "$SWEEP/wave2d_results_N64.json" "$HERE/stage/$cell/code/in/"
  (cd "$HERE/stage/$cell" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
done

# ---- Stage 2: auto-decoder K in {4,8,16} at N=64, K=8 at N=128 ---------------
for k in 4 8 16; do
  cell="wad_n64_k$k"
  extra=""
  # the K=8 cell additionally sweeps the ROM time step
  [[ $k == 8 ]] && extra=" && ROM_SUBSTEPS=8 VARIANTS=$RSVAR POD_KS=8 POD_VARIANTS=lspg:full:fd DO_TIMING=1 \$PY -u wlat_rom.py ../out/wlat_ad_N64_K8.pkl ../out rs8 && ROM_SUBSTEPS=40 VARIANTS=$RSVAR POD_KS=8 POD_VARIANTS=lspg:full:fd DO_TIMING=1 \$PY -u wlat_rom.py ../out/wlat_ad_N64_K8.pkl ../out rs40"
  mk "$cell" 96G 20 "N=64 K_LAT=$k ROM_SUBSTEPS=$RS AD_STEPS=80000 AD_BATCH=128 P_SUB=2048 $BASE POD_KS=6,8,16,32,64 VARIANTS=$VAR POD_VARIANTS=$PODV" \
     "\$PY -u wlat_verify.py ../out && \$PY -u wlat_train_ad.py ../out && \$PY -u wlat_rom.py ../out/wlat_ad_N64_K$k.pkl ../out$extra"
done
GPU_TYPE=a100 mk wad_n128_k8 160G 24 \
  "N=128 K_LAT=8 ROM_SUBSTEPS=$RS AD_STEPS=80000 AD_BATCH=128 P_SUB=2048 $BASE POD_KS=6,8,16,32,64 VARIANTS=$VAR POD_VARIANTS=$PODV OMP_NUM_THREADS=8" \
  '$PY -u wlat_verify.py ../out && $PY -u wlat_train_ad.py ../out && $PY -u wlat_rom.py ../out/wlat_ad_N128_K8.pkl ../out'

if [[ "$ACT" == launch ]]; then
  for cell in ws1_n64_icw1 ws1_n64_icwsqrt50 wad_n64_k4 wad_n64_k8 wad_n64_k16 wad_n128_k8; do
    "$HERE/launch.sh" "$cell"
  done
fi
