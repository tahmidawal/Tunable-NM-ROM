#!/usr/bin/env bash
# Follow-up cells for the Burgers-2D latent-stepping ROM (cluster namespace blat2/).
#   ./fu_cells.sh wave1     k-ladder, multi-seed, m/M ladders, timing-vs-N
#   ./fu_cells.sh wave2     timing-vs-K (needs every wave-1 K checkpoint pulled into runs/)
# One cell = one job dir = one job.  Build here, then launch.sh <cell>.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$(dirname "$HERE")")"
MK="$HERE/make_cell.sh"
K8="$EXP/runs/ad_n64_k8/blat_ad_N64_K8.pkl"

# training recipe: IDENTICAL to the frozen round (equal budget across the k ladder)
TRAIN="N=64 AD_STEPS=60000 AD_BATCH=128 P_SUB=2048 N_TEST=16 FLOOR_BUDGET=40 GN_BUDGET=30"
EVAL="N=64 N_TEST=16 FLOOR_BUDGET=40 GN_BUDGET=30"
# the four arms the frozen K=4/8/16 rows also carry, so the ladder is comparable
KVARS="lspg:full:fd,lspg:full:weak64,lspg:eq256:weak64,lspg:eq512:weak64"

stamp() { (cd "$1" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256); }

wave="${1:?wave1|wave2}"
if [[ "$wave" == "wave1" ]]; then
  # ---- k ladder: retrain the auto-decoder per K at equal budget (K=4,8,16 already exist).
  # The POD control (basis from the same TRAIN snapshots, same solver, same objective) does
  # not depend on K, so it is computed once, in the K=2 cell, over the whole k ladder.
  for K in 2 6 12 24 32; do
    PODKS=8; PODV=lspg:full:fd
    if [[ $K == 2 ]]; then PODKS=2,4,6,8,12,16,24,32,64; PODV=lspg:full:fd,lspg:full:weak64,lspg:eq256:weak64; fi
    "$MK" bk_K$K 64G 12 "$TRAIN K_LAT=$K POD_KS=$PODKS POD_VARIANTS=$PODV VARIANTS=$KVARS" \
      "\$PY -u blat_train_ad.py ../out && \$PY -u blat_rom.py ../out/blat_ad_N64_K$K.pkl ../out"
  done
  # ---- multi-seed: K=8, TRAIN_SEED 1 and 2 (seed 0 = the frozen runs/ad_n64_k8 cell).
  # TRAIN_SEED changes the FiLM net initialisation and the minibatch / collocation-point
  # order ONLY.  The per-snapshot latents are initialised deterministically from the POD
  # coefficients, and the data draw, the train/val split and the TEST_SEED test set are
  # untouched -- so the spread below is training randomness, nothing else.
  for S in 1 2; do
    "$MK" bs_S$S 64G 12 "$TRAIN K_LAT=8 TRAIN_SEED=$S POD_KS=8,16,32,64 POD_VARIANTS=lspg:full:fd VARIANTS=$KVARS" \
      "\$PY -u blat_train_ad.py ../out && \$PY -u blat_rom.py ../out/blat_ad_N64_K8_S$S.pkl ../out"
  done
  # ---- m ladder at fixed (K=8, M=64): NNLS-EQ on GRID nodes for the exact-FOM weak form,
  # and grid vs MESHFREE pool for the continuum weak form (the meshfree pool is only
  # available for weakc -- the FOM's upwind stencil needs grid neighbours).
  MV="lspg:full:weak64"
  for m in 64 128 256 512 1024; do MV="$MV,lspg:eq$m:weak64"; done
  MV="$MV,lspg:full:weakc64"
  for m in 64 128 256 512 1024; do MV="$MV,lspg:eq$m:weakc64"; done
  for m in 64 128 256 512 1024; do MV="$MV,lspg:eqoff$m:weakc64"; done
  d=$("$MK" bm_m 64G 12 "$EVAL K_LAT=8 POD_KS=8 POD_VARIANTS=lspg:full:fd VARIANTS=$MV" \
      "\$PY -u blat_rom.py in/blat_ad_N64_K8.pkl ../out")
  cp "$K8" "$d/code/in/"; stamp "$d"
  # ---- M ladder at m ~ 4M (and the full grid at each M), fixed K=8
  MV=""
  for M in 16 32 64 128 256; do MV="$MV,lspg:full:weak$M,lspg:eq$((4*M)):weak$M"; done
  d=$("$MK" bm_M 64G 12 "$EVAL K_LAT=8 POD_KS=8 POD_VARIANTS=lspg:full:fd VARIANTS=${MV#,}" \
      "\$PY -u blat_rom.py in/blat_ad_N64_K8.pkl ../out")
  cp "$K8" "$d/code/in/"; stamp "$d"
  # ---- per-rollout / per-iteration cost across the m and M ladders on ONE GPU, all
  # variants sequential in ONE process (median of 7, device sync).  ACCURACY for those
  # ladders comes from bm_m / bm_M (blat_rom, 16 trajectories); this cell is the cost column.
  TV="lspg:full:weak64"
  for m in 64 128 256 512 1024; do TV="$TV,lspg:eq$m:weak64"; done
  TV="$TV,lspg:full:weakc64"
  for m in 64 128 256 512 1024; do TV="$TV,lspg:eqoff$m:weakc64"; done
  for M in 16 32 64 128 256; do TV="$TV,lspg:full:weak$M,lspg:eq$((4*M)):weak$M"; done
  d=$("$MK" bt_m 64G 8 "MODE=k N=64 K_LAT=8 N_TEST=16 PKLS=in/blat_ad_N64_K8.pkl POD_KS= TIME_REPS=7 VARIANTS=$TV" \
      "\$PY -u followup/fu_timing.py ../out/timing_m.json")
  cp "$K8" "$d/code/in/"; stamp "$d"
  # ---- timing vs N on ONE GPU, all N sequential in ONE process (K=8 decoder; it is
  # meshfree, so the SAME checkpoint is used at every N and the EQ weights are refit per N)
  d=$("$MK" bt_n 96G 8 "MODE=n N=64 K_LAT=8 N_TEST=16 PKL=in/blat_ad_N64_K8.pkl NS=32,64,128,256 TIME_REPS=7 VARIANTS=lspg:eq256:weak64,lspg:eq512:weak64,lspg:full:weak64,lspg:eqoff512:weakc64" \
      "\$PY -u followup/fu_timing.py ../out/timing_n.json")
  cp "$K8" "$d/code/in/"; stamp "$d"
elif [[ "$wave" == "wave2" ]]; then
  # ---- timing vs K on ONE GPU, all K sequential in ONE process (+ the POD ladder)
  PK=""
  for K in 2 4 6 8 12 16 24 32; do
    src="$EXP/runs/ad_n64_k$K/blat_ad_N64_K$K.pkl"; [[ -f "$src" ]] || src="$EXP/runs/bk_K$K/blat_ad_N64_K$K.pkl"
    [[ -f "$src" ]] || { echo "missing K=$K checkpoint (looked in runs/ad_n64_k$K and runs/bk_K$K)" >&2; exit 1; }
    PK="$PK,in/blat_ad_N64_K$K.pkl"
  done
  d=$("$MK" bt_k 64G 8 "MODE=k N=64 K_LAT=8 N_TEST=16 PKLS=${PK#,} POD_KS=2,4,6,8,12,16,24,32,64 TIME_REPS=7 VARIANTS=lspg:eq256:weak64,lspg:full:weak64 POD_VARIANT=lspg:full:fd" \
      "\$PY -u followup/fu_timing.py ../out/timing_k.json")
  for K in 2 4 6 8 12 16 24 32; do
    src="$EXP/runs/ad_n64_k$K/blat_ad_N64_K$K.pkl"; [[ -f "$src" ]] || src="$EXP/runs/bk_K$K/blat_ad_N64_K$K.pkl"
    cp "$src" "$d/code/in/"
  done
  stamp "$d"
else
  echo "unknown wave '$wave' (expected wave1 or wave2)" >&2; exit 2
fi
echo "cells built under $HERE/stage"
