#!/usr/bin/env bash
# Follow-up cells (namespace blat2/).  Usage: ./fu_cells.sh <wave1|wave2>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$(dirname "$HERE")")"
MK="$HERE/make_cell.sh"
K8="$EXP/runs/ad_n64_k8/blat_ad_N64_K8.pkl"
TRAIN="N=64 AD_STEPS=60000 AD_BATCH=128 P_SUB=2048 N_TEST=16 FLOOR_BUDGET=40 GN_BUDGET=30"
KVARS="lspg:full:fd,lspg:full:weak64,lspg:eq256:weak64,lspg:eq512:weak64"
wave="${1:-wave1}"
if [[ "$wave" == "wave1" ]]; then
  # k-ladder (K=4,8,16 exist from the main round)
  for K in 2 6 12 24 32; do
    PODKS=8; [[ $K == 2 ]] && PODKS=2,4,6,8,12,16,24,32,64
    "$MK" bk_K$K 64G 10 "$TRAIN K_LAT=$K POD_KS=$PODKS POD_VARIANTS=lspg:full:fd,lspg:full:weak64,lspg:eq256:weak64 VARIANTS=$KVARS" \
      "\$PY -u blat_train_ad.py ../out && \$PY -u blat_rom.py ../out/blat_ad_N64_K$K.pkl ../out"
  done
  # multi-seed (seed 0 = ad_n64_k8)
  for S in 1 2; do
    "$MK" bs_S$S 64G 10 "$TRAIN K_LAT=8 TRAIN_SEED=$S POD_KS=8 POD_VARIANTS=lspg:full:fd,lspg:full:weak64,lspg:eq256:weak64 VARIANTS=$KVARS" \
      "\$PY -u blat_train_ad.py ../out && \$PY -u blat_rom.py ../out/blat_ad_N64_K8.pkl ../out"
  done
  # m / M ladder on the existing K=8 checkpoint
  MVARS="lspg:eq64:weak64,lspg:eq128:weak64,lspg:eq256:weak64,lspg:eq512:weak64,lspg:eq1024:weak64,lspg:full:weak64"
  MVARS="$MVARS,lspg:eq64:weak16,lspg:full:weak16,lspg:eq128:weak32,lspg:full:weak32,lspg:eq512:weak128,lspg:full:weak128,lspg:eq1024:weak256,lspg:full:weak256"
  d=$("$MK" bm_K8 64G 8 "N=64 K_LAT=8 N_TEST=16 FLOOR_BUDGET=40 GN_BUDGET=30 POD_KS=8 POD_VARIANTS=lspg:full:fd VARIANTS=$MVARS" \
      "\$PY -u blat_rom.py in/blat_ad_N64_K8.pkl ../out")
  cp "$K8" "$d/code/in/"; (cd "$d" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
  # timing vs N on ONE GPU (K=8 decoder, meshfree, reused at every N)
  d=$("$MK" bt_n 96G 6 "MODE=n N=64 K_LAT=8 PKL=in/blat_ad_N64_K8.pkl NS=32,64,128,256 TIME_REPS=7" \
      "\$PY -u followup/fu_timing.py ../out/timing_n.json")
  cp "$K8" "$d/code/in/"; (cd "$d" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
else
  # timing vs K on ONE GPU: all K checkpoints (needs wave-1 results pulled into runs/)
  PK=""
  for K in 2 4 6 8 12 16 24 32; do
    src="$EXP/runs/ad_n64_k$K/blat_ad_N64_K$K.pkl"; [[ -f "$src" ]] || src="$EXP/runs/bk_K$K/blat_ad_N64_K$K.pkl"
    [[ -f "$src" ]] || { echo "missing $src" >&2; exit 1; }
    PK="$PK,in/blat_ad_N64_K$K.pkl"
  done
  d=$("$MK" bt_k 64G 6 "MODE=k N=64 K_LAT=8 PKLS=${PK#,} POD_KS=2,4,6,8,12,16,24,32,64 TIME_REPS=7" \
      "\$PY -u followup/fu_timing.py ../out/timing_k.json")
  for K in 2 4 6 8 12 16 24 32; do
    src="$EXP/runs/ad_n64_k$K/blat_ad_N64_K$K.pkl"; [[ -f "$src" ]] || src="$EXP/runs/bk_K$K/blat_ad_N64_K$K.pkl"
    cp "$src" "$d/code/in/"
  done
  (cd "$d" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
fi
echo "cells built under $HERE/stage"
