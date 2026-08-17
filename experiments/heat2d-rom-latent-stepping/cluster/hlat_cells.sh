#!/usr/bin/env bash
# Build every heat2d latent-stepping cell under cluster/stage/.  One job per dir.
#   ./hlat_cells.sh            # build all
#   ./hlat_cells.sh s1_n64     # build one
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WT="$(cd "$EXP/../../.." && pwd)"
SWEEP="$WT/2026-08-13-heat2d-coord-decoder/experiments/heat2d-coord-decoder/sweep"
only="${1:-}"

want() { [[ -z "$only" || "$only" == "$1" ]]; }
manifest() { (cd "$1" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256); }

AD_ENV_COMMON="AD_STEPS=60000 AD_BATCH=128 P_SUB=2048 N_TEST=16 FLOOR_BUDGET=40 GN_BUDGET=30 POD_KMAX=64 POD_KS=6,8,16,32,64"

# ---- Stage 1: space-time LSPG on the existing (z,t) sweep decoder, N=64 -------
if want s1_n64; then
  S=$("$HERE/make_cell.sh" s1_n64 64G 8 \
    "N=64 N_TEST=16 S1_BUDGET=100" \
    'mkdir -p ../out/icw_sqrt50 ../out/icw1 && $PY -u hlat_stage1.py in/heat2d_film_N64.pkl ../out/icw_sqrt50 && IC_W=1 $PY -u hlat_stage1.py in/heat2d_film_N64.pkl ../out/icw1')
  cp "$SWEEP/heat2d_film_N64.pkl" "$SWEEP/heat2d_results_N64.json" "$S/code/in/"
  manifest "$S"; echo "$S"
fi

# ---- Stage 2: auto-decoder + full ROM study ---------------------------------
for K in 4 8 16; do
  cell="ad_n64_k$K"
  want "$cell" || continue
  S=$("$HERE/make_cell.sh" "$cell" 96G 16 \
    "N=64 K_LAT=$K $AD_ENV_COMMON" \
    '$PY -u hlat_train_ad.py ../out && $PY -u hlat_rom.py ../out/hlat_ad_N64_K'"$K"'.pkl ../out')
  manifest "$S"; echo "$S"
done

# ---- N-flatness: same K, M, m at N=128 (weakall dropped: (N-2)^2 = 15876
#      modes would need a 2 GB dense Phi; the exactness check is done at N=64) --
if want ad_n128_k8; then
  V128="lspg:full:fd,galerkin:full:fd,lspg:full:weak64,galerkin:full:weak64,"
  V128+="lspg:full:weak256,lspg:eq256:weak64,lspg:eq512:weak64,"
  V128+="lspg:eqoff256:weak64,lspg:eqoff512:weak64,lspg:eq1024:weak256,lspg:full:weakc64"
  S=$("$HERE/make_cell.sh" ad_n128_k8 200G 20 \
    "N=128 K_LAT=8 $AD_ENV_COMMON VARIANTS=$V128" \
    '$PY -u hlat_train_ad.py ../out && $PY -u hlat_rom.py ../out/hlat_ad_N128_K8.pkl ../out')
  manifest "$S"; echo "$S"
fi
