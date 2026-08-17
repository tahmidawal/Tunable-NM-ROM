#!/usr/bin/env bash
# cells.sh : build every job directory for this cell.  ONE JOB PER DIRECTORY.
#
# FAN-OUT ("panel" role, one N per job, all submitted simultaneously, all on a100):
#   wsp_n{32,64,128,256,512}   Poisson, one mesh each
#   wsb_n{32,64,128,256}       Burgers, one mesh each
#   These produce the GPU-IDENTITY-INDEPENDENT quantities -- EQ/NNLS refits, ROM
#   accuracy, CG / Newton / BiCGStab ITERATION COUNTS from both starts, the NaN-guard
#   checks -- plus a WITHIN-N timing breakdown that is valid because one panel is one
#   job on one GPU.  Their wall clock must never be placed on a cross-N axis.
#
# CONSOLIDATION ("consolidated" role, every N sequentially in ONE job on ONE GPU):
#   wsp_cons   Poisson: the full (rom_tau x N) grid + the pure-FOM baseline at every N
#   wsb_cons   Burgers: the full rollout from both starts at every N
#   These are the ONLY timing source for the cross-N figures and the crossover-N claim.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MK="$HERE/make_cell.sh"
GPU="${GPU_TYPE:-h100}"
# All eleven jobs request the SAME GPU TYPE.  h100 rather than a100 only because the
# a100 queue was 28 deep with our fairshare exhausted while the h100 queue was empty;
# what matters for the contract is that every job is on one type and that each
# consolidation job measures its whole N ladder on ONE card.

PENV="PKL=in/autodec_K8_N64_hbc_stages.pkl N=64 M=64 MQ=256 EQ_SNAPS=64 EQ_PERTURB=3 \
EQ_ROWS=3072 INIT=mean N_TEST=16 N_TIME=8 GN_ITERS=60 TIME_REPS=7 TIME_WARM=2 \
ROM_TAUS=0.5,1e-1,1e-2,1e-3,0 FOM_TAUS=1e-6,1e-8,1e-10"

BENV="PKL=in/blat_ad_N64_K8.pkl N=64 K_LAT=8 VARIANT=lspg:eq256:weak64 EQ_SNAPS=64 \
N_TEST_TRAJ=4 TIME_REPS=7 TIME_WARM=2 FOM_TAUS=1e-6,1e-8,1e-10 MAX_NEWTON=25"

# Walltimes are deliberately TIGHT: a 6 h request backfills far worse than a 3 h
# one, and with fairshare exhausted backfill is the only thing that starts a job.
# ---------------- Poisson panels ----------------
for n in 32 64 128 256 512; do
  mem=48G; hrs=00:45:00
  [[ $n -ge 256 ]] && { mem=64G; hrs=01:15:00; }
  bash "$MK" "wsp_n$n" "$mem" "$hrs" "$PENV NS=$n RUN_ROLE=panel" \
    "\$PY wsf_poisson.py ../../out/wsp_n$n.json" "$GPU" >/dev/null
  echo "built wsp_n$n"
done

# ---------------- Burgers panels ----------------
for n in 32 64 128 256; do
  mem=48G; hrs=01:00:00
  [[ $n -ge 256 ]] && { mem=64G; hrs=01:30:00; }
  bash "$MK" "wsb_n$n" "$mem" "$hrs" "$BENV NS=$n RUN_ROLE=panel" \
    "\$PY wsf_burgers.py ../../out/wsb_n$n.json" "$GPU" >/dev/null
  echo "built wsb_n$n"
done

# ---------------- consolidation (cross-N wall clock; ONE GPU, sequential) ----------------
bash "$MK" wsp_cons 64G 01:30:00 "$PENV NS=32,64,128,256,512 RUN_ROLE=consolidated" \
  "\$PY wsf_poisson.py ../../out/wsp_cons.json" "$GPU" >/dev/null
echo "built wsp_cons"
bash "$MK" wsb_cons 64G 03:00:00 "$BENV NS=32,64,128,256 RUN_ROLE=consolidated" \
  "\$PY wsf_burgers.py ../../out/wsb_cons.json" "$GPU" >/dev/null
echo "built wsb_cons"
