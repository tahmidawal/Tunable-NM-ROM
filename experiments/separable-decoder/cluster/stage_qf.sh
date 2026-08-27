#!/bin/bash
# Build a staged code tree for one quadrature-free Poisson job (sep_poisson_qf.py):
# cluster/stage/<jobname>/.  Modeled on stage_exlin.sh, restricted to the poisson
# dependency chain -- this cell trains nothing and never touches the Burgers stack,
# so only the modules sep_poisson_qf.py actually imports are staged:
#   code/sep_poisson_qf.py, sep_common.py            (this experiment)
#   code/ctol_eq.py, ctol_tol.py                     (flat, per sep_common._bootstrap)
#   code/deps/poisson2d-rom-objective/pro_common.py
#   code/deps/poisson2d-rom-objective/deps/{ms_parametric.py, ms_autodecoder.py}
#   code/deps/nonlinear-decoder-architecture/nda_arch.py
# every file byte-copied from where the worktree's sys.path bootstrap resolves it.
# The frozen checkpoint goes to in/.  A MANIFEST.sha256 covers code/ + in/, checked
# by the sbatch before running (same protocol as run_cdmm_*.sbatch).
#
# Usage: ./stage_qf.sh <jobname> <checkpoint.pkl> [extra files to copy into in/ ...]
#   e.g. ./stage_qf.sh qf_n128 ../runs/inherited_qf/sep_poisson_N128_K16_R96.pkl
set -euo pipefail
JOB=${1:?jobname}
CKPT=${2:?checkpoint pkl}
shift 2 || true
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")                       # experiments/separable-decoder
EXP=$(dirname "$SEP")                        # experiments/
WT=$(dirname "$EXP")                         # worktree root
WTS=/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees
MSP=$WTS/2026-08-14-multistage-precision/experiments/multistage-precision

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }

DST=$HERE/stage/$JOB
rm -rf "$DST"
mkdir -p "$DST/code/deps/poisson2d-rom-objective/deps" \
         "$DST/code/deps/nonlinear-decoder-architecture" \
         "$DST/in" "$DST/out" "$DST/logs"

cp "$SEP/sep_poisson_qf.py" "$SEP/sep_common.py" "$DST/code/"
cp "$EXP/cost-to-tolerance/ctol_eq.py" "$EXP/cost-to-tolerance/ctol_tol.py" \
   "$DST/code/"
cp "$EXP/poisson2d-rom-objective/pro_common.py" \
   "$DST/code/deps/poisson2d-rom-objective/"
cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" \
   "$DST/code/deps/poisson2d-rom-objective/deps/"
cp "$EXP/nonlinear-decoder-architecture/nda_arch.py" \
   "$DST/code/deps/nonlinear-decoder-architecture/"
git -C "$WT" rev-parse HEAD > "$DST/code/GIT_COMMIT" 2>/dev/null \
  || echo unknown > "$DST/code/GIT_COMMIT"

cp "$CKPT" "$DST/in/"
for extra in "$@"; do cp "$extra" "$DST/in/"; done

( cd "$DST" && find code in -type f \
    \( -name '*.py' -o -name '*.pkl' -o -name '*.npz' -o -name GIT_COMMIT \) \
    -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
echo "staged -> $DST  ($(wc -l < "$DST/MANIFEST.sha256") files)"
