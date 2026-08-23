#!/bin/bash
# Assemble the staged code tree for one sepdec_n128 cluster job and rsync it.
# Usage: stage.sh <jobname>   (e.g. j1)  -- one job per remote directory.
set -euo pipefail
JOB=${1:?usage: stage.sh <jobname>}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")                                  # separable-decoder
EXP=$(dirname "$SEP")                                   # experiments/
WTS=$(cd "$EXP/../.." && pwd)                           # worktrees root
BCR=$WTS/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom
MSP=$WTS/2026-08-14-multistage-precision/experiments/multistage-precision
STAGE=$HERE/stage/$JOB
rm -rf "$STAGE"
mkdir -p "$STAGE"/{code,out,logs,data}
C=$STAGE/code

cp "$SEP"/sep_common.py "$SEP"/sep_poisson.py "$SEP"/sep_burgers.py "$C/"
cp "$EXP"/cost-to-tolerance/ctol_eq.py "$EXP"/cost-to-tolerance/ctol_tol.py "$C/"

mkdir -p "$C"/deps/burgers2d-rom-latent-stepping/{followup,deps/burgers2d-coord-rom,deps/multistage-precision}
cp "$EXP"/burgers2d-rom-latent-stepping/blat_common.py \
   "$EXP"/burgers2d-rom-latent-stepping/blat_train_ad.py \
   "$C"/deps/burgers2d-rom-latent-stepping/
cp "$EXP"/burgers2d-rom-latent-stepping/followup/fu_common.py \
   "$EXP"/burgers2d-rom-latent-stepping/followup/fu_style.py \
   "$C"/deps/burgers2d-rom-latent-stepping/followup/
cp "$BCR"/burgers2d_film.py "$C"/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/
cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py "$C"/deps/burgers2d-rom-latent-stepping/deps/multistage-precision/

mkdir -p "$C"/deps/nonlinear-decoder-architecture
cp "$EXP"/nonlinear-decoder-architecture/nda_arch.py "$C"/deps/nonlinear-decoder-architecture/

mkdir -p "$C"/deps/poisson2d-rom-objective/{deps,followup}
cp "$EXP"/poisson2d-rom-objective/pro_common.py "$C"/deps/poisson2d-rom-objective/
cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py "$C"/deps/poisson2d-rom-objective/deps/
cp "$EXP"/poisson2d-rom-objective/followup/fu_eq.py \
   "$EXP"/poisson2d-rom-objective/followup/fu_style.py \
   "$EXP"/poisson2d-rom-objective/followup/fu_train.py \
   "$C"/deps/poisson2d-rom-objective/followup/

cp "$HERE/run_$JOB.sbatch" "$STAGE/run.sbatch"
( cd "$STAGE" && find code run.sbatch -type f | sort | xargs sha256sum > MANIFEST.sha256 )
echo "staged $(find "$C" -type f | wc -l) code files -> $STAGE"

REMOTE=/cluster/tufts/paralab/tawal01/sepdec_n128/$JOB
ssh tufts-login "mkdir -p $REMOTE"
rsync -a --delete "$STAGE"/ tufts-login:"$REMOTE"/
ssh tufts-login "cd $REMOTE && sha256sum -c MANIFEST.sha256 --quiet && echo REMOTE-MANIFEST-OK"
echo "synced -> tufts-login:$REMOTE"
