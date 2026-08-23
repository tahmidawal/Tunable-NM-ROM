#!/bin/bash
# Assemble the self-contained cluster stage for the separable-decoder N=1024
# round.  Layout matches the bootstraps in sep_common / pro_common /
# blat_common (deps/ trees).  Run from this directory.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SEP="$HERE/.."
EXP="$SEP/.."                                   # experiments/
WTS="$EXP/../.."                                # worktrees/
STAGE="$HERE/stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/deps/cost-to-tolerance" \
         "$STAGE/deps/poisson2d-rom-objective/deps" \
         "$STAGE/deps/nonlinear-decoder-architecture" \
         "$STAGE/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom" \
         "$STAGE/deps/burgers2d-rom-latent-stepping/deps/multistage-precision"

cp "$SEP"/sep_common.py "$SEP"/sep_poisson.py "$SEP"/sep_burgers.py "$STAGE"/
cp "$EXP"/cost-to-tolerance/ctol_eq.py "$EXP"/cost-to-tolerance/ctol_tol.py \
   "$STAGE/deps/cost-to-tolerance/"
cp "$EXP"/poisson2d-rom-objective/pro_common.py \
   "$STAGE/deps/poisson2d-rom-objective/"
MSP="$WTS/2026-08-14-multistage-precision/experiments/multistage-precision"
cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py \
   "$STAGE/deps/poisson2d-rom-objective/deps/"
cp "$EXP"/nonlinear-decoder-architecture/nda_arch.py \
   "$STAGE/deps/nonlinear-decoder-architecture/"
cp "$EXP"/burgers2d-rom-latent-stepping/blat_common.py \
   "$STAGE/deps/burgers2d-rom-latent-stepping/"
BCR="$WTS/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom"
cp "$BCR"/burgers2d_film.py \
   "$STAGE/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/"
cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py \
   "$STAGE/deps/burgers2d-rom-latent-stepping/deps/multistage-precision/"

( cd "$STAGE" && find . -name '*.py' -type f | sort | xargs sha256sum ) \
  > "$HERE/stage.manifest"
echo "staged $(grep -c . "$HERE/stage.manifest") files -> $STAGE"
