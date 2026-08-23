#!/bin/bash
# Assemble the staged code tree for one sepdec_n512 cluster job.
# Usage: ./make_stage.sh <jobname>   (e.g. j1)  -> stage/<jobname>/code/...
# Deps are copied from THIS worktree's sibling experiment dirs (same commit
# as the audited sepdec_r1 stage); sep_*/ctol_* come from this directory.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SEP="$(dirname "$HERE")"                       # experiments/separable-decoder
EXP="$(dirname "$SEP")"                        # experiments/
WTS="$(cd "$EXP/../.." && pwd)"                # worktrees/
JOB=${1:?jobname}
ST="$HERE/stage/$JOB"
rm -rf "$ST"
mkdir -p "$ST/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom" \
         "$ST/code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision" \
         "$ST/code/deps/burgers2d-rom-latent-stepping/followup" \
         "$ST/code/deps/nonlinear-decoder-architecture" \
         "$ST/code/deps/poisson2d-rom-objective/deps" \
         "$ST/code/deps/poisson2d-rom-objective/followup"

cp "$SEP"/sep_common.py "$SEP"/sep_poisson.py "$SEP"/sep_burgers.py "$ST/code/"
cp "$EXP"/cost-to-tolerance/ctol_eq.py "$EXP"/cost-to-tolerance/ctol_tol.py "$ST/code/"

B="$ST/code/deps/burgers2d-rom-latent-stepping"
cp "$EXP"/burgers2d-rom-latent-stepping/blat_common.py "$B/"
cp "$EXP"/burgers2d-rom-latent-stepping/blat_train_ad.py "$B/"
cp "$WTS"/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/followup/fu_common.py "$B/followup/" 2>/dev/null || \
  cp "$EXP"/burgers2d-rom-latent-stepping/followup/fu_common.py "$B/followup/"
cp "$EXP"/burgers2d-rom-latent-stepping/followup/fu_style.py "$B/followup/" 2>/dev/null || true
cp "$WTS"/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py "$B/deps/burgers2d-coord-rom/"
cp "$WTS"/2026-08-14-multistage-precision/experiments/multistage-precision/ms_parametric.py \
   "$WTS"/2026-08-14-multistage-precision/experiments/multistage-precision/ms_autodecoder.py \
   "$B/deps/multistage-precision/"

P="$ST/code/deps/poisson2d-rom-objective"
cp "$EXP"/poisson2d-rom-objective/pro_common.py "$P/"
cp "$EXP"/poisson2d-rom-objective/followup/fu_eq.py "$EXP"/poisson2d-rom-objective/followup/fu_train.py \
   "$EXP"/poisson2d-rom-objective/followup/fu_style.py "$P/followup/" 2>/dev/null || true
cp "$B/deps/multistage-precision/ms_parametric.py" "$B/deps/multistage-precision/ms_autodecoder.py" "$P/deps/"
cp "$EXP"/nonlinear-decoder-architecture/nda_arch.py "$ST/code/deps/nonlinear-decoder-architecture/"

( cd "$ST" && find code -name '*.py' | sort | xargs sha256sum > MANIFEST.sha256 )
echo "staged -> $ST"
wc -l "$ST/MANIFEST.sha256"
