#!/bin/bash
# Build a staged code tree for one ROUND-3 cluster job: cluster/stage/<jobname>/
# Usage: ./stage_r3.sh <jobname>          (e.g. r3a, r3d)
#
# Same construction as stage_n256.sh (every dep that also existed in the
# verified N=64 stage is checked byte-identical to it), plus ONE deliberate
# difference recorded here:
#
#   the STAGED burgers2d_film.py gets the exact sine-basis Helmholtz
#   preconditioner on the truth generator's inner BiCGStab
#   (cluster/patch_bf_precond.py, carried over from the N=1024 scaling arm).
#   The unpreconditioned solver STALLS at N=1024 (job 2825735: max Newton rel
#   residual 8.67e-2, data generation aborted).  The discrete residual, the
#   Newton acceptance guard, and the <=1e-8 truth-convergence check are
#   untouched -- only how the linear solve reaches them changes.  It is applied
#   at BOTH resolutions so the N=256 and N=1024 truth are produced by the same
#   generator and the cross-N comparison is like-for-like.
set -euo pipefail
JOB=${1:?jobname}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")
EXP=$(dirname "$SEP")
WTS=/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees
BCR=$WTS/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom
MSP=$WTS/2026-08-14-multistage-precision/experiments/multistage-precision
OLD=$WTS/2026-08-22-separable-decoder/experiments/separable-decoder/cluster/stage/sepdec_r1

DST=$HERE/stage/$JOB
rm -rf "$DST"
mkdir -p "$DST/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom" \
         "$DST/code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision" \
         "$DST/code/deps/burgers2d-rom-latent-stepping/followup" \
         "$DST/code/deps/poisson2d-rom-objective/deps" \
         "$DST/code/deps/poisson2d-rom-objective/followup" \
         "$DST/code/deps/nonlinear-decoder-architecture" \
         "$DST/out" "$DST/logs"

cp "$SEP"/sep_common.py "$SEP"/sep_poisson.py "$SEP"/sep_burgers.py "$DST/code/"
for f in sep_solvers.py sep_poisson_r1.py sep_burgers_r1.py sep_poisson_r2.py \
         sep_burgers_r2.py sep_burgers_r3.py sep_burgers_r4.py sep_speed_r4.py pod_floor_n256.py; do
  [ -f "$SEP/$f" ] && cp "$SEP/$f" "$DST/code/"
done
cp "$EXP"/cost-to-tolerance/ctol_eq.py "$EXP"/cost-to-tolerance/ctol_tol.py "$DST/code/"
cp "$EXP"/burgers2d-rom-latent-stepping/blat_common.py \
   "$EXP"/burgers2d-rom-latent-stepping/blat_train_ad.py \
   "$DST/code/deps/burgers2d-rom-latent-stepping/"
cp "$EXP"/burgers2d-rom-latent-stepping/followup/fu_common.py \
   "$EXP"/burgers2d-rom-latent-stepping/followup/fu_style.py \
   "$DST/code/deps/burgers2d-rom-latent-stepping/followup/"
cp "$EXP"/poisson2d-rom-objective/pro_common.py \
   "$DST/code/deps/poisson2d-rom-objective/"
cp "$EXP"/poisson2d-rom-objective/followup/fu_eq.py \
   "$EXP"/poisson2d-rom-objective/followup/fu_style.py \
   "$EXP"/poisson2d-rom-objective/followup/fu_train.py \
   "$DST/code/deps/poisson2d-rom-objective/followup/"
cp "$EXP"/nonlinear-decoder-architecture/nda_arch.py \
   "$DST/code/deps/nonlinear-decoder-architecture/"
cp "$BCR"/burgers2d_film.py \
   "$DST/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/"
cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py \
   "$DST/code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision/"
cp "$MSP"/ms_parametric.py "$MSP"/ms_autodecoder.py \
   "$DST/code/deps/poisson2d-rom-objective/deps/"

# verify UNPATCHED deps against the verified N=64 stage, before patching
fail=0
while read -r f; do
  rel=${f#"$DST/code/"}
  case "$rel" in sep_*) continue;; esac
  if [ -f "$OLD/code/$rel" ]; then
    a=$(sha256sum "$f" | cut -d' ' -f1); b=$(sha256sum "$OLD/code/$rel" | cut -d' ' -f1)
    if [ "$a" != "$b" ]; then echo "DEP MISMATCH vs verified N=64 stage: $rel"; fail=1; fi
  else
    echo "NOTE: $rel not in N=64 stage (new file)"
  fi
done < <(find "$DST/code" -name '*.py')
[ "$fail" -eq 0 ] || { echo "STAGING ABORTED"; exit 1; }

# deliberate, recorded patch (see header)
python3 "$HERE/patch_bf_precond.py" \
  "$DST/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py"

( cd "$DST" && find code -type f -name '*.py' -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
echo "staged -> $DST  ($(wc -l < "$DST/MANIFEST.sha256") files)"
