#!/bin/bash
# Stage one 2D-Burgers TENSOR-arm job (2026-08-29/30): cluster/stage/<jobname>/.
# Same construction as stage_exlin.sh (every dep verified byte-identical to the
# verified N=64 stage; the staged burgers2d_film.py gets the exact sine-basis
# Helmholtz preconditioner on the truth generator's inner BiCGStab -- the
# unpreconditioned solver stalls at N>=512).  Extra payloads (checkpoint .pkl,
# node .npz) go under in/.
# Usage: ./stage_b2dtensor.sh <jobname> [extra files to copy into in/ ...]
set -euo pipefail
JOB=${1:?jobname}; shift || true
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
         "$DST/in" "$DST/out" "$DST/logs"

cp "$SEP"/sep_*.py "$SEP"/exlin_common.py "$SEP"/b2d_tensor_common.py "$DST/code/"
[ -f "$SEP/pod_floor_n256.py" ] && cp "$SEP/pod_floor_n256.py" "$DST/code/"
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

for extra in "$@"; do cp "$extra" "$DST/in/"; done

# verify UNPATCHED deps against the verified N=64 stage, before patching
fail=0
while read -r f; do
  rel=${f#"$DST/code/"}
  case "$rel" in sep_*|exlin_common.py|b2d_tensor_common.py|b1d_*|pod_floor_n256.py) continue;; esac
  if [ -f "$OLD/code/$rel" ]; then
    a=$(sha256sum "$f" | cut -d' ' -f1); b=$(sha256sum "$OLD/code/$rel" | cut -d' ' -f1)
    if [ "$a" != "$b" ]; then echo "DEP MISMATCH vs verified N=64 stage: $rel"; fail=1; fi
  else
    echo "NOTE: $rel not in N=64 stage (new file)"
  fi
done < <(find "$DST/code" -name '*.py')
[ "$fail" -eq 0 ] || { echo "STAGING ABORTED"; exit 1; }

python3 "$HERE/patch_bf_precond.py" \
  "$DST/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py"

git -C "$SEP" rev-parse HEAD > "$DST/COMMIT.txt"
( cd "$DST" && find code in COMMIT.txt -type f \( -name '*.py' -o -name '*.pkl' -o -name '*.npz' -o -name 'COMMIT.txt' \) -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
echo "staged -> $DST  ($(wc -l < "$DST/MANIFEST.sha256") files, commit $(cat "$DST/COMMIT.txt"))"
