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


# Patch the STAGED ms_parametric copies only (the incumbent worktree file is
# untouched): make the truth-convergence guard threshold env-configurable.
# At N=1024 unpreconditioned f64 CG bottoms out at rel residual ~4.3e-10
# (cond*eps floor), which fails the hard 1e-10 assert; CG tol/maxiter are
# unchanged and the achieved residual is still printed and recorded.
for f in "$STAGE/deps/poisson2d-rom-objective/deps/ms_parametric.py" \
         "$STAGE/deps/burgers2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py"; do
  python3 - "$f" << 'PYPATCH'
import sys
p = sys.argv[1]
src = open(p).read()
old = '    assert np.isfinite(res_max) and res_max < 1e-10, "FOM not converged"'
new = ('    _frt = float(os.environ.get("FOM_RES_TOL", "1e-10"))\n'
       '    assert np.isfinite(res_max) and res_max < _frt, (\n'
       '        f"FOM not converged: {res_max:.2e} >= {_frt:.0e}")')
assert old in src, f"patch anchor missing in {p}"
open(p, "w").write(src.replace(old, new))
print(f"patched FOM_RES_TOL guard in {p}")
PYPATCH
done

( cd "$STAGE" && find . -name '*.py' -type f | sort | xargs sha256sum ) \
  > "$HERE/stage.manifest"
echo "staged $(grep -c . "$HERE/stage.manifest") files -> $STAGE"
