# Dataset preparation (NGSolve, CPU work inside the GPU allocation): smoke on 3 saves, then all cases, then read-out calibration.
set -euo pipefail
"$PY" install_deps.py --wheel-dir "$JOB_DIR/wheels" --target "$JOB_DIR/deps"
DEPS_SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_path('purelib', vars={'base':'$JOB_DIR/deps','platbase':'$JOB_DIR/deps'}))")
export PYTHONPATH="$DEPS_SITE:$JOB_DIR/vendor:$PYTHONPATH"
export LD_LIBRARY_PATH="$JOB_DIR/deps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PY" -c "import ngsolve; print('ngsolve', ngsolve.__version__)"
"$PY" prep.py --manifest inputs/manifest_production.json --out results/smoke/data --limit 3
"$PY" prep.py --manifest inputs/manifest_production.json --out results/data
"$PY" ops_common.py results/data re95 results/data/readout_re95.json
du -sh results/data results/sealed
