#!/bin/bash
# imp-fronts DEV references FC1 dev j=0 and FC2 dev j=0 (Amendment 2 B3; run inside the SM-r smoke job, Amendment 11:
# SM-r ceiling 0.4 GPU-h total, H200). Development data only; never scored with the test split.
#
#   bash experiments/imp-fronts/cluster/refs/dev_refs_steps.sh <TASK_ROOT> <PY> <DEADLINE_EPOCH_S>
#
# Requirements on the host job (SM-r): the staged tree contains experiments/imp-fronts/phaseb/frefs/,
# experiments/imp-fronts/phaseb/fresh_cohorts.json, experiments/imp-fronts/cluster/refs/ and the read-only vendor files
# listed in cluster/refs/stage_refs.py EXTRA; the GPU preflight (jax_backend=gpu, exit 42) and
# JAX_DEFAULT_MATMUL_PRECISION=highest / JAX_ENABLE_X64 are already set; TASK_ROOT is under the imp-fronts namespace.
# DEADLINE_EPOCH_S = the host's self-timer cutoff (90 % of its wall, Amendment 4 B8); a step that cannot start 60 s
# before it is skipped, and a step still running at it is killed (its case has no .json -> no dev reference).
# Outputs: $TASK_ROOT/output/refs_d_FC1_0/, refs_d_FC2_0/ (full), $TASK_ROOT/output/refslim_dev/ (slim copies + MANIFEST),
# $TASK_ROOT/output/REF-LEDGER-dev.json (per-case outcome; needs $TASK_ROOT/logs/). Exit 0 iff refslim and ledger ok.
set -uo pipefail
TASK_ROOT=$1; PY=$2; DEADLINE=$3; RC=0
case "$TASK_ROOT" in /cluster/tufts/paralab/tawal01/imp-fronts/*) ;; *) echo "dev_refs: bad namespace $TASK_ROOT"; exit 2;; esac
[ "${JAX_DEFAULT_MATMUL_PRECISION:-}" = highest ] || { echo "dev_refs: precision not highest"; exit 3; }
cd "$TASK_ROOT/experiments/imp-fronts/phaseb/frefs"
for cfg in d_FC1_0.json d_FC2_0.json; do
  left=$(( DEADLINE - $(date +%s) ))
  if [ "$left" -le 60 ]; then echo "DEV_REFS DEADLINE: skipping $cfg"
    printf '%s\tskipped-deadline\t0\n' "$cfg" >> "$TASK_ROOT/logs/steps_dev.tsv"; continue; fi
  stem=$(basename "$cfg" .json)
  echo "DEV_REFS STEP $cfg left=${left}s $(date -u +%FT%TZ)"
  t0=$(date +%s); rc=0
  timeout --signal=TERM --kill-after=30 "$left" "$PY" refjob.py --config "../../cluster/refs/configs/$cfg" \
    --out "$TASK_ROOT/output/refs_$stem" || rc=$?
  echo "DEV_REFS STEP $cfg rc=$rc (124/137 = terminated at the cutoff)"
  printf '%s\t%s\t%s\n' "$cfg" "$rc" "$(( $(date +%s) - t0 ))" >> "$TASK_ROOT/logs/steps_dev.tsv"
done
"$PY" - "$TASK_ROOT/output" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
src = sorted(p.name for p in out.glob('refs_d_*') if p.is_dir())
(out / 'refslim_dev_config.json').write_text(json.dumps(dict(sources=src, allow_empty=True), indent=1))
EOF
JAX_PLATFORMS=cpu "$PY" refslim.py --config "$TASK_ROOT/output/refslim_dev_config.json" --out "$TASK_ROOT/output/refslim_dev" \
  || { echo "DEV_REFS refslim rc=$?"; RC=1; }
JAX_PLATFORMS=cpu "$PY" ../../cluster/refs/ref_ledger.py "$TASK_ROOT/output" "$TASK_ROOT/logs/steps_dev.tsv" --split dev-smoke \
  --configs ../../cluster/refs/configs/d_FC1_0.json ../../cluster/refs/configs/d_FC2_0.json \
  --out "$TASK_ROOT/output/REF-LEDGER-dev.json" || RC=1
echo "DEV_REFS done rc=$RC $(date -u +%FT%TZ)"   # host: treat rc != 0 as a failed step
exit $RC
