#!/usr/bin/env bash
# Regression for the two deterministic files Slurm creates before run.sbatch starts.
# Usage: test_phase8_runtime_file_set.sh STAGED_CELL
set -euo pipefail
stage="$1"
[[ -d "$stage" && -f "$stage/MANIFEST.sha256" \
   && -x "$stage/verify_manifest_file_set.sh" ]] || exit 2

"$stage/verify_manifest_file_set.sh" "$stage" "$stage/MANIFEST.sha256"

fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT
cp -a "$stage/." "$fixture/"
job_id=9999999
: > "$fixture/logs/$job_id.out"
: > "$fixture/logs/$job_id.err"

"$fixture/verify_manifest_file_set.sh" "$fixture" "$fixture/MANIFEST.sha256" \
  "./logs/$job_id.out" "./logs/$job_id.err"
if "$fixture/verify_manifest_file_set.sh" "$fixture" "$fixture/MANIFEST.sha256" \
    >/dev/null 2>&1; then
  echo "runtime logs were not required as explicit extras" >&2
  exit 3
fi

: > "$fixture/out/unexpected-extra"
if "$fixture/verify_manifest_file_set.sh" "$fixture" "$fixture/MANIFEST.sha256" \
    "./logs/$job_id.out" "./logs/$job_id.err" >/dev/null 2>&1; then
  echo "unexpected third runtime file was accepted" >&2
  exit 4
fi

echo "phase8_runtime_file_set_regression=PASS exact_runtime_extras=2 unexpected_extra=rejected"
