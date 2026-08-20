#!/usr/bin/env bash
# pull_spline_train.sh CELL JOB_ID EXPECTED_COMMIT EXPECTED_MANIFEST_SHA256 ARM SEED S0_CELL [PRIOR_CELL ...]
set -euo pipefail
cell="$1"; job_id="$2"; expected_commit="$3"; expected_manifest="$4"
arm="$5"; seed="$6"; s0_cell="$7"
shift 7
prior_cells=("$@")
[[ "$cell" =~ ^spline_[abc]_s(11|29|47)_r[0-9]+$ ]] || { echo "invalid cell" >&2; exit 2; }
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "invalid job id" >&2; exit 2; }
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid commit" >&2; exit 2; }
[[ "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid manifest hash" >&2; exit 2; }
[[ "$arm" =~ ^[ABC]$ && "$seed" =~ ^(11|29|47)$ ]] || { echo "invalid arm/seed" >&2; exit 2; }
[[ "$s0_cell" =~ ^s0_spline_r[0-9]+$ ]] || { echo "invalid S0 cell" >&2; exit 2; }
for prior in "${prior_cells[@]}"; do
  [[ "$prior" =~ ^spline_[abc]_s(11|29|47)_r[0-9]+$ ]] || {
    echo "invalid prior cell" >&2; exit 2;
  }
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
LOCAL="$EXP/runs/$cell"
S0_RUN="$EXP/runs/$s0_cell"
PY=/home/tahmid/Dev/.venv/bin/python
[[ ! -e "$LOCAL" ]] || { echo "local run already exists" >&2; exit 3; }
[[ -f "$S0_RUN/LOCAL.sha256" ]] || { echo "missing local S0 run" >&2; exit 3; }
(cd "$S0_RUN" && sha256sum -c LOCAL.sha256)
prior_jsons=()
for prior in "${prior_cells[@]}"; do
  run="$EXP/runs/$prior"
  [[ -f "$run/LOCAL.sha256" ]] || { echo "missing local prior: $prior" >&2; exit 3; }
  (cd "$run" && sha256sum -c LOCAL.sha256)
  prior_jsons+=("$run/out/train.json")
done

echo "squeue_before_pull"
ssh tufts-login "squeue -j '$job_id' -o '%i|%j|%T|%M|%R' || true"
ssh tufts-login "
  set -euo pipefail
  test -f '$REMOTE/logs/$job_id.out'
  test -f '$REMOTE/logs/$job_id.err'
  test -f '$REMOTE/out/train.json'
  test -f '$REMOTE/out/train.npz'
  test -f '$REMOTE/out/checkpoint.pkl'
  grep -q '^ALL-DONE$' '$REMOTE/logs/$job_id.out'
  grep -q 'jax_backend=gpu' '$REMOTE/logs/$job_id.out'
  grep -q 'commit=$expected_commit ' '$REMOTE/logs/$job_id.out'
  cd '$REMOTE'
  sha256sum -c MANIFEST.sha256
  sacct -j '$job_id' --format=JobIDRaw,JobName%32,State,Elapsed,ExitCode,NodeList%24,AllocTRES%80 > SACCT.txt
  awk -v id='$job_id' '\$1==id {seen=1; if (\$2 != \"ctol_b10_$cell\" || \$3 != \"COMPLETED\" || \$5 != \"0:0\") exit 2} END {exit(seen?0:3)}' SACCT.txt
  find logs out -type f -exec sha256sum {} \; | sort > REMOTE.sha256
  sha256sum SACCT.txt MANIFEST.sha256 >> REMOTE.sha256
"
mkdir -p "$LOCAL"
scp -q -r "tufts-login:$REMOTE/logs" "tufts-login:$REMOTE/out" \
  "tufts-login:$REMOTE/MANIFEST.sha256" "tufts-login:$REMOTE/REMOTE.sha256" \
  "tufts-login:$REMOTE/SACCT.txt" "$LOCAL/"
(cd "$LOCAL" && sha256sum -c REMOTE.sha256)
actual_manifest="$(sha256sum "$LOCAL/MANIFEST.sha256" | cut -d' ' -f1)"
[[ "$actual_manifest" == "$expected_manifest" ]] || {
  echo "manifest-file hash mismatch: $actual_manifest" >&2; exit 5;
}
if [[ -s "$LOCAL/logs/$job_id.err" ]]; then
  if rg -v '^E[0-9]{4} [0-9:.]+ [0-9]+ numa_hwloc\.cc:121\] Call to hwloc_set_cpubind\(\) failed: Invalid argument \[22\]$' \
    "$LOCAL/logs/$job_id.err"; then
    echo "unclassified stderr; remote retained" >&2; exit 5
  fi
fi
health_pattern='captured[^[:cntrl:]]{0,80}large[^[:cntrl:]]{0,80}constant|large[^[:cntrl:]]{0,80}constant[^[:cntrl:]]{0,80}captur|out of memory|(^|[^[:alpha:]])oom([^[:alpha:]]|$)|disk.full|no space|traceback|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)'
if rg -n -i "$health_pattern" "$LOCAL/logs/$job_id.out" "$LOCAL/logs/$job_id.err"; then
  echo "health-warning pattern; remote retained" >&2; exit 5
fi

"$PY" "$EXP/b10_audit_spline_train.py" \
  "$LOCAL/out/train.json" "$LOCAL/out/train.npz" "$LOCAL/out/checkpoint.pkl" \
  "$LOCAL/out/AUDIT.json" "$expected_commit" "$job_id" "$arm" "$seed" \
  "$S0_RUN/out/s0.json" "$S0_RUN/out/AUDIT.json" "$LOCAL/MANIFEST.sha256" \
  "$expected_manifest" \
  "${prior_jsons[@]}"
(cd "$LOCAL" && find . -type f -not -name LOCAL.sha256 -exec sha256sum {} \; | sort > LOCAL.sha256)
ssh tufts-login "
  set -euo pipefail
  test '$REMOTE' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell'
  rm -rf '$REMOTE'
  test ! -e '$REMOTE'
"
echo "squeue_after_cleanup"
ssh tufts-login "squeue -j '$job_id' -o '%i|%j|%T|%M|%R' || true"
echo "$LOCAL"
