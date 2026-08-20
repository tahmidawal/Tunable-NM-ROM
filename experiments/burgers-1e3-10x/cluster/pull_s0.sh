#!/usr/bin/env bash
# pull_s0.sh CELL JOB_ID EXPECTED_COMMIT EXPECTED_MANIFEST_FILE_SHA256
set -euo pipefail
cell="$1"
job_id="$2"
expected_commit="$3"
expected_manifest="$4"
[[ "$cell" =~ ^s0_spline_r[0-9]+$ ]] || { echo "invalid S0 cell" >&2; exit 2; }
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "invalid numeric job id" >&2; exit 2; }
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid commit" >&2; exit 2; }
[[ "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid manifest hash" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
LOCAL="$EXP/runs/$cell"
PY=/home/tahmid/Dev/.venv/bin/python
[[ ! -e "$LOCAL" ]] || { echo "local run already exists: $LOCAL" >&2; exit 3; }

ssh tufts-login "
  set -euo pipefail
  test -f '$REMOTE/logs/$job_id.out'
  test -f '$REMOTE/logs/$job_id.err'
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
actual_manifest="$(sha256sum "$LOCAL/MANIFEST.sha256" | cut -d' ' -f1)"
[[ "$actual_manifest" == "$expected_manifest" ]] || {
  echo "manifest-file hash mismatch: $actual_manifest" >&2; exit 4;
}
(cd "$LOCAL" && sha256sum -c REMOTE.sha256)

if [[ -s "$LOCAL/logs/$job_id.err" ]]; then
  if rg -v '^E[0-9]{4} [0-9:.]+ [0-9]+ numa_hwloc\.cc:121\] Call to hwloc_set_cpubind\(\) failed: Invalid argument \[22\]$' \
    "$LOCAL/logs/$job_id.err"; then
    echo "unclassified stderr found; remote directory retained" >&2
    exit 5
  fi
  echo "classified known non-numerical Tufts/JAX hwloc diagnostic: $(wc -l < "$LOCAL/logs/$job_id.err") line(s)" >&2
fi
if rg -n -i 'captured.large.constant|out of memory|(^|[^[:alpha:]])oom([^[:alpha:]]|$)|disk.full|no space|traceback|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
  "$LOCAL/logs/$job_id.out" "$LOCAL/logs/$job_id.err"; then
  echo "health-warning pattern found; remote directory retained" >&2
  exit 5
fi

"$PY" "$EXP/b10_audit_s0.py" \
  "$LOCAL/out/s0.json" "$LOCAL/out/s0.npz" "$LOCAL/out/AUDIT.json" \
  "$expected_commit" "$job_id"
(cd "$LOCAL" && find . -type f -not -name LOCAL.sha256 -exec sha256sum {} \; | sort > LOCAL.sha256)

# Delete only the explicitly validated completed S0 directory after all local
# checks and independent audit pass.  Then prove that exact directory is absent.
ssh tufts-login "
  set -euo pipefail
  test '$REMOTE' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell'
  rm -rf '$REMOTE'
  test ! -e '$REMOTE'
"
echo "$LOCAL"
