#!/usr/bin/env bash
# launch_phase3_d_cell.sh CELL EXPECTED_COMMIT
set -euo pipefail
cell="$1"; expected_commit="$2"
[[ "$cell" =~ ^p3_d_r[0-9]+$ ]] || { echo "invalid P3-D cell" >&2; exit 2; }
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid commit" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
[[ -f "$STAGE/MANIFEST.sha256" && -f "$STAGE/run.sbatch" ]] || {
  echo "missing staged P3-D cell" >&2; exit 3;
}
(cd "$STAGE" && sha256sum -c MANIFEST.sha256)
manifest="$(sha256sum "$STAGE/MANIFEST.sha256" | cut -d' ' -f1)"
echo squeue_before
ssh tufts-login "squeue -u tawal01 -o '%i|%j|%T|%M|%R'; df -h /cluster/tufts/paralab/tawal01"
ssh tufts-login "test ! -e '$REMOTE'; mkdir -p '$REMOTE'"
scp -q -r "$STAGE"/. "tufts-login:$REMOTE/"
job_id="$(ssh tufts-login "
  set -euo pipefail
  cd '$REMOTE'
  test \"\$(sha256sum MANIFEST.sha256 | cut -d' ' -f1)\" = '$manifest'
  sha256sum -c MANIFEST.sha256 >&2
  sbatch --parsable --job-name='ctol_b10_$cell' \
    --export=ALL,B10_COMMIT='$expected_commit',B10_CELL='$cell',B10_REMOTE='$REMOTE' \
    run.sbatch
")"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "unexpected sbatch result: $job_id" >&2; exit 4; }
echo "job_id=$job_id"
echo "manifest_file_sha256=$manifest"
echo squeue_after
ssh tufts-login "squeue -j '$job_id' -o '%i|%j|%T|%M|%R'"
