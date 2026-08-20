#!/usr/bin/env bash
# launch_spline_train_cell.sh CELL
set -euo pipefail
cell="$1"
[[ "$cell" =~ ^spline_[abc]_s(11|29|47)_r[0-9]+$ ]] || {
  echo "invalid spline trainer cell" >&2; exit 2;
}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
[[ -f "$STAGE/MANIFEST.sha256" && -f "$STAGE/run.sbatch" ]] || {
  echo "missing staged cell" >&2; exit 3;
}
(cd "$STAGE" && sha256sum -c MANIFEST.sha256)
expected_manifest="$(sha256sum "$STAGE/MANIFEST.sha256" | cut -d' ' -f1)"
echo "squeue_before"
ssh tufts-login "squeue -u tawal01 -o '%i|%j|%T|%M|%R'; df -h /cluster/tufts/paralab/tawal01"
ssh tufts-login "test ! -e '$REMOTE'; mkdir -p '$REMOTE'"
scp -q -r "$STAGE"/. "tufts-login:$REMOTE/"
job_id="$(ssh tufts-login "
  set -euo pipefail
  cd '$REMOTE'
  test \"\$(sha256sum MANIFEST.sha256 | cut -d' ' -f1)\" = '$expected_manifest'
  sha256sum -c MANIFEST.sha256
  echo remote_manifest_file_sha256=\$(sha256sum MANIFEST.sha256 | cut -d' ' -f1) >&2
  sbatch --parsable run.sbatch
")"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "unexpected sbatch result: $job_id" >&2; exit 4; }
echo "job_id=$job_id"
echo "manifest_file_sha256=$expected_manifest"
echo "squeue_after"
ssh tufts-login "squeue -j '$job_id' -o '%i|%j|%T|%M|%R'"
