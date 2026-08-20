#!/usr/bin/env bash
# launch_phase8_d_cell.sh p8_d_r2 EXPECTED_COMMIT EXPECTED_MANIFEST
set -euo pipefail
cell="$1"; expected_commit="$2"; expected_manifest="$3"
[[ "$cell" == p8_d_r2 && "$expected_commit" =~ ^[0-9a-f]{40}$ \
   && "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
[[ "$(sha256sum "$STAGE/MANIFEST.sha256" | cut -d' ' -f1)" == "$expected_manifest" ]] || exit 3
"$STAGE/verify_manifest_file_set.sh" "$STAGE" "$STAGE/MANIFEST.sha256"
(cd "$STAGE" && sha256sum -c MANIFEST.sha256)
echo squeue_before_submit
ssh tufts-login "squeue -u tawal01 -o '%i|%j|%T|%M|%R'"
ssh tufts-login "set -euo pipefail; df -h /cluster/tufts/paralab/tawal01; test ! -e '$REMOTE'; mkdir -p '$REMOTE'"
scp -q -r "$STAGE/"* "tufts-login:$REMOTE/"
ssh tufts-login "set -euo pipefail; cd '$REMOTE'; test \"\$(sha256sum MANIFEST.sha256 | cut -d' ' -f1)\" = '$expected_manifest'; ./verify_manifest_file_set.sh . MANIFEST.sha256; sha256sum -c MANIFEST.sha256"
job_id="$(ssh tufts-login "cd '$REMOTE' && sbatch --parsable --job-name=ctol_b10_$cell --export=ALL,B10_COMMIT=$expected_commit,B10_CELL=$cell,B10_REMOTE=$REMOTE run.sbatch")"
[[ "$job_id" =~ ^[0-9]+$ ]] || exit 4
echo "job_id=$job_id remote=$REMOTE manifest=$expected_manifest"
echo squeue_after_submit
ssh tufts-login "squeue -j '$job_id' -o '%i|%j|%T|%M|%R'"
