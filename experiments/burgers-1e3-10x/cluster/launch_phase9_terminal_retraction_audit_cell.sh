#!/usr/bin/env bash
# launch_phase9_terminal_retraction_audit_cell.sh p9_terminal_recovery_audit_r1 COMMIT MANIFEST_SHA256
set -euo pipefail
cell="$1"; commit="$2"; manifest="$3"
[[ "$cell" == p9_terminal_recovery_audit_r1 && "$commit" =~ ^[0-9a-f]{40}$ && "$manifest" =~ ^[0-9a-f]{64}$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; STAGE="$HERE/stage/$cell"; REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
[[ "$(sha256sum "$STAGE/MANIFEST.sha256"|cut -d' ' -f1)" == "$manifest" ]]
"$STAGE/verify_manifest_file_set.sh" "$STAGE" "$STAGE/MANIFEST.sha256"; (cd "$STAGE" && sha256sum -c MANIFEST.sha256)
ssh tufts-login "test ! -e '$REMOTE' && ! squeue -h -u tawal01 -o '%j' | grep -Fx '$cell'"
ssh tufts-login "df -Pk /cluster/tufts/paralab/tawal01 | tail -1; mkdir -p '$REMOTE'"
scp -qr "$STAGE/." "tufts-login:$REMOTE/"
ssh tufts-login "cd '$REMOTE' && test \"\$(sha256sum MANIFEST.sha256|cut -d' ' -f1)\" = '$manifest' && ./verify_manifest_file_set.sh . MANIFEST.sha256 && sha256sum -c MANIFEST.sha256"
job_id="$(ssh tufts-login "cd '$REMOTE' && sbatch --parsable --job-name='$cell' --export=ALL,B10_COMMIT='$commit',B10_CELL='$cell',B10_REMOTE='$REMOTE' run.sbatch")"
[[ "$job_id" =~ ^[0-9]+$ ]] || exit 4
echo "job_id=$job_id remote=$REMOTE manifest=$manifest"
ssh tufts-login "squeue -j '$job_id' -o '%i|%j|%T|%M|%R'"
