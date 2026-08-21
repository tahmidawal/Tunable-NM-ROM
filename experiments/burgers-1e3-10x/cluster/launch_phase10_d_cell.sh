#!/usr/bin/env bash
set -euo pipefail
cell="$1"; commit="$2"; manifest="$3"
[[ "$cell" == p10_d_r2 && "$commit" =~ ^[0-9a-f]{40}$ && "$manifest" =~ ^[0-9a-f]{64}$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; STAGE="$HERE/stage/$cell"; REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
[[ "$(sha256sum "$STAGE/MANIFEST.sha256"|cut -d' ' -f1)" == "$manifest" ]]; "$STAGE/verify_manifest_file_set.sh" "$STAGE" "$STAGE/MANIFEST.sha256"; (cd "$STAGE" && sha256sum -c MANIFEST.sha256)
ssh tufts-login "test ! -e '$REMOTE' && ! squeue -h -u tawal01 -o '%j' | grep -Fx '$cell'; df -Pk /cluster/tufts/paralab/tawal01 | tail -1; mkdir -p '$REMOTE'"
scp -qr "$STAGE/." "tufts-login:$REMOTE/"
ssh tufts-login "cd '$REMOTE' && test \"\$(sha256sum MANIFEST.sha256|cut -d' ' -f1)\" = '$manifest' && ./verify_manifest_file_set.sh . MANIFEST.sha256 && sha256sum -c MANIFEST.sha256"
job="$(ssh tufts-login "cd '$REMOTE' && sbatch --parsable --job-name='$cell' --export=ALL,B10_COMMIT='$commit',B10_CELL='$cell',B10_REMOTE='$REMOTE' run.sbatch")"
[[ "$job" =~ ^[0-9]+$ ]] || exit 4; echo "job_id=$job remote=$REMOTE manifest=$manifest"; ssh tufts-login "squeue -j '$job' -o '%i|%j|%T|%M|%R'"
