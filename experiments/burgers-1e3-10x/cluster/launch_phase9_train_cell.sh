#!/usr/bin/env bash
# launch_phase9_train_cell.sh p9_t1_s11_r1 COMMIT MANIFEST_SHA
set -euo pipefail
cell="$1"; commit="$2"; expected_manifest="$3"
[[ "$cell" == p9_t1_s11_r1 && "$commit" =~ ^[0-9a-f]{40}$ && "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
[[ "$(sha256sum "$STAGE/MANIFEST.sha256"|cut -d' ' -f1)" == "$expected_manifest" ]] || exit 3
ssh tufts-login "test ! -e '$REMOTE' && ! squeue -h -u tawal01 -o '%j' | grep -Fx '$cell'"
ssh tufts-login "df -Pk /cluster/tufts/paralab/tawal01 | tail -1"
ssh tufts-login "mkdir -p '$REMOTE'"
scp -qr "$STAGE/." "tufts-login:$REMOTE/"
ssh tufts-login "cd '$REMOTE' && ./verify_manifest_file_set.sh . MANIFEST.sha256 && sha256sum -c MANIFEST.sha256"
ssh tufts-login "cd '$REMOTE' && B10_COMMIT='$commit' B10_CELL='$cell' B10_ARM=T1 B10_REMOTE='$REMOTE' sbatch --job-name='$cell' --export=ALL,B10_COMMIT='$commit',B10_CELL='$cell',B10_ARM=T1,B10_REMOTE='$REMOTE' run.sbatch"
ssh tufts-login "squeue -u tawal01 -o '%.18i %.24j %.10T %.10M %.20R'"
