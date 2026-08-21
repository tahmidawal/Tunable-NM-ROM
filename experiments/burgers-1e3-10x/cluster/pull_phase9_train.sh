#!/usr/bin/env bash
# pull_phase9_train.sh p9_t1_s11_r1 JOB COMMIT
set -euo pipefail
cell="$1"; job="$2"; commit="$3"
[[ "$cell" =~ ^p9_t[12]_s11_r1$ && "$job" =~ ^[0-9]+$ && "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP="$(dirname "$HERE")"
REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"; LOCAL="$EXP/runs/$cell"
[[ ! -e "$LOCAL" ]] || exit 3
state="$(ssh tufts-login "sacct -j '$job' -X -n -o State -P | head -1 | cut -d'|' -f1")"
[[ "$state" == COMPLETED ]] || exit 4
mkdir -p "$LOCAL"
ssh tufts-login "cd '$REMOTE' && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256 && sacct -j '$job' -X -o JobID,JobName,State,ExitCode,Elapsed,AllocTRES,NodeList -P > SACCT.txt"
scp -qr "tufts-login:$REMOTE/out" "tufts-login:$REMOTE/logs" "tufts-login:$REMOTE/MANIFEST.sha256" "tufts-login:$REMOTE/REMOTE.sha256" "tufts-login:$REMOTE/SACCT.txt" "$LOCAL/"
(cd "$LOCAL" && sha256sum -c REMOTE.sha256)
[[ "$(jq -r .status "$LOCAL/out/AUDIT.json")" == pass ]]
(cd "$LOCAL" && find out logs MANIFEST.sha256 REMOTE.sha256 SACCT.txt -type f -exec sha256sum {} \; | sort > LOCAL.sha256 && sha256sum -c LOCAL.sha256)
ssh tufts-login "test '$REMOTE' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell' && rm -rf -- '$REMOTE' && test ! -e '$REMOTE'"
echo "pulled=$LOCAL commit=$commit job=$job remote_removed=true"
