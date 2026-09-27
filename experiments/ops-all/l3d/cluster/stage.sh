#!/bin/bash
# stage.sh <job>: claim /cluster/tufts/paralab/tawal01/opsall_20260924/l3d/<job> atomically (never reused) and copy
# this group's code (python, configs, deps, expected) into <job>/code, with a sha256 manifest (SOURCE.sha256).
set -euo pipefail
JOB="$1"
NS=/cluster/tufts/paralab/tawal01/opsall_20260924/l3d
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$JOB" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad job name"; exit 1; }
ssh tufts-login "mkdir -p $NS && mkdir $NS/$JOB && mkdir $NS/$JOB/logs $NS/$JOB/output" || { echo "refusing: $NS/$JOB exists"; exit 1; }
rsync -a --exclude='runs/' --exclude='__pycache__/' --exclude='*.md' "$HERE/" "tufts-login:$NS/$JOB/code/"
ssh tufts-login "cd $NS/$JOB && find code -type f -print0 | sort -z | xargs -0 sha256sum > SOURCE.sha256 && wc -l SOURCE.sha256"
