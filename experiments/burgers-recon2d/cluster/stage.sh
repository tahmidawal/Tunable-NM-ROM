#!/usr/bin/env bash
# Stage directly to the paralab path; no submission. Parent reviews resulting exact hashes first.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 1 ] || [[ ! "$1" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then echo 'usage: stage.sh unique-job-id' >&2; exit 2; fi
remote_dir="/cluster/tufts/paralab/tawal01/b2d-recon-20261027/$1"
ssh tufts-login "test ! -e '$remote_dir' && mkdir -p '$remote_dir'"
# Frozen references/checkpoints are input evidence; no generated training data is staged.
rsync -a --exclude=runs --exclude=__pycache__ --exclude='*.pyc' ./ "tufts-login:$remote_dir/"
ssh tufts-login "cd '$remote_dir' && find . -type f -not -name STAGED.sha256 -print0 | sort -z | xargs -0 sha256sum > STAGED.sha256"
echo "$remote_dir"
