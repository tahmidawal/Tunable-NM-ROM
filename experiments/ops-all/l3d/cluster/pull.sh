#!/bin/bash
# pull.sh <job> [--delete]: copy logs + small outputs (json, logs; never *.pt / fields) into runs/<job>/, check
# jax_backend=gpu; --delete removes the remote job dir (only after the panel that needs its checkpoints ran).
set -euo pipefail
JOB="$1"; DEL="${2:-}"
NS=/cluster/tufts/paralab/tawal01/opsall_20260924/l3d
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/runs/$JOB"; mkdir -p "$DEST"
rsync -a --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' --exclude='code/' --exclude='cache/' --exclude='tmp/' "tufts-login:$NS/$JOB/" "$DEST/"
grep -q "jax_backend=gpu" "$DEST"/logs/*.out && echo "jax_backend=gpu OK" || { echo "NO jax_backend=gpu" >&2; exit 1; }
grep -h "EXIT\|ALL-DONE\|RUN_EXIT" "$DEST"/logs/*.out || true
if [[ "$DEL" == "--delete" ]]; then ssh tufts-login "rm -rf $NS/$JOB"; echo "remote $NS/$JOB deleted"; fi
