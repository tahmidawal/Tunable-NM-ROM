#!/usr/bin/env bash
# Collect a finished attempt: checksum every remote file, copy the compact evidence
# locally (everything except snapshot chunks and deps/wheels/cache), verify, and write a
# retention manifest. Snapshot chunks stay on the cluster (local disk is full) and are
# tracked by SHA-256 in the committed manifest. Nothing remote is deleted here.
# Usage: cluster/collect.sh <attempt-name>
set -euo pipefail
cd "$(dirname "$0")/.."
ATTEMPT=$1
DIR=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/$ATTEMPT
DEST=artifacts/$ATTEMPT
mkdir -p "$DEST"
ssh tufts-login "cd $DIR && find . -type f ! -path './deps/*' ! -path './wheels/*' ! -path './cache/*' ! -path './tmp/*' -print0 | sort -z | xargs -0 sha256sum" > "$DEST/REMOTE-SHA256SUMS"
ssh tufts-login "cd $DIR && find . -type f ! -path './deps/*' ! -path './wheels/*' ! -path './cache/*' ! -path './tmp/*' ! -name "snapshots_*.npz" ! -name "*.vol.bin" ! -name "basis.npz" ! -name "pod.npz" ! -name "trained_bank_eval.npz" -print0 | tar --null -T - -cf -" | tar -xf - -C "$DEST"
( cd "$DEST" && grep -v 'snapshots_.*\.npz$' REMOTE-SHA256SUMS | grep -v '\.vol\.bin$' | grep -v '/basis\.npz$' | grep -v '/pod\.npz$' | grep -v 'trained_bank_eval\.npz$' | sha256sum -c --quiet - ) && echo "local copies verified"
grep 'snapshots_.*\.npz$' "$DEST/REMOTE-SHA256SUMS" > "$DEST/REMOTE-ONLY-SNAPSHOTS.sha256" || true
echo "remote-only snapshot chunks: $(wc -l < "$DEST/REMOTE-ONLY-SNAPSHOTS.sha256")"
du -sh "$DEST"
