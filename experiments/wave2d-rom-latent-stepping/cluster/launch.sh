#!/usr/bin/env bash
# launch.sh <cell> : scp the staged dir DIRECTLY into paralab wlat/<cell>
# (never through login /tmp) -> verify checksums -> squeue before/after -> sbatch.
set -euo pipefail
cell="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/wlat/$cell"
[[ -d "$STAGE" ]] || { echo "no stage dir $STAGE" >&2; exit 1; }
before=$(ssh tufts-login "squeue -u tawal01 -h -o '%i %j' | grep -c ' $cell\$' || true")
if [[ "$before" != "0" ]]; then echo "job named $cell already queued ($before) -- abort" >&2; exit 2; fi
# the LOCAL stage must still match the manifest make_cell.sh wrote (a hand-edit
# between staging and launching would otherwise pass transport verification)
(cd "$STAGE" && sha256sum -c --quiet MANIFEST.sha256) || {
  echo "LOCAL STAGE does not match MANIFEST.sha256 for $cell -- restage" >&2; exit 4; }
# ONE JOB PER DIRECTORY: create the remote cell dir ATOMICALLY.  `mkdir -p` would
# happily overlay a finished run's code/out/logs; plain mkdir fails if it exists.
ssh tufts-login "mkdir -p /cluster/tufts/paralab/tawal01/wlat && mkdir $REMOTE" || {
  echo "remote cell dir $REMOTE already exists -- pull+delete it or pick a new cell name" >&2
  exit 5; }
scp -q -r "$STAGE"/. "tufts-login:$REMOTE/"
# and the STAGED BYTES must survive transport
ssh tufts-login "cd $REMOTE && sha256sum -c --quiet MANIFEST.sha256" || {
  echo "CHECKSUM MISMATCH after scp for $cell" >&2; exit 3; }
jid=$(ssh tufts-login "cd $REMOTE && sbatch --parsable run.sbatch")
after=$(ssh tufts-login "squeue -u tawal01 -h -o '%i %j' | grep -c ' $cell\$' || true")
echo "$cell job=$jid queued_with_name=$after"
[[ "$after" == "1" ]] || { echo "WARNING: $after jobs named $cell in queue" >&2; }
