#!/bin/bash
# Retain all artifacts, verify checksums, then optionally remove only the remote job duplicate.
set -euo pipefail
JOB="${1:?job tag required}"; ID="${2:?numeric job id required}"
[[ "$JOB" =~ ^[a-z0-9]+$ && "$ID" =~ ^[0-9]+$ ]] || exit 2
LANE=experiments/jcp-head-mechanism
NS=/cluster/tufts/paralab/tawal01/jcp-head-mechanism-20261118-488b0837
DEST="$NS/$JOB"
ROOT="$(git -c gc.auto=0 rev-parse --show-toplevel)"; cd "$ROOT"
[[ -z "$(ssh tufts-login "squeue -j $ID -h")" ]] || { echo 'Job still active';exit 2; }
mkdir -p "$LANE/runs/$JOB"
rsync -az --exclude='cache/' --exclude='tmp/' --exclude='__pycache__/' "tufts-login:$DEST/" "$LANE/runs/$JOB/"
(cd "$LANE/runs/$JOB" && sha256sum -c INPUTS.sha256 && sha256sum -c OUTPUTS.sha256)
ssh tufts-login "sacct -j $ID --units=G --format=JobID,JobName,State,ElapsedRaw,AllocTRES,NodeList,ExitCode -P" > "$LANE/runs/$JOB/sacct.txt"
# The caller inspects logs before passing --remove-verified-duplicate.
if [[ "${3:-}" == --remove-verified-duplicate ]]; then
  [[ -s "$LANE/runs/$JOB/output/summary.json" && -s "$LANE/runs/$JOB/OUTPUTS.sha256" ]] || exit 2
  ssh tufts-login "test \"\$(cat $NS/OWNER.txt)\" = 'head_mechanism_run 2026-11-18 exp/2026-11-18-jcp-head-mechanism' && test -d $DEST && rm -rf -- $DEST"
fi
