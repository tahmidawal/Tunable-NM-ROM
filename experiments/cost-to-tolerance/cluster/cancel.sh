#!/usr/bin/env bash
# cancel.sh <jobid> [jobid ...]
#
# The ONLY sanctioned way to cancel a job from this cell.  Never run `scancel`
# by hand here.
#
# WHY THIS FILE EXISTS (2026-08-17 incident).  An ad-hoc
#     scancel --name=ctol_p_n32,ctol_p_n64,...
# was issued from this cell.  `scancel --name` takes ONE job name, not a
# comma-separated list, so that value matched nothing -- which left scancel with
# no effective selector, and a scancel with no effective filter selects EVERY job
# belonging to the invoking user.  The tawal01 account is shared with another
# agent, and all eleven of its jobs were killed at 00:00:00 elapsed alongside
# nothing of ours.  Sixteen minutes of our compute was lost; their queue position
# was lost too.  At hour three of a multi-hour Burgers panel the same slip costs a
# working day.
#
# The rules this script enforces so they cannot be forgotten:
#   1. Job IDs only.  No --name, no --user, no globs, no "all my jobs".
#   2. Every ID must currently be queued under this account AND be named ctol_*.
#      Anything else -- another agent's job, an unknown ID, a typo -- aborts the
#      WHOLE call before a single job is touched.
#   3. squeue is re-checked afterwards and the result printed.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: cancel.sh <jobid> [jobid ...]   (numeric IDs only)" >&2; exit 1; }

for id in "$@"; do
  [[ "$id" =~ ^[0-9]+$ ]] || {
    echo "REFUSING: '$id' is not a numeric job id.  This script never takes names, "\
         "globs or user selectors -- that is exactly how the shared account got "\
         "wiped once already." >&2; exit 2; }
done

echo "current queue for this account:"
ssh tufts-login "squeue -u tawal01 -o '%.10i %.20j %.9T %.10M'"

owned=$(ssh tufts-login "squeue -u tawal01 -h -o '%i %j'")
bad=0
for id in "$@"; do
  name=$(awk -v i="$id" '$1==i {print $2}' <<<"$owned")
  if [[ -z "$name" ]]; then
    echo "REFUSING: job $id is not in this account's queue" >&2; bad=1
  elif [[ "$name" != ctol_* ]]; then
    echo "REFUSING: job $id is named '$name', which is NOT one of this cell's ctol_* jobs" >&2
    bad=1
  else
    echo "  ok: $id = $name"
  fi
done
[[ "$bad" -eq 0 ]] || { echo "aborting; nothing was cancelled" >&2; exit 3; }

ssh tufts-login "scancel $*"
sleep 3
echo "queue after:"
ssh tufts-login "squeue -u tawal01 -o '%.10i %.20j %.9T %.10M'"
