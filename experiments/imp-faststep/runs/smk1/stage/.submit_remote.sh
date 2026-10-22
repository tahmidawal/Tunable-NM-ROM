#!/bin/bash
# Runs on the login node under the campaign sbatch lock: bash .submit_remote.sh <upload dir> <job dir>
set -o pipefail
TMP=$1; REMOTE=$2
test ! -e "$REMOTE" || exit 4
q=$(squeue -u tawal01 -h -o '%j') || exit 70
n=$(printf '%s\n' "$q" | grep -c .)
[ "$n" -lt 8 ] || exit 75
l=$(printf '%s\n' "$q" | grep -c '^ifs_')
[ "$l" -lt 2 ] || exit 76
mv "$TMP" "$REMOTE" || exit 71
cd "$REMOTE" || exit 73
jid=$(sbatch --parsable run.sbatch) || exit 72          # ambiguous: directory and intent are kept for reconciliation
jid=${jid%%;*}
[[ "$jid" =~ ^[0-9]+$ ]] || exit 74
{ echo "$jid" > JOBID.tmp && mv JOBID.tmp JOBID; } || exit 74
echo "Submitted batch job $jid"
