#!/usr/bin/env bash
# Rebuttal push helper: submit one arm, then commit the ledger line (submit.sh needs a clean tree).
# Usage: cluster/sub.sh <name> <HH:MM:SS> [extra files]   -> attempt rb-<name>, task tasks/rb_<name>.sh
set -euo pipefail
cd "$(dirname "$0")/.."
n=$1; wall=$2; shift 2
git -c gc.auto=0 add -A . ; git -c gc.auto=0 diff --cached --quiet || git -c gc.auto=0 commit -qm "Rebuttal push: ledger/tasks before rb-$n"
bash cluster/submit.sh rb-$n $wall tasks/rb_$n.sh tasks/rb_common.sh tasks/manifest_production.json "$@" | tail -1
git -c gc.auto=0 add ledger/JOBS.jsonl; git -c gc.auto=0 commit -qm "Rebuttal push: ledger rb-$n"
