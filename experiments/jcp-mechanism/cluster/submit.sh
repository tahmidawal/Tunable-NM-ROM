#!/bin/bash
# jcp-mechanism: submit ONE staged job. Lane cap: 1 running-or-pending jm_* job (coordinator, 2026-10-08).
# Refuses if jm_<job> is queued, the cap is reached, or the remote dir exists. Copies runs/<job>/ and the references
# listed in REFS.json to /cluster/tufts/paralab/tawal01/jcpmech/<job>/; squeue before and after (duplicate-submit rule).
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcpmech
LOCAL="$ROOT/experiments/jcp-mechanism/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
echo "--- squeue before"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "jm_$A"; then echo "jm_$A already queued: refusing"; exit 3; fi
NL=$(echo "$Q" | awk '$2 ~ /^jm_/' | grep -c . || true)
[ "$NL" -lt 1 ] || { echo "lane cap (1) reached: refusing"; exit 4; }
ssh tufts-login "df -h /cluster/tufts/paralab | tail -1; test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
/home/tahmid/Dev/.venv/bin/python - "$LOCAL/REFS.json" "$NS/$A" <<'PY' | while read -r src dst; do ssh tufts-login "mkdir -p $(dirname "$dst")"; rsync -a "$src" "tufts-login:$dst"; done
import json, sys
for r in json.load(open(sys.argv[1])):
    print(r['source'], sys.argv[2] + '/' + r['dest'])
PY
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sha256sum -c REFS.sha256 --quiet && sbatch run.sbatch"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
