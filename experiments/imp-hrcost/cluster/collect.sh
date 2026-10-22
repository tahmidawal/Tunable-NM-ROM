#!/bin/bash
# collect.sh <attempt>: pull logs + outputs (files < 100 MB) into runs/<attempt>/, verify OUTPUTS.sha256, list larger files.
set -euo pipefail
A=$1; LANE=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/imp-hrcost
rsync -a --max-size=100m "tufts-login:$NS/$A/logs" "tufts-login:$NS/$A/output" "tufts-login:$NS/$A/OUTPUTS.sha256" "$LANE/runs/$A/"
timeout 120 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && find output -size +100M -printf '%s %p\n'" > "$LANE/runs/$A/LARGE.txt"
cd "$LANE/runs/$A" && grep -v -F -f <(awk '{print $2}' LARGE.txt) OUTPUTS.sha256 | sha256sum -c --quiet && echo verified
