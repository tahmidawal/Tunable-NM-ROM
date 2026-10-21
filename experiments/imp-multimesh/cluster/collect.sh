#!/bin/bash
# Pull one finished attempt: logs, files < 100 MB under output/ (verified against the remote OUTPUTS.sha256), then
# archive the rest to burgers-archive/imp-multimesh/<attempt>/ and delete the job dir. Usage: collect.sh <attempt>
set -euo pipefail
A="$1"; LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/imp-multimesh; ARC=/cluster/tufts/paralab/tawal01/burgers-archive/imp-multimesh
mkdir -p "$LANE/runs/$A/pulled"
rsync -a --max-size=100m "tufts-login:$NS/$A/logs" "tufts-login:$NS/$A/OUTPUTS.sha256" "tufts-login:$NS/$A/output" "$LANE/runs/$A/pulled/"
( cd "$LANE/runs/$A/pulled" && grep -v -e '^$' OUTPUTS.sha256 | while read h f; do [ -f "$f" ] && echo "$h  $f"; done | sha256sum -c --quiet ) && echo "pulled files verified"
timeout 120 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && find output -type f -size +100M -print0 | xargs -0 -r sha256sum" > "$LANE/runs/$A/LARGE_FILES.sha256" || true
echo "large files: $(wc -l < "$LANE/runs/$A/LARGE_FILES.sha256")"
