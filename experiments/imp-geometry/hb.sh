#!/bin/bash
# usage: hb.sh "<step>" '<waiting_on json>' "<expected_finish>" "<notes>"
D=$(cd "$(dirname "$0")" && pwd)
spent=$(/home/tahmid/Dev/.venv/bin/python -c "import json;print(json.load(open('$D/runs/GPU-LEDGER.json'))['total_gpu_hours'])" 2>/dev/null || echo 0.0)
cat > "$D/HEARTBEAT.json" <<JSON
{"lane": "imp-geometry", "utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "step": "$1",
 "waiting_on": ${2:-{\"slurm_jobs\": [], \"codex_output\": null, \"codex_pid\": null, \"local_pid\": null\}},
 "expected_finish_utc": "${3:-}", "last_commit": "$(git -C "$D" rev-parse --short HEAD)",
 "gpu_hours_spent": $spent, "gpu_hours_cap": 62, "notes": "${4:-}"}
JSON
