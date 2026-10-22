#!/usr/bin/env bash
# usage: bash hb.sh "<step>" '<waiting_on json>' "<expected_finish>" "<notes>"
cd "$(dirname "$0")"
spent=$(python3 -c "import json;print(json.load(open('runs/GPU-LEDGER.json'))['total_gpu_hours'])" 2>/dev/null || echo 0.0)
cat > HEARTBEAT.json <<J
{"lane": "imp-box", "utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "step": "$1",
 "waiting_on": ${2:-{\}},
 "expected_finish_utc": "$3", "last_commit": "$(git rev-parse --short HEAD)",
 "gpu_hours_spent": $spent, "gpu_hours_cap": 87, "notes": "$4"}
J
