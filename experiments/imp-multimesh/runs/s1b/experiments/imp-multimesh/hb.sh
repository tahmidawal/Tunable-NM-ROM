#!/usr/bin/env bash
# hb.sh "<step>" "<notes>" [slurm_jobs_csv] [codex_output] [codex_pid] [local_pid] [expected_finish] -- rewrite HEARTBEAT.json
L=$(cd "$(dirname "$0")" && pwd)
spent=$(python3 -c "import json;print(json.load(open('$L/runs/GPU-LEDGER.json'))['spent_gpu_hours'])" 2>/dev/null || echo 0.0)
jobs=${3:-}; jl=$(python3 -c "import sys,json;print(json.dumps([j for j in sys.argv[1].split(',') if j]))" "$jobs")
cat > "$L/HEARTBEAT.json" <<J
{"lane": "imp-multimesh", "utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "step": "$1",
 "waiting_on": {"slurm_jobs": $jl, "codex_output": "${4:-}", "codex_pid": ${5:-null}, "local_pid": ${6:-null}},
 "expected_finish_utc": "${7:-}", "last_commit": "$(git -C "$L" rev-parse --short HEAD)",
 "gpu_hours_spent": $spent, "gpu_hours_cap": 8, "notes": "$2"}
J
