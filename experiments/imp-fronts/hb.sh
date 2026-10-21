#!/bin/bash
# usage: hb.sh "<step>" "<notes>" [codex_output] [codex_pid] [local_pid] [expected_finish]
cd "$(dirname "$0")"
LC=$(git rev-parse --short HEAD)
SPENT=$(python3 -c "import json;print(json.load(open('runs/GPU-LEDGER.json'))['cumulative_gpu_hours'])" 2>/dev/null || echo 0.0)
python3 - "$@" <<PY
import json,sys,datetime
a=sys.argv[1:]+[None]*6
d={"lane":"imp-fronts","utc":datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),"step":a[0],
"waiting_on":{"slurm_jobs":[],"codex_output":a[2],"codex_pid":int(a[3]) if a[3] else None,"local_pid":int(a[4]) if a[4] else None},
"expected_finish_utc":a[5],"last_commit":"$LC","gpu_hours_spent":float("$SPENT"),"gpu_hours_cap":15,"notes":a[1] or ""}
json.dump(d,open("HEARTBEAT.json","w"),indent=1)
PY
