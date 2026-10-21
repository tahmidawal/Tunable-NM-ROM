#!/bin/bash
# usage: hb.sh "<step>" "<slurm ids comma>" "<codex_output>" "<codex_pid>" "<expected>" "<notes>" [spent]
L=/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef
c=$(git -C $L log -1 --format=%h)
spent=${7:-$(/home/tahmid/Dev/.venv/bin/python -c "import json;print(json.load(open('$L/runs/GPU-LEDGER.json'))['total_gpu_hours'])" 2>/dev/null || echo 0.0)}
ids=$(echo "$2" | /home/tahmid/Dev/.venv/bin/python -c "import sys,json;s=sys.stdin.read().strip();print(json.dumps([x for x in s.split(',') if x]))")
/home/tahmid/Dev/.venv/bin/python - "$1" "$ids" "$3" "$4" "$5" "$6" "$c" "$spent" <<'PY'
import sys,json,datetime
a=sys.argv
d={"lane":"imp-varcoef","utc":datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),"step":a[1],
"waiting_on":{"slurm_jobs":json.loads(a[2]),"codex_output":a[3] or None,"codex_pid":int(a[4]) if a[4] else None,"local_pid":None},
"expected_finish_utc":a[5] or None,"last_commit":a[7],"gpu_hours_spent":float(a[8]),"gpu_hours_cap":15,"notes":a[6]}
open('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/HEARTBEAT.json','w').write(json.dumps(d,indent=1))
PY
