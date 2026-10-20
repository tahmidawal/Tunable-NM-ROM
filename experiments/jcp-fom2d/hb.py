"""Heartbeat writer: python hb.py '<step>' [key=value ...] (waiting_on fields: slurm_jobs=a,b codex_output=... codex_pid=N local_pid=N expected=ISO notes=...)."""
import datetime
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = HERE / 'HEARTBEAT.json'
h = json.loads(p.read_text()) if p.exists() else {}
kv = dict(a.split('=', 1) for a in sys.argv[2:])
w = dict(slurm_jobs=[x for x in kv.get('slurm_jobs', '').split(',') if x], codex_output=kv.get('codex_output'),
         codex_pid=int(kv['codex_pid']) if kv.get('codex_pid') else None, local_pid=int(kv['local_pid']) if kv.get('local_pid') else None)
try:
    led = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
    spent = led.get('spent_gpu_hours', 0.)
except Exception:  # noqa: BLE001
    spent = h.get('gpu_hours_spent', 0.)
h.update(lane='jcp-fom2d', utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), step=sys.argv[1],
         waiting_on=w, expected_finish_utc=kv.get('expected'),
         last_commit=subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, text=True).strip(),
         gpu_hours_spent=spent, gpu_hours_cap=30, notes=kv.get('notes', h.get('notes', '')))
p.write_text(json.dumps(h, indent=1) + '\n')
