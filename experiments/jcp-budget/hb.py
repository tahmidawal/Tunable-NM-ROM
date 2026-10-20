"""Heartbeat writer: python hb.py '<step>' [--codex out pid] [--jobs id,id] [--local pid] [--eta iso] [--notes ...]."""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('step')
ap.add_argument('--codex', nargs=2)
ap.add_argument('--jobs', default='')
ap.add_argument('--local')
ap.add_argument('--eta')
ap.add_argument('--notes', default='')
a = ap.parse_args()
led = HERE / 'runs' / 'GPU-LEDGER.json'
spent = 0.0
if led.exists():
    L = json.loads(led.read_text())
    spent = sum((j.get('elapsed_hours') or 0.) + (j.get('prior_elapsed_hours') or 0.) for j in L.get('jobs', []))
commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', '--short', 'HEAD'], text=True).strip()
hb = dict(lane='jcp-budget', utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), step=a.step,
          waiting_on=dict(slurm_jobs=[j for j in a.jobs.split(',') if j],
                          codex_output=a.codex[0] if a.codex else None,
                          codex_pid=int(a.codex[1]) if a.codex else None,
                          local_pid=int(a.local) if a.local else None),
          expected_finish_utc=a.eta, last_commit=commit, gpu_hours_spent=round(spent, 3), gpu_hours_cap=45,
          notes=a.notes or 'pre-release: design/code/smoke only')
(HERE / 'HEARTBEAT.json').write_text(json.dumps(hb, indent=1) + '\n')
print(json.dumps(hb))
