"""Heartbeat writer: python hb.py '<step>' [--codex <out> <pid>] [--jobs id,id] [--local <pid>] [--eta <utc>] [--notes '...']"""
import argparse, json, subprocess, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('step'); p.add_argument('--codex', nargs=2); p.add_argument('--jobs', default='')
p.add_argument('--local'); p.add_argument('--eta'); p.add_argument('--notes', default='')
a = p.parse_args()
led = HERE / 'runs' / 'GPU-LEDGER.json'
spent = json.loads(led.read_text()).get('spent_gpu_hours', 0.) if led.exists() else 0.
commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', '--short', 'HEAD'], text=True).strip()
hb = dict(lane='jcp-dims', utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), step=a.step,
          waiting_on=dict(slurm_jobs=[j for j in a.jobs.split(',') if j], codex_output=a.codex[0] if a.codex else None,
                          codex_pid=int(a.codex[1]) if a.codex else None, local_pid=int(a.local) if a.local else None),
          expected_finish_utc=a.eta, last_commit=commit, gpu_hours_spent=spent, gpu_hours_cap=55, notes=a.notes)
(HERE / 'HEARTBEAT.json').write_text(json.dumps(hb, indent=1) + '\n')
print(hb['utc'], a.step)
