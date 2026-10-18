"""Heartbeat writer (copied from jcp-p1-scaling hb.py, lane name and cap edited):
python hb.py '<step>' [--codex <out> <pid>] [--jobs id,id] [--local <pid>] [--eta <utc>] [--notes ..]"""
import argparse, json, subprocess, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('step'); ap.add_argument('--codex', nargs=2); ap.add_argument('--jobs', default='')
ap.add_argument('--local'); ap.add_argument('--eta'); ap.add_argument('--notes', default='')
a = ap.parse_args()
led = HERE / 'runs' / 'GPU-LEDGER.json'
spent = json.loads(led.read_text()).get('spent_gpu_hours', 0.) if led.exists() else 0.
hb = {'lane': 'jcp-qualify', 'utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
      'step': a.step,
      'waiting_on': {'slurm_jobs': [j for j in a.jobs.split(',') if j],
                     'codex_output': a.codex[0] if a.codex else None, 'codex_pid': int(a.codex[1]) if a.codex else None,
                     'local_pid': int(a.local) if a.local else None},
      'expected_finish_utc': a.eta,
      'last_commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, text=True).strip(),
      'gpu_hours_spent': spent, 'gpu_hours_cap': 35, 'notes': a.notes}
(HERE / 'HEARTBEAT.json').write_text(json.dumps(hb, indent=1) + '\n')
print(json.dumps(hb))
