"""Heartbeat writer (COMMON-BRIEF 9). usage: python3 hb.py "<step>" [--jobs a,b] [--codex out.md] [--codex-pid N]
[--local-pid N] [--eta ISO] [--notes ...]. GPU-hours are read from runs/GPU-LEDGER.json when present."""
import argparse, json, subprocess, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('step'); p.add_argument('--jobs', default=''); p.add_argument('--codex'); p.add_argument('--codex-pid', type=int)
p.add_argument('--local-pid', type=int); p.add_argument('--eta'); p.add_argument('--notes', default='')
a = p.parse_args()
led = HERE / 'runs' / 'GPU-LEDGER.json'
spent = json.loads(led.read_text()).get('spent_gpu_hours', 0.0) if led.exists() else 0.0
commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', '--short', 'HEAD'], text=True).strip()
hb = dict(lane='imp-faststep', utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), step=a.step,
          waiting_on=dict(slurm_jobs=[j for j in a.jobs.split(',') if j], codex_output=a.codex, codex_pid=a.codex_pid,
                          local_pid=a.local_pid), expected_finish_utc=a.eta, last_commit=commit,
          gpu_hours_spent=spent, gpu_hours_cap=4, notes=a.notes)
(HERE / 'HEARTBEAT.json').write_text(json.dumps(hb, indent=1) + '\n')
