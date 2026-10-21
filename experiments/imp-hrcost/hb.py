"""Heartbeat writer (COMMON-BRIEF 9): python hb.py "<step>" [--codex FILE PID] [--slurm ID ...] [--local PID] [--eta UTC] [--notes TXT]"""
import argparse, datetime, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('step'); p.add_argument('--codex', nargs=2); p.add_argument('--slurm', nargs='*', default=[])
p.add_argument('--local', type=int); p.add_argument('--eta'); p.add_argument('--notes', default='')
p.add_argument('--handoff', action='store_true')
a = p.parse_args()
led = HERE / 'runs' / 'GPU-LEDGER.json'
spent = json.loads(led.read_text()).get('total_gpu_hours', 0.) if led.exists() else 0.
commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', '--short', 'HEAD'], text=True).strip()
w = dict(slurm_jobs=a.slurm, codex_output=a.codex[0] if a.codex else None, codex_pid=int(a.codex[1]) if a.codex else None,
         local_pid=a.local)
if a.handoff:
    w['handoff'] = 'handoff: imp-faststep HANDOFF-E6.md'
hb = dict(lane='imp-hrcost', utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), step=a.step,
          waiting_on=w, expected_finish_utc=a.eta, last_commit=commit, gpu_hours_spent=spent, gpu_hours_cap=3,
          notes=a.notes)
(HERE / 'HEARTBEAT.json').write_text(json.dumps(hb, indent=1))
