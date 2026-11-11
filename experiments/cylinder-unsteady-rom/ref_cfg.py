"""Second rebuttal push, PROTOCOL.md A9(a)/(c): configs for extra reference runs (vendored stepper, unchanged).

Copies an existing record config (mesh, orders, tolerances, integrator) and changes only the viscosity, the start
state and the stage list:
  settle  1/800, refresh 10, until the periodicity test passes or --duration (default 6.0 s); end state saved
  record  0.7 s at 1/3200 then the 1.7 s record at 1/3200 (refresh 40, saves every 1/400), as the existing records
  cont    continuation of an existing record from its saved end state: stage `record` of --duration (default 3.4 s)
Usage: ref_cfg.py --base CONFIG.json --Re R --initial STATE.npz --kind settle|record|cont --out CFG.json
"""
import argparse
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
for k in ('--base', '--initial', '--kind', '--out'):
    ap.add_argument(k, required=True)
ap.add_argument('--Re', type=float, required=True)
ap.add_argument('--duration', type=float); ap.add_argument('--threads', type=int, default=16)
a = ap.parse_args()
c = json.loads(Path(a.base).read_text())
c.update(nu=0.1 / a.Re, threads=a.threads, wall_limit_seconds=10 ** 6, kick=None,
         initial={'kind': 'state', 'path': a.initial, 'sha256': hashlib.sha256(Path(a.initial).read_bytes()).hexdigest()})
dt0 = 1 / 3200
if a.kind == 'settle':
    c['refactor_every'] = 10
    c['stages'] = [{'name': 'settle', 'dt': 1 / 800, 'duration': a.duration or 6.0, 'until_periodic': True,
                    'require_periodic': False, 'save_every_steps': 80, 'diagnostics_every_saves': 1, 'save_state': True}]
elif a.kind == 'record':
    c['refactor_every'] = 40
    c['stages'] = [{'name': 'settle_dt0', 'dt': dt0, 'duration': 0.7},
                   {'name': 'record', 'dt': dt0, 'duration': a.duration or 1.7, 'save_every_steps': 8,
                    'diagnostics_every_saves': 4, 'save_state': True}]
elif a.kind == 'cont':
    c['refactor_every'] = 40
    c['stages'] = [{'name': 'record', 'dt': dt0, 'duration': a.duration or 3.4, 'save_every_steps': 8,
                    'diagnostics_every_saves': 4, 'save_state': True}]
else:
    raise SystemExit('unknown kind')
Path(a.out).write_text(json.dumps(c, indent=2) + '\n')
print(json.dumps({k: c[k] for k in ('nu', 'refactor_every', 'threads', 'initial', 'stages')}))
