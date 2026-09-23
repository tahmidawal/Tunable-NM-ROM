"""Print the lab-log entry for this lane from reports/summary.json (numbers generated, not typed)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'reports' / 'summary.json').read_text())
SHA = hashlib.sha256((HERE / 'reports' / 'summary.json').read_bytes()).hexdigest()
commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', '--short', 'HEAD'], text=True).strip()


def f(x):
    return '—' if x is None else (f'{x:.3f}' if abs(x) >= 0.01 else f'{x:.2e}')


def sp(x):
    return '—' if x is None else f'{x:.3g}×'


def lab(name):
    if name.startswith('orig_q'):
        return f"R'=R q={name[6:]}"
    rp, rest = name[1:].split('_')
    return f"R'={rp} linear" if rest == 'linear' else f"R'={rp} q={rest[1:]}"


rows = [e for e in S['meshes'] if e['role'] in ('table', 'final') and e['usable']]
rows.sort(key=lambda e: (e['problem'] != 'lshape', e['intervals']))
L = []
w = L.append
w('### poisson-bank-knob-3d — nested bank truncation R\' on the L-shape and 3D Poisson models')
w('')
w(f"Branch `exp/2026-09-23-poisson-bank-knob-3d` (local only, not pushed), commit `{commit}`; "
  "worktree `worktrees/2026-09-23-poisson-bank-knob-3d`; namespace `/cluster/tufts/paralab/tawal01/pbank3_20260923/` "
  "(emptied). Design + amendment A1: `experiments/poisson-bank-knob-3d/DESIGN.md`; report: "
  "`experiments/poisson-bank-knob-3d/reports/2026-09-23-lshape-cube-bank-truncation-knob.md`; generated numbers: "
  f"`reports/summary.json` (sha256 `{SHA}`).")
w('')
w('**Rows under the pre-registered rule** (accurate = most accurate arm; fast = cheapest arm with worst error <= the '
  'paper fast setting q=0,R\'=R in the same job; speedup vs fastest tested CG at least as accurate as the accurate '
  'arm, same job; L-shape complete query, cube GPU query; worst same-grid error %):')
w('')
w('| problem | mesh | cohort | accurate | err % | S | fast | err % | S | FOM (err %) | job | GPU |')
w('|---|---|---|---|---:|---:|---|---:|---:|---|---|---|')
for e in rows:
    r = e['row']
    w(f"| {'L-shape' if e['problem'] == 'lshape' else 'cube'} | {e['intervals']} | {e['cohort']} ({e['cases']}) | "
      f"{lab(r['accurate'])} | {f(r['accurate_worst_pct'])} | {sp(r['accurate_speedup'])} | {lab(r['fast'])} | "
      f"{f(r['fast_worst_pct'])} | {sp(r['fast_speedup'])} | {r['fom']} ({f(r['fom_worst_pct'])}) | {e['job_id']} | "
      f"{e['gpu'].replace('NVIDIA ', '')} |")
w('')
print('\n'.join(L))
