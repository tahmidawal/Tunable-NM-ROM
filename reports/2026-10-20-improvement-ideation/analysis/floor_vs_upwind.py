"""Does the bank's worst-case floor track the training generator's numerical diffusion? (CPU, JSON only.)

The 2D bank was trained on first-order sign-upwind + BE data on 256 nodes per axis (jcp-smooth-bank DESIGN.md: 56-60).
First-order upwinding adds numerical viscosity of about a*h/2 (h = 1/255: 256 nodes per axis including the boundary; a = IC amplitude, a bound on |u| here).
For each of the 38 dev6+val32 cases: per-case bank floors (analysis/bank_vs_sine.json) against the ratio
q = (a h / 2) / nu and against the nominal steepness s = a / nu (cohort physical descriptors cx, cy, w, a, nu from
jcp-combined runs/local2dfb/result.json). Also reads the upwind-256 FOM's own E0 error (jcp-fom2d checks/match-final.json,
pareto.UB, key fd|UB|L256|0.005|t2). Writes analysis/floor_vs_upwind.json.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
HERE = Path(__file__).parent
bvs = json.loads((HERE / 'bank_vs_sine.json').read_text())
coh = json.loads((ROOT / 'worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/local2dfb/result.json').read_text())['cohorts']
phys = {f'ref_ST_{c}_{i:03d}': p for c in ('dev6', 'val32') for i, p in enumerate(coh[c]['physical'])}
mf = json.loads((ROOT / 'worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/checks/match-final.json').read_text())
ub = [p for p in mf['pareto']['UB']['points'] if p['key'] == 'fd|UB|L256|0.005|t2'][0]
h = 1 / 255   # 256 nodes per axis including the boundary (jcp-smooth-bank DESIGN.md training-data row)
rows = []
for r_ in bvs['rows']:
    cx, cy, w, a, nu = phys[r_['case']]
    rows.append(dict(case=r_['case'], a=a, nu=nu, s=a / nu, q=a * h / 2 / nu, floor384=r_['bank_384'], floor128=r_['bank_128']))
q = np.array([x['q'] for x in rows]); s = np.array([x['s'] for x in rows]); f = np.array([x['floor384'] for x in rows])
top = sorted(rows, key=lambda x: -x['floor384'])[:5]
out = dict(upwind256_vs_E0=dict(key=ub['key'], median=ub['median'], worst=ub['worst']),
           spearman_floor384_vs_q=float(spearmanr(f, q)[0]), spearman_floor384_vs_s=float(spearmanr(f, s)[0]),
           q_range=[float(q.min()), float(q.max())], worst5_by_floor384=top,
           median_q_worst5=float(np.median([x['q'] for x in top])), median_q_all=float(np.median(q)))
(HERE / 'floor_vs_upwind.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'worst5_by_floor384'}, indent=1)); [print(x) for x in top]
