"""Motivating evidence (DESIGN.md section 1): the space+time (ST) minus space-only (S) error gap of the 2D off-mesh
lane's backward-Euler rollouts, read from that lane's pulled job results (read-only), per mesh / setting / arm, on
dev6 u val32 (38 cases). Writes checks/evidence-be-gap.json.

    /home/tahmid/Dev/.venv/bin/python experiments/jcp-time2/evidence_be_gap.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / '2026-10-01-quadrature-study/experiments/quadrature-study/runs'
out = dict(source=str(SRC), note='percent of ||u0|| on the 257^2 shared nodes, worst over t=0.05..0.25', rows=[])
for L in (256, 1024, 4096):
    R = json.loads((SRC / f'dv{L}/archive/output/result.json').read_text())
    for s, arm in (('acc', 'gauss96'), ('acc', 'lat64'), ('acc', 'gref'), ('fast', 'fib1597'), ('fast', 'lat64'),
                   ('fast', 'gref')):
        rows = [r for r in R['rows'] if r['setting'] == s and r['arm'] == arm and 'ref_ST_evolved' in r]
        st = 100 * np.array([r['ref_ST_evolved'] for r in rows])
        S = 100 * np.array([r['ref_S_evolved'] for r in rows])
        out['rows'].append(dict(mesh=L, setting=s, arm=arm, cases=len(rows), job_id=R['job_id'],
                                ST_worst=st.max(), ST_median=float(np.median(st)), S_worst=S.max(),
                                S_median=float(np.median(S)), gap_median=float(np.median(st - S)),
                                gap_max=float((st - S).max()), cases_ST_gt_S=int(np.sum(st > S))))
# A1.1: the FOM's own BE time-error proxy at 8192^2, max_t ||S - ST|| / ||u0|| on the 257^2 nodes, per case
REF = SRC / 'refdv/archive/output'
d = []
for coh, n in (('dev6', 6), ('val32', 32)):
    for c in range(n):
        st = np.load(REF / f'ref_ST_{coh}_{c:03d}.npz')['f257']
        S = np.load(REF / f'ref_S_{coh}_{c:03d}.npz')['f257']
        n0 = np.linalg.norm(st[0])
        d.append(max(np.linalg.norm(st[k] - S[k]) / n0 for k in range(1, 6)))
d = 100 * np.array(d)
out['S_minus_ST'] = dict(cases=len(d), worst=float(d.max()), median=float(np.median(d)), per_case=d.tolist(),
                         note='percent of ||u0||, 257^2 nodes, max over t=0.05..0.25')
print('S-ST (FOM 8192^2) worst %.3f median %.3f pp' % (d.max(), np.median(d)))
(HERE / 'checks/evidence-be-gap.json').write_text(json.dumps(out, indent=1) + '\n')
for r in out['rows']:
    print('{mesh:5d} {setting:4s} {arm:8s} n={cases} ST {ST_worst:.3f}/{ST_median:.3f}  S {S_worst:.3f}/{S_median:.3f}  '
          'gap med {gap_median:.3f} max {gap_max:.3f}  ST>S {cases_ST_gt_S}'.format(**r))
