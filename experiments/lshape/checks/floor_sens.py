"""How far does a bank floor move under a round-off-sized perturbation of the bank itself?

The driver builds G on the GPU, the audit rebuilds it on the CPU; the two agree only to
relative ~1e-16 per entry.  This measures the resulting floor uncertainty PER ARM, entirely
inside NumPy, so the answer does not depend on either machine.  If the observed
driver-vs-audit difference sits inside this band, the floor is simply not determined to
better than that, and a fixed 1e-8 relative gate is testing round-off, not the science.
"""
import json, sys
import numpy as np
sys.path.insert(0, 'experiments/lshape')
import lsh_audit_np as L
from pathlib import Path

out = Path('experiments/lshape/runs/lsh02/archive/output')
d = json.loads((out / 'result.json').read_text())
cfg = d['config']; ntr = cfg['training_intervals']
g = L.Geom(ntr); S = L.Solver(g)
dc = cfg['cohorts']['development']
dev = L.cohort(seed=dc['seed'], draw=dc['draw'], count=dc['count'])
X = np.stack([S.ref(L.source_int(g, q))[0] for q in dev])
rng = np.random.default_rng(7)
prev = json.load(open('/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/6e5fe858-5d6b-4b34-a1bd-df699b1d4032/scratchpad/floor_diag.json'))
obs = {r['arm']: r['drv_vs_qr'] for r in prev if r['cohort'] == 'dev'}

def floor(G):
    Q, _ = np.linalg.qr(G)
    return np.linalg.norm(X - (X @ Q) @ Q.T, axis=1) / np.linalg.norm(X, axis=1)

print(f"{'arm':<22}{'cond(G)':>11}{'obs drv-audit':>15}{'eps-perturb band':>19}{'ratio':>8}")
rows=[]
for arm in d['bank_arms']:
    ck = next(x for x in d['checkpoints'] if x['id'] == arm['arm'])
    params, Z, c = L.load(out / ck['path'])
    G = L.features(params, g.coords, c['factor'], c['n_enrich'])
    f0 = floor(G)
    band = 0.0
    for _ in range(3):
        Gp = G * (1.0 + np.finfo(float).eps * rng.standard_normal(G.shape))
        band = max(band, float(np.max(np.abs(floor(Gp) - f0) / f0)))
    cond = float(np.linalg.cond(G))
    o = obs[arm['arm']]
    print(f"{arm['arm']:<22}{cond:11.3e}{o:15.3e}{band:19.3e}{o/band:8.2f}")
    rows.append(dict(arm=arm['arm'], cond=cond, observed=o, perturbation_band=band, ratio=o/band))
json.dump(rows, open('/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/6e5fe858-5d6b-4b34-a1bd-df699b1d4032/scratchpad/floor_sens.json','w'), indent=1)
