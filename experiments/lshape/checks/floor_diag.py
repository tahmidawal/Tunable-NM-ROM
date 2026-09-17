"""Is the bank-floor mismatch a driver error or a cancellation-limited round-off?

Decisive test: recompute the SAME floor by a SECOND, equally valid NumPy route (SVD of the
bank instead of QR).  Both routes are pure NumPy on the same machine, so any spread between
them is round-off alone.  If the NumPy-internal spread is the same size as the driver-vs-audit
difference, the driver is not wrong and the 1e-8 relative gate is mis-scaled for this quantity.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'experiments/lshape')
import lsh_audit_np as L

out = Path('experiments/lshape/runs/lsh02/archive/output')
d = json.loads((out / 'result.json').read_text())
cfg = d['config']
ntr = cfg['training_intervals']
g = L.Geom(ntr)
S = L.Solver(g)
dev = L.cohort(**{k: cfg['cohorts']['development'][k] for k in ('seed',)}, draw=cfg['cohorts']['development']['draw'], count=cfg['cohorts']['development']['count'])
com = L.cohort(seed=cfg['cohorts']['common']['seed'], draw=cfg['cohorts']['common']['draw'], count=cfg['cohorts']['common']['count'])
U = {'dev': np.stack([S.ref(L.source_int(g, q))[0] for q in dev]),
     'common': np.stack([S.ref(L.source_int(g, q))[0] for q in com])}

rows = []
for arm in d['bank_arms']:
    ck = next(x for x in d['checkpoints'] if x['id'] == arm['arm'])
    params, Z, c = L.load(out / ck['path'])
    G = L.features(params, g.coords, c['factor'], c['n_enrich'])
    Q, _ = np.linalg.qr(G)                      # route 1: QR (what the audit uses)
    Uq, sv, _ = np.linalg.svd(G, full_matrices=False)   # route 2: SVD
    rec = next(r for r in arm['floors'] if r['intervals'] == ntr)
    for name in ('dev', 'common'):
        X = U[name]
        f_qr = np.linalg.norm(X - (X @ Q) @ Q.T, axis=1) / np.linalg.norm(X, axis=1)
        f_svd = np.linalg.norm(X - (X @ Uq) @ Uq.T, axis=1) / np.linalg.norm(X, axis=1)
        f_drv = np.asarray(rec[f'floor_{name}']['per_case'])
        rows.append(dict(arm=arm['arm'], cohort=name,
                         floor_min=float(f_drv.min()),
                         drv_vs_qr=float(np.max(np.abs(f_qr - f_drv) / f_drv)),
                         qr_vs_svd=float(np.max(np.abs(f_qr - f_svd) / f_svd)),
                         drv_vs_svd=float(np.max(np.abs(f_svd - f_drv) / f_drv)),
                         # absolute, not relative
                         abs_drv_vs_qr=float(np.max(np.abs(f_qr - f_drv))),
                         abs_qr_vs_svd=float(np.max(np.abs(f_qr - f_svd))),
                         cond=float(sv[0] / sv[-1])))

print(f"{'arm':<22}{'coh':>7}{'min floor':>11}{'drv-qr rel':>12}{'qr-svd rel':>12}{'drv-svd rel':>13}{'drv-qr abs':>12}{'qr-svd abs':>12}")
for r in rows:
    print(f"{r['arm']:<22}{r['cohort']:>7}{r['floor_min']:11.3e}{r['drv_vs_qr']:12.3e}{r['qr_vs_svd']:12.3e}{r['drv_vs_svd']:13.3e}{r['abs_drv_vs_qr']:12.3e}{r['abs_qr_vs_svd']:12.3e}")
print()
print('worst driver-vs-QR  relative :', max(r['drv_vs_qr'] for r in rows))
print('worst QR-vs-SVD     relative :', max(r['qr_vs_svd'] for r in rows), '  <-- pure NumPy-internal spread')
print('worst driver-vs-QR  absolute :', max(r['abs_drv_vs_qr'] for r in rows))
print('worst QR-vs-SVD     absolute :', max(r['abs_qr_vs_svd'] for r in rows))
json.dump(rows, open('/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/6e5fe858-5d6b-4b34-a1bd-df699b1d4032/scratchpad/floor_diag.json','w'), indent=1)
