"""Independent establishment of the under-resolved-rule control fixture (DESIGN A1.1), before any lane job.

Fixture: R384 (M = 1536, frozen list = the 1024^2 builder list), LSPG-BE dt0, 1024^2, dev6; rule gauss32, successor
gauss64. Evidence (historical quadrature-study job dv1024, vendor BE query on the same configuration):
 (a) rollout bar: per dev6 case, d(g32, g64) >= d(g32, gref) - d(g64, gref) (triangle inequality on the same scoring
     lattice and normalisation), from the job's per-case vs_gref_restricted_evolved values;
 (b) rho bar: NumPy rho(gauss32 vs gauss64) on gauss32's OWN reached output states (historical audit trajectories,
     dev6|0 and dev6|2, internal[::10]), with Gauss-Legendre rules from numpy and the NumPy bank (npcheck).
Writes checks/ctrl-fixture-under-resolved.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import npcheck as NC

QS = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/experiments/quadrature-study/runs/dv1024/archive/output')


def modes_lean(L, M):
    k = np.arange(1, L)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    lam = 4 * L ** 2 * (np.sin(np.pi * kx / (2 * L)) ** 2 + np.sin(np.pi * ky / (2 * L)) ** 2)
    ind = np.argsort(lam.ravel(), kind='stable')[:M]
    return kx.ravel()[ind], ky.ravel()[ind]


def main():
    r = json.loads((QS / 'result.json').read_text())
    by = {(x['arm'], x['case']): x for x in r['rows'] if x['setting'] == 'acc' and x['cohort'] == 'dev6'}
    assert r['mesh'] == 1024
    roll = []
    for c in range(6):
        a, b = by[('gauss32', c)], by[('gauss64', c)]
        assert a['rule'] == 'gauss32' and b['rule'] == 'gauss64' and a['finite'] and b['finite']
        roll.append(dict(case=c, d32_gref=a['vs_gref_restricted_evolved'], d64_gref=b['vs_gref_restricted_evolved'],
                         lower_bound=a['vs_gref_restricted_evolved'] - b['vs_gref_restricted_evolved']))
    kx, ky = modes_lean(1024, 1536)
    X32, w32 = NC.gauss_rule(32)
    X64, w64 = NC.gauss_rule(64)
    t32, t64 = NC.tables(X32, w32, 384, kx, ky, 1024), NC.tables(X64, w64, 384, kx, ky, 1024)
    rh = {}
    for c in (0, 2):
        f = QS / f'audit_acc_gauss32_dev6{c}.npz'
        W = np.load(f)['internal'][::10]
        rh[f'dev6|{c}'] = dict(file=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                               rho=[NC.rho(W[k], t32, t64) for k in range(6)])
    res = dict(fixture=dict(Rp=384, M=1536, mesh=1024, scheme='LSPG-BE dt0', rule='gauss32', successor='gauss64',
                            cases='dev6', source_job=r.get('job_id'), source=str(QS / 'result.json'),
                            source_sha256=hashlib.sha256((QS / 'result.json').read_bytes()).hexdigest()),
               rollout=roll, rollout_min_lower_bound=min(x['lower_bound'] for x in roll),
               rollout_established=bool(min(x['lower_bound'] for x in roll) > 2.5e-4), rho_own=rh,
               rho_max=max(max(v['rho']) for v in rh.values()))
    res['rho_established'] = bool(res['rho_max'] > .116)
    Path('checks/ctrl-fixture-under-resolved.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ('rollout', 'rho_own')}, indent=1))
    print({k: [round(x, 4) for x in v['rho']] for k, v in rh.items()})


if __name__ == '__main__':
    main()
