"""Write the job configs (DESIGN §4, §7, A2.5) and the 2D rule manifest. python make_configs.py [--manifest2d]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STORE_VERSION = None   # filled from the vendored manifest below

GAUSS2 = [24, 32, 36, 40, 45, 50, 55, 60, 64, 72, 80, 96, 128, 256]
FIB2 = [610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368]
SOB2 = [1024, 2048, 4096, 8192, 16384, 32768]
CBC3 = [256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 6144, 8192, 16384, 32768]
GL3 = [8, 10, 12, 14, 16, 17, 18, 19, 20, 22, 24, 28, 32]
SOB3 = [1024, 2048, 4096, 8192, 16384, 32768]
KOR3 = [2039, 4093, 16381]


def store_version():
    import hashlib
    p = HERE / 'vendor/tree/experiments/jcp-references/store/MANIFEST.sha256'
    return hashlib.sha256(p.read_bytes()).hexdigest()


def arms2d():
    a = [dict(name=f'gauss{n}', rule=f'gauss{n}', family='gauss') for n in GAUSS2]
    a += [dict(name=f'fib{m}', rule=f'fib{m}', family='fibonacci') for m in FIB2]
    a += [dict(name=f'sobol{m}', rule=f'sobol{m}', family='sobol') for m in SOB2]
    a += [dict(name='ctrl_smolyak8', rule='smolyak8', family='control')]
    return a


def foms2d(L):
    f = [dict(name=f'UB_L{L}', kind='fd', fam='UB', L=L, dt=0.005, lev='2'),
         dict(name=f'CC_L{L}', kind='fd', fam='CC', L=L, dt=0.005, lev='2')]
    f += [dict(name=f'CC_L{Lc}', kind='fd', fam='CC', L=Lc, dt=0.005, lev='2') for Lc in (32, 64, 128)]
    f += [dict(name='UB_L128', kind='fd', fam='UB', L=128, dt=0.005, lev='2')]
    f += [dict(name=f'PS_N{N}', kind='ps', N=N, dt=0.0125) for N in (12, 16, 20, 24, 32, 48, 64)]
    return f


def cfg2d(L, cohort='test64', cases=None, local=False, arms=None, foms=None, timing=None):
    c = dict(dim=2, mesh=L, cohort=cohort, gtol=1e-3, step_budget=600,
             checkpoint='vendor/tree/experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
             rotation='vendor/tree/experiments/quadrature-study/inputs/rotation_R512.npz',
             rules_manifest='rules2d_manifest.json', expected_store_version=store_version(),
             arms=arms or arms2d(), conv_arm='gauss256', anchor_check_arm='fib46368', controls=['ctrl_smolyak8'],
             foms=foms or foms2d(L), audit_cases=[0, 2],
             timing=timing or dict(cases=6, reps=3, confirm_reps=5, burn=0.1, initial_burn=30.0, seed=20261105))
    if cases is not None:
        c['cases'] = cases
    if local:
        c['local_smoke'] = True
    return c


def arms3d():
    a = [dict(name=f'lat{n}', rule=f'lat{n}', family='lattice') for n in CBC3]
    a += [dict(name=f'gl{p}', rule=f'gl{p}', family='gauss') for p in GL3]
    a += [dict(name=f'sob{m}', rule=f'sob{m}', family='sobol') for m in SOB3]
    a += [dict(name=f'kor{n}', rule=f'kor{n}', family='korobov') for n in KOR3]
    a += [dict(name='ctrl_smol8', rule='smol8', family='control')]
    return a


def foms3d(n):
    L = n - 1
    f = [dict(name=f'hist_dt{dt:g}', kind='hist', dt=dt, ntol=1e-3, ltol=0.1) for dt in (0.005, 0.01, 0.025)]
    f += [dict(name=f'UB_L{L}', kind='fd', fam='UB', L=L, dt=0.01, lev=1),
          dict(name=f'CC_L{L}', kind='fd', fam='CC', L=L, dt=0.01, lev=1)]
    if L != 64:
        f += [dict(name='CC_L64', kind='fd', fam='CC', L=64, dt=0.01, lev=1)]
    f += [dict(name=f'PS_N{N}', kind='ps', N=N, dt=0.0125) for N in (8, 12, 16, 20, 24, 32)]
    return f


def cfg3d(n, cohort='test32', cases=None, local=False, arms=None, foms=None, timing=None):
    c = dict(dim=3, mesh=n, cohort=cohort, Rp=512, dt=0.01, gtol=1e-3, trust_fraction=0.05,
             model='vendor/tree/experiments/quadrature-burgers3d/inputs/model_M2',
             expected_model_sha256='6687f259947ec08a986732db157db69c66f1110ab28125a029de17b4aa389af9',
             rules_npz='rules3d.npz', rules_manifest='rules3d_manifest.json', expected_store_version=store_version(),
             arms=arms or arms3d(), conv_arm='lat32768', anchor_check_arm='gl32', controls=['ctrl_smol8'],
             foms=foms or foms3d(n),
             timing=timing or dict(cases=6, reps=3, confirm_reps=5, burn=0.1, initial_burn=30.0, seed=20261105))
    if cases is not None:
        c['cases'] = cases
    if local:
        c['local_smoke'] = True
    return c


def manifest2d():
    sys.path.insert(0, str(HERE))
    import plateau2d as P2  # noqa: F401  (sets sys.path for the vendored modules)
    import qcore as Q
    out = {}
    for a in arms2d():
        X, w = Q.offmesh_rule(a['rule'])
        out[a['rule']] = P2.rule_sha(X, w)
    (HERE / 'rules2d_manifest.json').write_text(json.dumps(dict(
        note='sha256(X.tobytes() + w.tobytes()) of every 2D rule, from vendored qcore.offmesh_rule, computed locally',
        rules=out), indent=1) + '\n')
    print('rules2d_manifest.json', len(out))


def main():
    if '--manifest2d' in sys.argv:
        manifest2d()
    d = HERE / 'configs'
    d.mkdir(exist_ok=True)
    for L in (256, 512, 1024, 2048, 4096):
        (d / f'p2d_{L}.json').write_text(json.dumps(cfg2d(L), indent=1) + '\n')
    # local smoke: dev6 cases 0,1 at 256^2, a few rungs, two FOMs, tiny timing
    sm = [a for a in arms2d() if a['name'] in ('gauss32', 'gauss64', 'gauss256', 'fib1597', 'fib46368', 'sobol4096',
                                               'ctrl_smolyak8')]
    fs = [f for f in foms2d(256) if f['name'] in ('UB_L256', 'CC_L64', 'PS_N24')]
    (d / 'smoke2d_local.json').write_text(json.dumps(cfg2d(256, 'dev6', [0, 1], True, sm, fs,
                                                           dict(cases=2, reps=1, confirm_reps=1, burn=0.05,
                                                                initial_burn=1.0, seed=1)), indent=1) + '\n')
    for n in (65, 129, 257):
        (d / f'p3d_{n}.json').write_text(json.dumps(cfg3d(n), indent=1) + '\n')
    sm3 = [a for a in arms3d() if a['name'] in ('lat256', 'lat4096', 'lat32768', 'gl16', 'gl32', 'sob4096', 'ctrl_smol8')]
    fs3 = [f for f in foms3d(65) if f['name'] in ('hist_dt0.01', 'CC_L64', 'PS_N16')]
    (d / 'smoke3d_local.json').write_text(json.dumps(cfg3d(65, 'dev4', [0, 1], True, sm3, fs3,
                                                           dict(cases=2, reps=1, confirm_reps=1, burn=0.05,
                                                                initial_burn=1.0, seed=1)), indent=1) + '\n')
    print('configs written')


if __name__ == '__main__':
    main()
