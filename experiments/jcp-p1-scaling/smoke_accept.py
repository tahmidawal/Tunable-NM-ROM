"""Real-size smoke acceptance (DESIGN 6 as amended): every control predicate recomputed on CPU from the smoke's raw
outputs, independently of the in-job code where the design asks for it. Writes checks/smoke-accept.json.

    python smoke_accept.py runs/<smoke attempt>/archive/output
Exit status 0 iff every REQUIRED control behaves as registered.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import npcheck as NC
import p1score as SC
import p1stats as S

HERE = Path(__file__).resolve().parent
RES = {}


def req(name, cond, **info):
    RES[name] = dict(required=True, passed=bool(cond), **info)
    print(('PASS ' if cond else 'FAIL ') + name, {k: v for k, v in info.items() if not isinstance(v, (list, dict))}, flush=True)


def rep(name, **info):
    RES[name] = dict(required=False, **info)
    print('INFO ' + name, {k: v for k, v in info.items() if not isinstance(v, (list, dict))}, flush=True)


def modes(L, M):
    k = np.arange(1, L)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    lam = 4 * L ** 2 * (np.sin(np.pi * kx / (2 * L)) ** 2 + np.sin(np.pi * ky / (2 * L)) ** 2)
    ind = np.argsort(lam.ravel(), kind='stable')[:M]
    return kx.ravel()[ind], ky.ravel()[ind]


def acc_checks(d):
    r = json.loads((d / 'result.json').read_text())
    rows = r['rows']
    assert r['complete']
    for name, cfg in r['configs'].items():
        for sc in cfg['schemes']:
            R = dict(np.load(d / f'R_{name}_{sc}.npz'))
            q = SC.qualify(R, [x for x in rows if x['config'] == name and x['scheme'] == sc], cfg['Rp'])
            req(f'acc {name} {sc}: offline recomputation agrees with job', not q['disagreements'],
                disagreements=q['disagreements'])
            req(f'acc {name} {sc}: NumPy decode equals job decode', max(v['decode_rel'] for v in q['per_case'].values()) <= 1e-12,
                worst=max(v['decode_rel'] for v in q['per_case'].values()))
            req(f'acc {name} {sc}: G-cert', q['gcert_all'])
            rep(f'acc {name} {sc}: Q1/Q2 labels (smoke population)', Q1=q['Q1'], Q2=q['Q2'])
            if name in ('R128', 'R384'):
                g = SC.reproduction(R, name, sc, f'LSPG|{sc}|0.005|prod', job=r)
                req(f'reproduction gate {name} {sc} (dev2, <= 1e-10)', g['passed'], worst=g['worst'])
            if name == 'R384' and sc == 'CNR':
                # E0 scoring path and identity swaps (real records)
                ids = [str(x) for x in R['case_ids']]
                rec = dict(case_id=ids[0], time_tag=SC.TIME_TAG, scheme=sc, dt=.005, requested=tuple(ids[0].split('|')))
                rec['requested'] = (rec['requested'][0], int(rec['requested'][1]))
                s0 = SC.score(R['f257'][0], rec, 'ST')
                rep('E0 scoring path', e=s0['e'], eps_ref=s0['eps_ref'], eligible=s0['eligible'])
                for nm_, mut in (('case swap', dict(requested=(rec['requested'][0], rec['requested'][1] + 1))),
                                 ('time swap', dict(time_tag='t5,t4,t3,t2,t1,t0')), ('reference-kind swap', {})):
                    try:
                        SC.score(R['f257'][0], dict(rec, **mut), 'S' if nm_ == 'reference-kind swap' else 'ST')
                        ok = False
                    except SC.IdentityError:
                        ok = True
                    req(f'scorer rejects {nm_}', ok)
                # independent NumPy gamma vs in-job certifier on dev6|0 (A1.3)
                kx, ky = modes(1024, 1536)
                X, w = NC.gauss_rule(128)
                t128 = NC.tables(X, w, 384, kx, ky, 1024)
                j = ids.index('dev6|0')
                nu = nu_of('dev6', 0)
                gn = NC.gammas(R['states'][j], R['A'], R['lam'], t128, 'CNR', .005, nu)
                agree = np.all(np.abs(gn - R['gamma'][j]) <= 1e-6 * np.abs(R['gamma'][j]) + 1e-12)
                req('NumPy gamma equals the in-job certifier (R384 CN-R dev6|0, 50 steps)', agree,
                    worst=float(np.max(np.abs(gn - R['gamma'][j]))))
                # zero-update control (A1.6, round 3): constant trajectory vs the verified tight trajectory
                ft = NC.decode257(R['W_tight'][j], 384)
                n0 = float(R['n0'][j])
                motion = float(np.linalg.norm(ft[5] - ft[0])) / n0
                fz = NC.decode257(np.tile(R['w0'][j], (6, 1)), 384)
                dz = NC.fdist(fz, ft, n0)
                req('zero-update fixture moves (> 100 x 2.5e-4)', motion > 100 * 2.5e-4, motion=motion)
                req('zero-update solver fails Q1 distance', dz > 2.5e-4, distance=dz)
    c = r['controls']
    ur = c['under_resolved']
    req('under-resolved: in-job Q2 rollout bar fails on >= 1 case', ur['fails_rollout_bar'],
        dists=[x['dist'] for x in ur['rows']])
    req('under-resolved: in-job rho_max > 0.116', ur['rho_max'] > .116, rho_max=ur['rho_max'])
    z = np.load(d / 'CTRL_under_resolved.npz')
    kx, ky = modes(1024, 1536)
    t32 = NC.tables(*NC.gauss_rule(32), 384, kx, ky, 1024)
    t64 = NC.tables(*NC.gauss_rule(64), 384, kx, ky, 1024)
    off = [[NC.rho(z['W'][i][k], t32, t64) for k in range(6)] for i in range(len(z['W']))]
    job = [x['rho'] for x in ur['rows']]
    ag = all(SC.agree(a_, b_, 1e-9, 1e-13) for oa, ob in zip(off, job) for a_, b_ in zip(oa, ob))
    req('under-resolved: NumPy rho on own states agrees with job and > 0.116', ag and max(map(max, off)) > .116,
        rho_max_offline=max(map(max, off)))
    b0 = np.load(d / 'CTRL_budget0.npz')
    X, w = NC.gauss_rule(128)
    t128 = NC.tables(X, w, 384, b0['kx'], b0['ky'], int(b0['L']))
    gam = [NC.gammas(b0['states'][i], b0['A'], b0['lam'], t128, str(b0['scheme']), .005, float(b0['nu'][i]))
           for i in range(len(b0['states']))]
    pooled = np.concatenate(gam)
    nst = int(np.sum(~(pooled <= 1e-3)))
    req('budget-0: independent NumPy gamma > 3 of 300 pooled steps non-stationary', nst > .01 * pooled.size,
        nonstationary=nst, steps=int(pooled.size))
    req('budget-0: in-job pooled non-stationary > 1 %', c['budget0']['pooled_fraction'] > .01,
        fraction=c['budget0']['pooled_fraction'])
    req('budget-0: NumPy and in-job gamma agree', bool(np.all(np.abs(pooled - np.concatenate(b0['gamma'])) <=
                                                              1e-6 * np.abs(np.concatenate(b0['gamma'])) + 1e-12)))
    rep('sign-flipped Jacobian probe', rows=c['jflip_probe'])
    req('wrong-mesh table refused', c['wrong_mesh']['passed'], raised=c['wrong_mesh']['raised'])
    for rule, v in r['vendor_bridge']['R384'].items():
        req(f'vendor vs generic LSPG-BE bridge: {rule}', v['passed'], worst=max(x['W_rel'] for x in v['rows']))
    o = r['order']
    req('CN-R order check executed (GAL gate computed)', o['GAL']['passed'] is not None, GAL=o['GAL'], LSPG=o['LSPG'])
    rep('P1e block (smoke)', p1e=r.get('p1e'))


_NU = {}


def nu_of(coh, c):
    if not _NU:
        sys.path.insert(0, str(HERE / 'vendor' / 'quad2d'))
        sys.path.insert(0, str(HERE.parents[1] / 'experiments' / 'mr-burgers2d'))
        sys.path.insert(0, str(HERE.parents[1] / 'experiments' / 'separable-decoder'))
        import engines as e
        _NU['dev6'] = np.concatenate((e.params_draw(7090702, 4), e.params_draw(911702, 2)))
        _NU['val32'] = e.params_draw(20260927, 32)
    return float(_NU[coh][c][4])


def time_checks(d):
    r = json.loads((d / 'result.json').read_text())
    assert r['complete']
    req('split vs fused equivalence (every subject)', all(x['passed'] for x in r['equivalence']), rows=r['equivalence'])
    inj = set(r['config'].get('inject_identity_fault', []))
    normal = [x for x in r['invocations'] if x['B'] not in inj]
    bad = [x for x in r['invocations'] if x['B'] in inj]
    req('timed = scored identity on every non-injected triplet', all(x['all_hashes_ok'] for x in normal))
    req('injected identity fault: every triplet invalid', bool(bad) and not any(x['valid'] for x in bad))
    for lb in inj:
        xs = [x for x in r['invocations'] if x['B'] == lb]
        cases = sorted({x['case'] for x in xs})
        ratio = np.array([[x['ratio'] for x in xs if x['case'] == c_] for c_ in cases])[:, None, :]
        valid = np.array([[x['valid'] for x in xs if x['case'] == c_] for c_ in cases])[:, None, :]
        ratio = np.concatenate([ratio, ratio], 1)
        valid = np.concatenate([valid, valid], 1)
        out = S.h_cost_n(ratio, valid, ['a', 'b'])
        req('injected identity fault -> H-cost-N verdict not reached', out['outcome'] == 'verdict not reached', outcome=out['outcome'])
    for sub in sorted({x['subject'] for x in r['doubled']}):
        xs = [x for x in r['doubled'] if x['subject'] == sub]
        med = float(np.median([x['ratio'] for x in xs]))
        req(f'doubled work {sub}: B/A in [1.8, 2.2], identical iterations, identity', 1.8 <= med <= 2.2 and
            all(x['same_iterations'] and x['identity_ok'] for x in xs), median=med)
    for lb, s_ in r['subjects'].items():
        rows = s_['certification']
        req(f'time subject {lb}: certified (G-cert, valid)', all(x['gcert_ok'] and x['valid'] for x in rows))
    # production timing fault probe (reported): R128 CN-R at 256^2 / 1024^2 with one mesh x 3
    fam = ['256|R128|fib4181|frozen|CNR', '1024|R128|fib4181|frozen|CNR']
    xs = {lb: [x for x in normal if x['B'] == lb] for lb in fam}
    cases = sorted({x['case'] for x in xs[fam[0]]})
    ratio = np.array([[[x['ratio'] for x in xs[lb] if x['case'] == c_] for lb in fam] for c_ in cases])
    valid = np.array([[[x['valid'] for x in xs[lb] if x['case'] == c_] for lb in fam] for c_ in cases])
    base = S.h_cost_n(ratio, valid, [256, 1024], nboot=2000)
    r3 = ratio.copy()
    r3[:, 1] *= 3.
    rep('production timing fault probe (x3 on 1024^2; no requirement)', base=base, faulted=S.h_cost_n(r3, valid, [256, 1024], nboot=2000))
    rep('component-sum diagnostic', rows=r['components'])
    rep('recording overhead', rows=r['recording_overhead'])
    rep('A100 smoke ratios (hardware probe)', medians={lb: float(np.median([x['ratio'] for x in normal if x['B'] == lb]))
                                                       for lb in sorted({x['B'] for x in normal})})


def kern_checks(d, tag, rmax):
    r = json.loads((d / 'result.json').read_text())
    assert r['complete']
    for comp, v in r['sync'].items():
        req(f'{tag} dispatch-sync {comp} at r={rmax}', v['passed'], ratio=v['ratio_median'])
    for panel, v in r['drift'].items():
        rep(f'{tag} drift {panel}', rows=v)
        req(f'{tag} drift {panel} within band (after <= 1 re-run)', v[-1]['ok'])
    ce = r['component_equivalence']
    req(f'{tag} kernel components equal the solver advection / Jacobian', ce['N'] <= 1e-12 and ce['J'] <= 1e-12, **ce)


def main():
    root = Path(sys.argv[1])
    acc_checks(root / 'p1acc_smkacc')
    time_checks(root / 'p1time_smktime')
    kern_checks(root / 'p1kern_smkkern', '2D', 512)
    kern_checks(root / 'p1three_smkkern3d', '3D', 1024)
    t = json.loads((HERE / 'checks' / 'test_p1.json').read_text())
    req('CPU synthetic controls (test_p1.py) all pass', all(v.get('passed', True) for v in t.values() if isinstance(v, dict)))
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'smoke-accept.json').write_text(json.dumps(RES, indent=1, default=float))
    bad = [k for k, v in RES.items() if v.get('required') and not v['passed']]
    print('SMOKE ACCEPTED' if not bad else f'SMOKE NOT ACCEPTED: {bad}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
