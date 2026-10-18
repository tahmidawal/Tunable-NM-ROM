"""jcp-qualify smoke acceptance (DESIGN 7, 8; A1.3, A1.4, A2.3, A3.4): every required control recomputed from the smoke
archives (smkA qacc, smkB qtime + pilot, smk3 q3run) and the CPU tests. Writes checks/smoke-accept.json; exit 1 if any
required check fails. CPU only.

    JAX_PLATFORMS=cpu python smoke_accept.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import analyze as AN
import npcheck as NC
import qscore as SC
import qstats as QS

HERE = Path(__file__).resolve().parent
OUT = {}


def req(name, ok, **kw):
    OUT[name] = dict(required=True, passed=bool(ok), **kw)


def rep(name, **kw):
    OUT[name] = dict(required=False, **kw)


def smk_acc():
    d = AN.out_dir('smkA', 'qacc')
    if d is None:
        req('smkA archive present', False)
        return
    res = json.loads((d / 'result.json').read_text())
    req('smkA complete', res['complete'])
    cache = {}
    for s in res['config']['settings']:
        for arm in s['arms']:
            e = AN.lane_config(1024, d, res, cache, s['name'], arm)
            ak = f"R{s['Rp']}|{arm['form']}|{arm['scheme']}|{float(arm['factor']):g}"
            req(f'{ak}: offline recomputation agrees with job', not e['q']['disagreements'], disagreements=[str(x) for x in e['q']['disagreements'][:5]])
            rep(f'{ak}: labels', Q1=e['q']['Q1'], Q2=e['q']['Q2'].get('label'))
            if s['Rp'] in (128, 512):                          # preliminary combined-archive gates (dev cases)
                g = SC.combined_reproduction(e['R'], s['Rp'], arm['form'], arm['scheme'], .005 * float(arm['factor']))
                me = AN.manifest_equal_combined(e['R'], s['Rp'])
                req(f'{ak}: preliminary combined reproduction <= 1e-10 with numerical-manifest equality', g['passed'] and me,
                    worst=g['worst'], key=g['archive_key'], manifest_equal=me)
            if s['Rp'] == 384 and arm['form'] == 'LSPG':      # Delta-t0 gate against P1's archive (dev6)
                g = SC.p1_reproduction(e['R'], 'R384', arm['scheme'], 1024)
                req(f'{ak}: equals P1 acc1024 archive <= 1e-10', g['passed'], worst=g['worst'])
            if s['Rp'] == 384 and arm['form'] == 'GAL':       # designated positive GAL fixture (A2.3, A3.3)
                q1 = e['q']['Q1']
                req('positive GAL fixture: certificate, G-cert, validity pass', q1['certificate'] and q1['gcert'] and q1['all_valid'],
                    Q1=q1)
            if e.get('zero_update') and s['Rp'] == 384:       # zero-update control (A1.3): motion established, then fires
                z = e['zero_update']
                req(f'{ak}: zero-update motion > 100 x 2.5e-4 (independent establishment)', z['motion'] > 100 * 2.5e-4, motion=z['motion'])
                req(f'{ak}: zero-update control fires (distance > 2.5e-4)', z['dist'] > 2.5e-4, dist=z['dist'])
    c = res.get('controls', {})
    z = np.load(d / 'CTRL_under_resolved.npz')       # the actual Q2 classifier on the control's own arrays (offline decode)
    q2 = SC.q2_from_runs(z['W'], z['W_succ'], z['rho'], z['n0'], int(z['Rp']), bool(np.all(z['succ_valid'])))
    req('under-resolved rule: Q2 classifier returns fail', q2['label'] == 'fail' and not q2['rollout'] and not q2['rho'],
        worst=q2['worst_dist_qp'], rho_max=q2['rho_max'], label=q2['label'])
    for form in ('LSPG', 'GAL'):
        b = c.get(f'budget0_{form}', {})
        req(f'Q1 negative ({form}, LM budget 0): pooled non-certified > 1 %', (b.get('pooled_fraction') or 0) > .01,
            pooled=b.get('pooled_fraction'))
        z = np.load(d / f'CTRL_budget0_{form}.npz')
        ok, agree = independent_budget0(z, form)
        req(f'Q1 negative ({form}): independent NumPy certificate > threshold on > 1 % of steps', ok)
        req(f'Q1 negative ({form}): in-job certificate agrees with NumPy per step', agree)
    req('wrong-mesh table refused', c.get('wrong_mesh', {}).get('passed'))
    # B4 path test + full TH06 control (A3.4)
    o = AN.b4_from(d, res)
    th = o.get('R384|GAL_TH06')
    req('TH06 control: >= 5 interpretable dev6 cases and label order 1', th is not None and th['label']['label'] == 'order 1'
        and th['label']['interpretable'] >= 5, label=th['label'] if th else None)
    for k, v in o.items():
        if k != 'R384|GAL_TH06':
            rep(f'B4 path test {k}', cases=[dict(case=c_['case'], p=c_['p'], interpretable=c_['interpretable'], checks=c_['checks'])
                                             for c_ in v['cases']])


def independent_budget0(z, form):
    """Independent NumPy certificate of the budget-0 trajectories (P1 A1.6 pattern; GAL via its own QR, A1.3)."""
    sys.path.insert(0, str(HERE))
    states, A, lam = z['states'], z['A'], z['lam']
    t = SC.rule_tables(str(z['rule']), A.shape[1], z['kx'], z['ky'], int(z['L']))
    co = {'CNR': (1., -1., 0., .5, .5, 2), 'CN': (1., -1., 0., .5, .5, 0)}[str(z['scheme'])]
    nonc, agree_all, total = 0, True, 0
    Qm = np.linalg.qr(A)[0]
    for i in range(len(states)):
        nu = float(z['nu'][i])
        cj = np.asarray(z['cert'][i], float)
        for k in range(states.shape[1] - 1):
            a0, a1, a2, b0, b1 = (1., -1., 0., 1., 0.) if k < co[5] else co[:5]
            ck, cm, cn = states[i, k], states[i, max(k - 1, 0)], states[i, k + 1]
            Fn = NC.N(ck, t) + nu * lam * (A @ ck) if b1 != 0 else 0.
            Acn = A @ cn
            R = a0 * Acn + a1 * (A @ ck) + a2 * (A @ cm) + .005 * (b0 * (NC.N(cn, t) + nu * lam * Acn) + b1 * Fn)
            if form == 'LSPG':
                D = a0 + .005 * b0 * nu * lam
                r = R / D
                J = (a0 * A + .005 * b0 * (NC.dN(cn, t) + (nu * lam)[:, None] * A)) / D[:, None]
                v = float(np.linalg.norm(J.T @ r) / (np.linalg.norm(J) * np.linalg.norm(r) + 1e-300))
                bad = not v <= SC.GAMMA
                tol = 1e-6 * v + 1e-12
            else:
                v = float(np.linalg.norm(Qm.T @ R) / (np.linalg.norm(A @ ck) + 1e-300))
                bad = not v <= SC.ETA
                tol = 1e-6 * v + 1e-12
            nonc += bad
            total += 1
            agree_all = agree_all and abs(v - cj[k]) <= tol
    return nonc > .01 * total, bool(agree_all)


def smk_time():
    d = AN.out_dir('smkB', 'qtime')
    if d is None:
        req('smkB archive present', False)
        return
    res = json.loads((d / 'result.json').read_text())
    req('smkB complete', res['complete'])
    req('split = fused (fields 1e-12, iterations equal)', all(x['passed'] for x in res['equivalence']))
    bad = res['config']['inject_identity_fault']
    inj = [x for x in res['invocations'] if x['B'] in bad]
    req('injected identity fault invalidates every triplet of its subject', inj and all(not x['valid'] for x in inj))
    ok = [x for x in res['invocations'] if x['B'] not in bad]
    req('identity holds for every other subject', ok and all(x['all_hashes_ok'] for x in ok))
    fz = [x for x in res['fused'] if x['B'] not in bad]
    req('fused triplets: identity holds and drift in band for the other subjects', fz and all(x['hash_ok'] for x in fz)
        and sum(not x['valid'] for x in fz) <= len(fz) / 3)
    req('fused triplets of the injected subject invalid', all(not x['valid'] for x in res['fused'] if x['B'] in bad))
    dbl = res['doubled']
    med = float(np.median([x['ratio'] for x in dbl])) if dbl else None
    req('doubled work B/A in [1.8, 2.2] with identical iterations and identity', dbl and 1.8 <= med <= 2.2 and
        all(x['same_iterations'] and x['identity_ok'] and x['drift_ok'] for x in dbl), median=med)
    for lb, s in res['subjects'].items():
        rep(f'timing-job certification {lb}', rows=[{k: r_.get(k) for k in ('case', 'valid', 'noncertified', 'gcert_ok', 'dist_tight', 'dist_qp')}
                                                   for r_ in s['certification']])
        req(f'{lb}: G-cert holds in the timing job', all(r_['gcert_ok'] for r_ in s['certification']))
    p = AN.out_dir('smkB', 'qpilot', 'pilot')
    if p is None:
        req('pilot archive present', False)
    else:
        pr = json.loads((p / 'result.json').read_text())
        req('4096^2 pilot complete', pr['complete'])
        rep('4096^2 pilot peak bytes', peak=pr.get('peak'), feasible=pr.get('feasible'))


def smk_3d():
    d = AN.out_dir('smk3', 'q3run')
    if d is None:
        req('smk3 archive present', False)
        return
    res = json.loads((d / 'result.json').read_text())
    req('smk3 complete', res['complete'])
    for k, m in res['members'].items():
        lab = AN.label3d(m, 'smk3')
        req(f'{k}: reproduces speed3d archive <= 1e-10', lab['reproduction'] and lab['reproduction']['passed'], rep=lab['reproduction'])
        req(f'{k}: offline recomputation (q3offline) agrees with the job', lab['offline']['agrees'], offline=lab['offline'])
        req(f'{k}: populations and array shapes complete', lab['population_ok'])
        rep(f'{k}: Q2-3D (smoke)', label=lab['label'], worst=lab['worst_dist_qp'], rho=lab['rho_max'])
    c = AN.label3d(res['control'], 'smk3')
    req('3D under-resolved control fails Q2-3D', c['label'] == 'fail' and not c['rollout'], worst=c['worst_dist_qp'])
    req('3D control: offline recomputation agrees', c['offline']['agrees'], offline=c['offline'])
    rep('3D pilot', pilot=res.get('pilot'), peak_job=res.get('peak_bytes_job'))


def cpu_tests():
    for name in ('test_q.json', 'test_p1.json'):
        p = HERE / 'checks' / name
        r = json.loads(p.read_text()) if p.exists() else {}
        req(f'CPU {name} all pass', r and all(v['passed'] for v in r.values()))
    p = HERE / 'checks' / 'test_lmm.json'
    r = json.loads(p.read_text()) if p.exists() else None
    req('manufactured LMM oracle gates pass (positive B4 gate)', r is not None and r.get('all_pass', False))


if __name__ == '__main__':
    for f in (cpu_tests, smk_acc, smk_time, smk_3d):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            req(f'{f.__name__} ran', False, error=repr(e))
    (HERE / 'checks' / 'smoke-accept.json').write_text(json.dumps(OUT, indent=1, default=str) + '\n')
    bad = [k for k, v in OUT.items() if v['required'] and not v['passed']]
    print(f'{sum(v["required"] for v in OUT.values()) - len(bad)} of {sum(v["required"] for v in OUT.values())} required checks pass')
    for k in bad:
        print('FAIL', k, {kk: vv for kk, vv in OUT[k].items() if kk not in ('required', 'passed')})
    sys.exit(1 if bad else 0)
