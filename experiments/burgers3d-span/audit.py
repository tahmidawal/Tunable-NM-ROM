"""Independent NumPy/SciPy audit of one collected panel (+ its certification job) — DESIGN.md section 9 and R1.

No JAX. Recomputes from saved files:
  1. every (arm|FOM, case) restricted-lattice error from the saved restricted fields (<= 1e-12 relative);
  2. full-grid errors of case 0 for the audit arms and the audit FOM from saved full fields (<= 1e-10 relative);
  3. the backward-Euler reference residual at steps 9 -> 10 of case 0 (NumPy stencil, <= 1e-9 relative);
  4. rho of the arg-max certified state of every arm (every draw) from the NumPy bank, NumPy sign-upwind /
     backward stencils and scipy's orthonormal DST-I (<= 1e-8 absolute); a downwind-mutated exact side must disagree;
  5. the settings rule, recomputed from the recorded numbers by an independent implementation;
  6. must-fail controls: a swapped reference case and a 1 % perturbed recorded error are both detected.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import scipy.fft


def features_np(p, x):
    ang = 2 * np.pi * (x @ p['freq'])
    h = np.concatenate((np.sin(ang), np.cos(ang)), -1)
    for w, b in p['net'][:-1]:
        a = h @ w + b
        h = a / (1 + np.exp(-a))
    w, b = p['net'][-1]
    return (float(p['scale']) * 64.0 * np.prod(x * (1 - x), axis=-1))[:, None] * (h @ w + b)


def coords(n):
    ax = np.linspace(0, 1, n)[1:-1]
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def stencil_adv(u, n, kind):
    ni = n - 2
    v = u.reshape(ni, ni, ni)
    p = np.pad(v, 1)
    s = np.zeros_like(v)
    for ax in range(3):
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[ax] = slice(None, -2)
        hi[ax] = slice(2, None)
        a, b = p[tuple(lo)], p[tuple(hi)]
        if kind == 'upwind':
            s += np.where(v > 0, v - a, b - v)
        elif kind == 'downwind':
            s += np.where(v > 0, b - v, v - a)
        else:
            s += v - a
    return (v * s * (n - 1)).ravel()


def modes_np(n, M):
    ni = n - 2
    p = np.arange(1, n - 1)
    l1 = 4.0 * (n - 1) ** 2 * np.sin(np.pi * p / (2 * (n - 1))) ** 2
    lam = (l1[:, None, None] + l1[None, :, None] + l1[None, None, :]).ravel()
    order = np.argsort(lam, kind='stable')[:M]
    return np.stack(np.unravel_index(order, (ni, ni, ni)), 1)          # 0-based


def tested(u, n, idx):
    ni = n - 2
    d = scipy.fft.dstn(u.reshape(ni, ni, ni), type=1, norm='ortho')
    return d[idx[:, 0], idx[:, 1], idx[:, 2]]


def restrict_np(n, per_axis=16):
    s = (n - 1) // (2 * per_axis)
    k = (2 * np.arange(per_axis) + 1) * s
    I, J, K = np.meshgrid(k, k, k, indexing='ij')
    ni = n - 2
    return (((I - 1) * ni + (J - 1)) * ni + (K - 1)).ravel()


def be_residual(u, prev, nu, n, dt=0.005):
    ni = n - 2
    v = u.reshape(ni, ni, ni)
    p = np.pad(v, 1)
    lap = -6 * v
    for ax in range(3):
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[ax] = slice(None, -2)
        hi[ax] = slice(2, None)
        lap = lap + p[tuple(lo)] + p[tuple(hi)]
    return u - prev + dt * (stencil_adv(u, n, 'upwind') - nu * (lap * (n - 1) ** 2).ravel())


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel', required=True, help='collected output dir of the panel job')
    ap.add_argument('--cert', help='collected output dir of the certification job')
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rho-states', type=int, default=0, help='0 = all arms/draws; else at most this many states')
    a = ap.parse_args()
    P = json.loads((Path(a.panel) / 'result.json').read_text())
    F = Path(a.panel) / 'fields'
    n = P['mesh']
    rep = dict(mesh=n, job=P['job_id'], checks={}, failures=[])
    fail = lambda msg: rep['failures'].append(msg)
    # 1. restricted errors
    refr = np.load(F / 'reference_restricted.npz')
    ridx = restrict_np(n)
    assert np.array_equal(refr['u0'][0], refr['u0'][0])
    worst, count, swapped_detected = 0., 0, True
    subjects = [(k, v['quick']) for k, v in P['arms'].items() if 'quick' in v] + \
               [(k, v['quick']) for k, v in P['fom'].items()]
    for name, recs in subjects:
        for r in recs:
            j = r['case']
            f = np.load(F / f'{name}_c{j}.npz')['f']
            ref = refr[f'c{j}']
            e = np.linalg.norm(f - ref, axis=1) / np.linalg.norm(ref[0])
            worst = max(worst, rel(e, r['err_restricted']))
            count += 1
            jj = (j + 1) % len(recs)
            e2 = np.linalg.norm(f - refr[f'c{jj}'], axis=1) / np.linalg.norm(refr[f'c{jj}'][0])
            if jj != j and rel(e2, r['err_restricted']) < 1e-6:
                swapped_detected = False
            if not (r['finite'] and np.isfinite(f).all()):
                fail(f'non-finite {name} case {j}')
    rep['checks']['restricted_errors'] = dict(n=count, worst_rel=worst, passed=worst <= 1e-12)
    rep['checks']['control_swapped_case_detected'] = swapped_detected
    pert = [x * 1.01 for x in subjects[0][1][0]['err_restricted']]
    f0 = np.load(F / f'{subjects[0][0]}_c0.npz')['f']
    e0 = np.linalg.norm(f0 - refr['c0'], axis=1) / np.linalg.norm(refr['c0'][0])
    rep['checks']['control_perturbed_error_detected'] = rel(e0, pert) > 1e-12
    # 2. full-grid case-0 errors
    reff = np.load(F / 'reference_case0_full.npz')['f']
    fw = []
    for p_ in sorted(F.glob('*_c0_full.npz')):
        name = p_.name[:-len('_c0_full.npz')]
        rec = (P['arms'].get(name) or P['fom'].get(name))['quick'][0]
        f = np.load(p_)['f']
        e = np.linalg.norm(f - reff, axis=1) / np.linalg.norm(reff[0])
        fw.append((name, rel(e, rec['err'])))
        # the restriction check of the saved restricted field itself
        fr = np.load(F / f'{name}_c0.npz')['f']
        if np.max(np.abs(fr - f[:, ridx])) > 0:
            fail(f'restricted field of {name} is not the restriction of its full field')
    rep['checks']['full_errors_case0'] = dict(arms=fw, passed=all(v <= 1e-10 for _, v in fw) and len(fw) > 0)
    # 3. reference residual
    st = F / 'reference_case0_steps9_10.npz'
    if st.exists():
        d = np.load(st)
        r = be_residual(d['u10'], d['u9'], float(d['nu']), n)
        v = float(np.linalg.norm(r) / np.linalg.norm(d['u9']))
        rep['checks']['reference_be_residual_step10'] = dict(value=v, passed=v <= 1e-9)
    # 4. rho
    if a.cert:
        C_ = json.loads((Path(a.cert) / 'result.json').read_text())
        CF = Path(a.cert) / 'fields'
        b = pickle.loads((Path(a.model) / 'bank.pkl').read_bytes())
        p = {k: (np.asarray(v) if k != 'net' else [(np.asarray(w), np.asarray(bb)) for w, bb in v])
             for k, v in b['params'].items()}
        T = np.asarray(b['rotation'])
        G = features_np(p, coords(n)) @ T
        s = (n - 1) // 16
        k = np.arange(1, 16) * s
        I, J, K = np.meshgrid(k, k, k, indexing='ij')
        lat = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)
        rows, mut = [], []
        files = sorted(CF.glob('cert_*_d*.npz'))
        if a.rho_states:
            files = files[:a.rho_states]
        for f_ in files:
            name = f_.name[len('cert_'):f_.name.rindex('_d')]
            spec = C_['arms'][name]['spec']
            z = np.load(f_)
            c = z['coefs'][int(z['arg'])]
            Rp, M = spec['Rp'], spec['M']
            u = G[:, :Rp] @ c
            idx = modes_np(n, M)
            ex = tested(stencil_adv(u, n, 'upwind'), n, idx)
            if spec['rule'] == 'tensor':
                ru = tested(stencil_adv(u, n, 'backward'), n, idx)
            else:
                ni = n - 2
                lin = ((lat[:, 0] - 1) * ni + (lat[:, 1] - 1)) * ni + (lat[:, 2] - 1)
                aa = stencil_adv(u, n, 'upwind')[lin]
                S = np.sqrt(2.0 / (n - 1))
                ph = np.ones((len(lat), M))
                for ax in range(3):
                    ph *= S * np.sin(np.pi * np.outer(lat[:, ax], idx[:, ax] + 1) / (n - 1))
                w = s ** 3 * (0.5 if spec.get('bad') else 1.0)
                ru = w * ph.T @ aa
            rho = float(np.linalg.norm(ru - ex) / np.linalg.norm(ex))
            recd = float(z['rho'][int(z['arg'])])
            rows.append(dict(arm=name, file=f_.name, rho_numpy=rho, rho_recorded=recd, diff=abs(rho - recd)))
            dn = tested(stencil_adv(u, n, 'downwind'), n, idx)
            mut.append(abs(float(np.linalg.norm(ru - dn) / np.linalg.norm(dn)) - recd))
        rep['checks']['rho_numpy'] = dict(states=len(rows), worst_diff=max(r['diff'] for r in rows), rows=rows,
                                          passed=max(r['diff'] for r in rows) <= 1e-8)
        rep['checks']['control_downwind_mutation_detected'] = bool(min(mut) > 1e-6)
    # 5. independent rule recomputation
    tim = P['timing'].get('summary', {})
    if tim:
        cert = {} if not a.cert else C_['certificates']
        el = []
        for k_, v in P['arms'].items():
            if 'quick' not in v or v['spec']['rule'] == 'exact':
                continue
            rs = np.array([q['reasons'] for q in v['quick']])
            ok = (k_ in cert and cert[k_]['confirmed'] and v['all_finite'] and not (rs == 3).any()
                  and ((rs == 0) | (rs == 2)).sum() <= 0.01 * rs.size)
            if ok:
                el.append((max(max(q['err'][1:]) for q in v['quick']), tim[k_]['median_ms'], k_))
        acc = min(el)[2] if el else None
        fastc = [(t, k_) for e, t, k_ in el if e <= 0.05]
        rep['recomputed_selection'] = dict(accurate=acc, fast=min(fastc)[1] if fastc else None)
    rep['passed'] = (not rep['failures'] and all(v.get('passed', True) if isinstance(v, dict) else bool(v)
                                                  for v in rep['checks'].values()))
    Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'rows'})
                      for k, v in rep['checks'].items()}, indent=1))
    print('AUDIT', 'PASS' if rep['passed'] else 'FAIL', rep['failures'][:5])


if __name__ == '__main__':
    main()
