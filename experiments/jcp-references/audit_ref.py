"""jcp-references gate G5: independent NumPy audit of the collected store (DESIGN A1.13, A2.8, A5.9).

Re-implements, WITHOUT importing verdict.py / refcore.py / refstore.py:
  - the canonical t0 lattice field from the PINNED cohort definition (dimension, cohort, case index) -> descriptors by
    the campaign generators, then the IC formula written out here (2D Gaussian with zeroed walls; 3D masked blobs with
    s_star recomputed on the 257^3 grid), compared with the stored t0 (detects swapped records even when fields and
    metadata are swapped together);
  - per-step acceptance from the stored per-stage residual / Newton arrays;
  - d12, d23, observed order (own solver), eps_ref, direction c, floor condition, Peclet, outcome;
  - the cross-check verdicts from the stored pseudo-spectral runs; old-vs-new distances.
Compares everything with the stored metadata / results/summary.json (relative 1e-12), and runs two fault injections
(2D val32 cases 0 and 1 swapped; u_{L=4096}(t_3) of 2D val32 case 0 x 1.01) that must be detected.

    python audit_ref.py            -> results/audit_ref.json (all_pass)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STORE = HERE / 'store'
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'mr-burgers2d'))
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d' / 'vendor' / 'paper-b3d' / 'vendor'))

PIN = {2: {'dev6': [(7090702, 4), (911702, 2)], 'val32': [(20260927, 32)], 'test64': [(20260916, 64)]},
       3: {'dev4': [(923651, 4)], 'val64': [(923801, 64)], 'test32': [(923901, 32)]}}
XC = {2: ('val32', [0, 1, 2, 3, 4, 5, 19, 26]), 3: ('val64', [0, 1, 2, 3, 5, 33, 43, 47])}
TOL = 1e-12          # relative agreement ...
ATOL = 1e-15         # ... plus an absolute floor for round-off on tiny quantities (A6.4)
OLD2D = {'dev6': 'refdv', 'val32': 'refdv', 'test64': 'reft'}
WT = HERE.parents[2]


# ------------------------------------------------------------------------------------------- independent IC --

def draws_2d(cohort):
    import engines                                     # the generator is the cohort definition (numpy only)
    return np.concatenate([engines.params_draw(s, c) for s, c in PIN[2][cohort]])


def t0_2d(desc):
    cx, cy, w, a, _ = desc
    x = np.arange(257) / 256
    X, Y = np.meshgrid(x, x, indexing='ij')
    u = a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w * w))
    u[[0, -1], :] = 0.0
    u[:, [0, -1]] = 0.0
    return u


_TAB = {}


def table_3d(cohort):
    if cohort not in _TAB:
        import b3d_common
        (s, c), = PIN[3][cohort]
        _TAB[cohort] = b3d_common.draw_param_table(s, c)    # raw draws (numpy); s_star recomputed below
    return _TAB[cohort]


def blob(X, Y, Z, T, j):
    s = 0.0
    for b in range(int(T['B'][j])):
        c, w = T['c'][j][b], T['w'][j][b]
        s = s + T['rho'][j][b] * np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / (2 * w * w))
    return s * 64 * X * (1 - X) * Y * (1 - Y) * Z * (1 - Z)


def s_star(T, j):
    x = np.linspace(0, 1, 257)
    best = 0.0
    Y, Z = np.meshgrid(x, x, indexing='ij')
    for xi in x:
        best = max(best, float(np.max(blob(xi, Y, Z, T, j))))
    return best


def t0_3d(T, j):
    x = np.arange(1, 64) / 64
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return (T['A'][j] * blob(X, Y, Z, T, j) / s_star(T, j)).reshape(-1)


# --------------------------------------------------------------------------------------------- independent maths --

def norms(a):
    a = np.asarray(a, float)
    return np.sqrt((a.reshape(a.shape[0], -1) ** 2).sum(1))


def order(d12, d23, h):
    if not (np.isfinite(d12) and np.isfinite(d23)) or d12 <= 0 or d23 <= 0:
        return None
    g = lambda p: (h[0] ** p - h[1] ** p) / (h[1] ** p - h[2] ** p)
    r = d12 / d23
    lo, hi = 0.25, 8.0
    if not (g(lo) <= r <= g(hi)):
        return None
    for _ in range(300):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(mid) < r else (lo, mid)
    return (lo + hi) / 2


def judge(us, Ls, taus, oks, umax, t0, nu):
    h = [1 / L for L in Ls]
    n0 = float(np.linalg.norm(t0.ravel()))
    d12, d23 = norms(us[0] - us[1]), norms(us[1] - us[2])
    p = [order(d12[k], d23[k], h) for k in range(1, 6)]
    eps = [None if q is None else d23[k + 1] / n0 * h[2] ** q / (h[1] ** q - h[2] ** q) for k, q in enumerate(p)]
    c = []
    for k, q in enumerate(p):
        if q is None:
            c.append(None)
            continue
        rho = (h[0] ** q - h[1] ** q) / (h[1] ** q - h[2] ** q)
        c.append(float(np.linalg.norm(((us[0][k + 1] - us[1][k + 1]) - rho * (us[1][k + 1] - us[2][k + 1])).ravel())
                       / d12[k + 1]))
    t0dev = max(float(np.linalg.norm((u[0] - t0).ravel())) / n0 for u in us)
    floor = all(d12[k] / n0 >= 10 * (taus[0] + taus[1]) and d23[k] / n0 >= 10 * (taus[1] + taus[2]) for k in range(1, 6))
    pec = umax * h[0] / nu
    if not (t0dev <= 1e-13):                            # NaN-safe, same semantics as production
        out = 'ic-mismatch'
    elif not all(oks):
        out = 'solver-fail'
    elif not floor:
        out = 'unresolved-floor'
    elif not pec < 2:
        out = 'peclet-fail'
    elif not all(q is not None and 1.8 <= q <= 2.2 for q in p):
        out = 'order-fail'
    elif not all(x is not None and x <= 0.5 for x in c):
        out = 'direction-inconsistent'
    else:
        out = 'accepted'
    e_ok = [e for e in eps if e is not None]
    return dict(p=p, eps_ref=max(e_ok) if len(e_ok) == 5 else None, eps_t=eps, c=c, n0=n0, outcome=out, t0dev=t0dev,
                d12=(d12[1:] / n0).tolist(), d23=(d23[1:] / n0).tolist(), peclet=pec, floor=floor)


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOL * max(1.0, abs(b)) if abs(b) < 1 else abs(a - b) <= TOL * abs(b)


def relclose(a, b):
    # production serialises non-finite numbers as JSON null (V_clean): treat null and non-finite alike
    if a is not None and not np.isfinite(a):
        a = None
    if b is not None and not np.isfinite(b):
        b = None
    if a is None or b is None:
        return a is None and b is None
    if not (np.isfinite(a) and np.isfinite(b)):
        return bool(a == b)
    return abs(a - b) <= TOL * max(abs(a), abs(b)) + ATOL


# ---------------------------------------------------------------------------------------------------- audit ----

def audit_case(dim, cohort, kind, j, desc_cache, t0_cache, arr=None, meta=None):
    stem = STORE / f'{dim}d/{cohort}/{kind}/c{j:03d}'
    meta = meta if meta is not None else json.loads(Path(str(stem) + '.json').read_text())
    arr = arr if arr is not None else dict(np.load(str(stem) + '.npz'))
    if (dim, cohort, j) not in t0_cache:
        if dim == 2:
            if cohort not in desc_cache:
                desc_cache[cohort] = draws_2d(cohort)
            t0_cache[(dim, cohort, j)] = (t0_2d(desc_cache[cohort][j]), float(desc_cache[cohort][j][4]))
        else:
            T = table_3d(cohort)
            t0_cache[(dim, cohort, j)] = (t0_3d(T, j), float(T['nu'][j]))
    t0, nu = t0_cache[(dim, cohort, j)]
    issues = []
    if float(np.max(np.abs(np.asarray(arr['t0']) - t0))) > 1e-13 * float(np.max(np.abs(t0))):
        issues.append('t0 differs from the pinned-definition IC (identity)')
    atts = {a['id']: a for a in meta['attempts']}
    for i, a in atts.items():                           # every attempt, superseded ones included
        res = np.asarray(arr['res_' + i])
        ok_i = bool(len(res) == a['expected_stages'] and np.all(np.isfinite(res)) and np.all(res <= 2e-11)
                    and np.isfinite(arr[i]).all())
        if ok_i != a['steps_ok']:
            issues.append(f'{i}: steps_ok mismatch (attempt history)')
    us, Ls, taus, oks = [], [], [], []
    for i in meta['final']:
        a = atts[i]
        res, nwt = np.asarray(arr['res_' + i]), np.asarray(arr['nwt_' + i])
        ok = bool(len(res) == a['expected_stages'] and np.all(np.isfinite(res)) and np.all(res <= 2e-11)
                  and np.isfinite(arr[i]).all())
        tau = 10 * float(np.sum(np.maximum(res, 1e-14))) if np.all(np.isfinite(res)) else math.inf
        if ok != a['steps_ok']:
            issues.append(f'{i}: steps_ok mismatch')
        if not relclose(tau, a['tau']):
            issues.append(f'{i}: tau mismatch')
        if int(nwt.sum()) != a['newton_total']:
            issues.append(f'{i}: newton total mismatch')
        us.append(np.asarray(arr[i], float))
        Ls.append(a['L'])
        taus.append(tau)
        oks.append(ok)
    umax = (lambda v: float(v) if v is not None and np.isfinite(v) else float('inf'))(atts[meta['final'][0]].get('umax_running'))    # null (non-finite) -> inf -> Peclet fails
    jd = judge(us, Ls, taus, oks, umax, t0, nu)
    sj = meta['judge']
    if jd['outcome'] != meta['outcome']:
        issues.append(f"outcome {jd['outcome']} != stored {meta['outcome']}")
    if not relclose(jd['eps_ref'], sj['eps_ref']):
        issues.append(f"eps_ref {jd['eps_ref']} != stored {sj['eps_ref']}")
    for a_, b_ in zip(jd['p'], sj['p']):
        if not relclose(a_, b_):
            issues.append(f'p mismatch {a_} vs {b_}')
            break
    for key in ('c', 'd12', 'd23', 'eps_t', 'p'):
        if len(jd[key]) != 5 or len(sj[key]) != 5:
            issues.append(f'{key}: wrong length')
            continue
        for a_, b_ in zip(jd[key], sj[key]):
            if not relclose(a_, b_):
                issues.append(f'{key} mismatch {a_} vs {b_}')
                break
    for key in ('n0', 'peclet'):
        if not relclose(jd[key], sj[key]):
            issues.append(f'{key} mismatch')
    if jd['floor'] != sj['checks']['floor']:
        issues.append('floor flag mismatch')
    if not relclose(nu, meta['desc']['nu']):
        issues.append('descriptor nu differs from the pinned definition')
    return dict(dim=dim, cohort=cohort, kind=kind, case=j, issues=issues, outcome=jd['outcome'], eps_ref=jd['eps_ref'],
                p=jd['p'])


def audit_crosscheck(dim, summary):
    coh, cases = XC[dim]
    out = {}
    for j in cases:
        att = sorted(STORE.glob(f'{dim}d/{coh}/PS/c{j:03d}_a*.json'))
        if not att or not (STORE / f'{dim}d/{coh}/ST/c{j:03d}.json').exists():
            out[j] = dict(verdict='MISSING', match=False)          # missing evidence never matches (completeness)
            continue
        ps = json.loads(att[-1].read_text())
        a = np.load(att[-1].with_suffix('.npz'))
        meta = json.loads((STORE / f'{dim}d/{coh}/ST/c{j:03d}.json').read_text())
        u3 = np.load(STORE / f'{dim}d/{coh}/ST/c{j:03d}.npz')[meta['final'][2]]
        n0 = meta['judge']['n0']
        eps = meta['judge']['eps_ref']
        runs = {r['id']: r for r in ps['runs']}
        sp = [np.asarray(a[i], float) for i in ps['selected_spatial']]
        Ns = [runs[i]['N'] for i in ps['selected_spatial']]
        half = np.asarray(a[ps['selected_half']], float)
        full = sp[-1]
        nums = {}
        sel_runs = [runs[i] for i in ps['selected_spatial'] + [ps['selected_half']]]
        # stability recomputed from dt, N3 and the recorded running maxima (A2.12 heuristic)
        sel_att = [r for r in ps['runs'] if r['attempt'] == sel_runs[0]['attempt']]
        um = max((lambda v: float(v) if v is not None and np.isfinite(v) else float('inf'))(r.get('umax')) for r in sel_att)
        dt_sel = runs[ps['selected_spatial'][0]]['dt']
        stab = bool(np.isfinite(um) and dt_sel <= 0.66 * 2.8 / (2 * dim * np.pi * Ns[-1] * um))
        if stab != ps['stability_ok']:
            nums['stability_mismatch'] = 1.0
        finite = all(np.isfinite(x).all() for x in sp + [half]) and all(r['finite'] for r in sel_runs)
        gates_ok = all(r['checks_ok'] and r['ic_quad_rel'] <= 1e-12 and r['ic_quad_max'] <= 1e-13
                       and r['contour_rel'] <= 1e-10 and r['job_gates_ok'] for r in sel_runs)
        finite = finite and bool(np.isfinite(u3).all())
        if meta['outcome'] != 'accepted' or eps is None or not np.isfinite(eps):
            v = 'FD-INVALID'
        elif not (finite and gates_ok):
            v = 'UNRESOLVED'
        else:
            da, db = norms(sp[0] - sp[1]), norms(sp[1] - sp[2])
            hh = [1 / N for N in Ns]
            q = [order(da[k], db[k], hh) for k in range(1, 6)]
            ind, okq = [], True                          # A10.1, re-implemented
            for k in range(1, 6):
                if not np.isfinite(db[k]):
                    okq = False
                elif db[k] / n0 <= 1e-12:                 # A10.3(ii)
                    qq = q[k - 1]
                    m = hh[2] ** qq / (hh[1] ** qq - hh[2] ** qq) if (qq is not None and qq >= 0.5) else 10.0
                    ind.append(max(1e-12, db[k] / n0 * max(1.0, m)))
                elif q[k - 1] is None or q[k - 1] < 0.5:
                    okq = False
                else:
                    qq = q[k - 1]
                    ind.append(db[k] / n0 * max(1.0, hh[2] ** qq / (hh[1] ** qq - hh[2] ** qq)))
            if not okq or not stab:
                v = 'UNRESOLVED'
            else:
                s_x = max(ind)
                s_t = float(np.max(norms(full - half)[1:]) / n0)
                x = float(np.max(norms(half - u3)[1:]) / n0)
                nums = dict(nums, s_x=s_x, s_t=s_t, sigma=s_x + s_t, x_ps=x, eta=(x / eps if eps > 0 else None))
                nums.update({f'q{k}': v_ for k, v_ in enumerate(q)})
                if not all(v_ is not None and np.isfinite(v_) for k_, v_ in nums.items() if not k_.startswith('q')):
                    v = 'UNRESOLVED'
                else:
                    v = 'UNRESOLVED' if s_x + s_t > eps / 2 else ('PASS' if x <= 2 * eps else 'FAIL')
        if v == 'UNRESOLVED' and ps.get('ladder_attempt', 0) == 0:
            v = 'UNRESOLVED-PENDING-ESCALATION'
        # A10.3(iv): the pre-A10 rule on the same ladder, re-derived independently
        if meta['outcome'] != 'accepted' or eps is None or not np.isfinite(eps):
            old_v = 'FD-INVALID'
        elif not (finite and gates_ok):
            old_v = 'UNRESOLVED'
        else:
            da_, db_ = norms(sp[0] - sp[1]), norms(sp[1] - sp[2])
            q_ = [order(da_[k], db_[k], [1 / N for N in Ns]) for k in range(1, 6)]
            raw = float(np.max(db_[1:]) / n0) + float(np.max(norms(full - half)[1:]) / n0)
            xx = float(np.max(norms(half - u3)[1:]) / n0)
            if any(x_ is None or x_ < 2.5 for x_ in q_) or not stab or not np.isfinite(raw) or raw > eps / 2:
                old_v = 'UNRESOLVED'
            else:
                old_v = 'PASS' if xx <= 2 * eps else 'FAIL'
        st = summary['dims'][str(dim)]['crosscheck'][f'{coh}/{j}']
        stq = {f'q{k}': v_ for k, v_ in enumerate(st.get('q') or [])}
        num_ok = 'stability_mismatch' not in nums and all(
            relclose(val, stq.get(k) if k.startswith('q') else st.get(k)) for k, val in nums.items())
        if any(k.startswith('q') for k in nums) and len(stq) != 5:
            num_ok = False
        out[j] = dict(verdict=v, stored=st['verdict'], numbers=nums, old_rule=old_v,
                      match=bool(v == st['verdict'] and num_ok and old_v == st.get('old_rule_verdict')))
    return out


def main():
    summary = json.loads((HERE / 'results' / 'summary.json').read_text())
    man = {}
    for line in (STORE / 'MANIFEST.sha256').read_text().splitlines():
        h, p = line.split(None, 1)
        man[p.strip()] = h
    bad_hash = [p for p, h in man.items() if hashlib.sha256((STORE / p).read_bytes()).hexdigest() != h]
    desc_cache, t0_cache = {}, {}
    rows = []
    for dim in (2, 3):
        for cohort in PIN[dim]:
            for kind in ('ST', 'S'):
                for p in sorted(STORE.glob(f'{dim}d/{cohort}/{kind}/c*.json')):
                    rows.append(audit_case(dim, cohort, kind, int(p.stem[1:]), desc_cache, t0_cache))
    xc = {d: audit_crosscheck(d, summary) for d in (2, 3)}
    # completeness: every historical case x kind must be present and audited
    have = {(r['dim'], r['cohort'], r['kind'], r['case']) for r in rows}
    missing = [f'{d}d/{c}/{k}/{j}' for d in PIN for c in PIN[d] for k in ('ST', 'S')
               for j in range(sum(n for _, n in PIN[d][c])) if (d, c, k, j) not in have]
    # old-vs-new distances, recomputed from the old reference files
    ovn_bad = []
    for d in (2, 3):
        for row in summary['dims'][str(d)]['old_vs_new']:
            meta = json.loads((STORE / f"{d}d/{row['cohort']}/{row['kind']}/c{row['case']:03d}.json").read_text())
            u3 = np.load(STORE / f"{d}d/{row['cohort']}/{row['kind']}/c{row['case']:03d}.npz")[meta['final'][2]]
            if d == 2:
                old = np.load(WT / '2026-10-01-quadrature-study/experiments/quadrature-study/runs' / OLD2D[row['cohort']] /
                              'archive/output' / f"ref_{row['kind']}_{row['cohort']}_{row['case']:03d}.npz")['f257']
            else:
                seed = PIN[3][row['cohort']][0][0]
                old = np.load(WT / '2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/runs/ref1/code/output'
                              / f'ref_{seed}.npz')[f"c{row['case']}"]
            dl = float(np.max(norms(np.asarray(old) - u3)[1:]) / float(np.linalg.norm(np.asarray(meta['judge']['n0']))))
            if not relclose(dl, row['delta_old']):
                ovn_bad.append(row)
    # fault injection 1: swap 2D val32 cases 0 and 1, fields AND metadata together
    faults = {}
    s0, s1 = (STORE / f'2d/val32/ST/c{j:03d}' for j in (0, 1))
    if Path(str(s0) + '.json').exists() and Path(str(s1) + '.json').exists():
        m1 = json.loads(Path(str(s1) + '.json').read_text())
        a1 = dict(np.load(str(s1) + '.npz'))
        r = audit_case(2, 'val32', 'ST', 0, desc_cache, t0_cache, arr=a1, meta=dict(m1, case=0))
        faults['swap'] = dict(detected=any('identity' in i for i in r['issues']), issues=r['issues'])
        # fault injection 2: u_{L=4096}(t_3) of case 0 multiplied by 1.01
        m0 = json.loads(Path(str(s0) + '.json').read_text())
        a0 = dict(np.load(str(s0) + '.npz'))
        i4096 = [i for i in m0['final'] if i.endswith('_L4096')]
        if i4096:
            a0[i4096[0]] = np.array(a0[i4096[0]], copy=True)
            a0[i4096[0]][3] *= 1.01
            r = audit_case(2, 'val32', 'ST', 0, desc_cache, t0_cache, arr=a0, meta=m0)
            sj = m0['judge']
            dp = max((abs(a_ - b_) for a_, b_ in zip(r['p'], sj['p']) if a_ is not None and b_ is not None), default=0)
            de = abs(r['eps_ref'] - sj['eps_ref']) / sj['eps_ref'] if r['eps_ref'] and sj['eps_ref'] else float('inf')
            faults['perturb'] = dict(detected=bool(np.isfinite(de) and max(dp, de) > 1e-6),
                                     max_dp=dp, rel_d_eps=de, issues=r['issues'])
        else:
            faults['perturb'] = dict(detected=False, issues=['L=4096 not among the final levels of val32 case 0'])
    else:
        faults = dict(swap=dict(detected=False, issues=['val32 cases 0/1 not in store']),
                      perturb=dict(detected=False, issues=['val32 case 0 not in store']))
    case_issues = [r for r in rows if r['issues']]
    xc_bad = {d: [j for j, v in xc[d].items() if not v['match']] for d in (2, 3)}
    out = dict(cases_audited=len(rows), case_issues=case_issues, bad_hashes=bad_hash, crosscheck=xc,
               crosscheck_mismatch=xc_bad, faults=faults, missing=missing, old_vs_new_mismatch=ovn_bad,
               all_pass=bool(not case_issues and not bad_hash and not any(xc_bad.values()) and not missing
                             and not ovn_bad and all(f['detected'] for f in faults.values())))
    (HERE / 'results' / 'audit_ref.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print('audited', len(rows), 'case-kinds; issues', len(case_issues), '; faults', {k: v['detected'] for k, v in faults.items()},
          '; all_pass', out['all_pass'])
    if not out['all_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
