"""imp-hrcost offline scoring, qualification and stop rules (CPU only, no JAX): DESIGN 3-5, A1.2-A1.4, A2.1-A2.3, A3.1,
A4.1.

    python hscore.py <run dir> [<out json>]

* every in-job distance is recomputed from the saved coefficients with the independent NumPy decoder (npcheck.decode257)
  and compared with the in-job value (1e-9 relative + 1e-13);
* the binding tested-vector rho is recomputed on the dev6 states with independent NumPy tables (npcheck bank evaluator,
  numpy Gauss / Fibonacci points, saved fitted weights) and compared with the in-job value;
* E0 scoring through p1score.score (identity-asserted records; ST only); C8 (swapped case id) must raise;
* configuration labels, errors on the common eligible set, timing statistics, and the stop-rule outcomes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import hlabel as HL
import npcheck as NC
import p1score as PS

TIME_TAG = PS.TIME_TAG
AGREE = (1e-9, 1e-13)
GAIN_F3 = 1.5


def agree(a, b):
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return bool(a == b)
    return abs(a - b) <= AGREE[0] * abs(b) + AGREE[1]


def rule_points(rule):
    if rule.startswith('gauss'):
        return NC.gauss_rule(int(rule[5:]))
    return _fib(rule)


def _fib(rule):
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor' / 'quad2d' / 'vendor'))
    import hari_quadrature as HQ
    a, b, k = 1, 1, 2
    n = int(rule[3:])
    while b < n:
        a, b, k = b, a + b, k + 1
    X, w = HQ.fibonacci_lattice(k, seed=0)
    return np.asarray(X, float), np.asarray(w, float)


class Tables:
    """Independent NumPy tested-vector tables per (projection, rule) (and fitted subsets)."""

    def __init__(self, run, Rp):
        self.run, self.Rp = Path(run), Rp
        z = np.load(self.run / 'arrays' / 'bank_ops.npz')
        self.kx, self.ky = z['kx'], z['ky']
        self.L = int(json.loads((self.run / 'result.json').read_text())['setup']['L'])
        self.cache = {}

    def get(self, proj, rule, fit=None):
        key = (proj, rule, fit)
        if key in self.cache:
            return self.cache[key]
        X, w = rule_points(rule)
        Gq, Gx, Gy = NC.values_grads(X, self.Rp)
        if proj == 'sine':
            x, y = X[:, 0:1], X[:, 1:2]
            psi = 2. * np.sin(self.kx[None] * np.pi * x) * np.sin(self.ky[None] * np.pi * y)
            T = self.L * w[:, None] * psi
        else:
            T = w[:, None] * Gq
        t = dict(Gq=Gq, Gs=Gx + Gy, T=T)
        if fit is not None:
            zf = np.load(self.run / 'arrays' / f'fit_{fit}.npz')
            idx, wh = zf['idx'], zf['w']
            sc = (wh / w[idx])[:, None]
            t = dict(Gq=Gq[idx], Gs=t['Gs'][idx], T=T[idx] * sc)
        self.cache[key] = t
        return t


def tested(c, t):
    return t['T'].T @ ((t['Gq'] @ c) * (t['Gs'] @ c))


def score_arm(run, name, R):
    z = np.load(Path(run) / 'arrays' / f'q1_{name}.npz')
    ids = [str(x) for x in z['case_ids']]
    out = {}
    for j, cid in enumerate(ids):
        coh, c = cid.split('|')
        f = NC.decode257(z['W'][j], z['W'].shape[-1])
        rec = dict(case_id=cid, time_tag=TIME_TAG, scheme='CN', dt=float(R['setup']['dt']), requested=(coh, int(c)))
        s = PS.score(f, rec, 'ST')
        out[cid] = dict(e=float(s['e']), eps_ref=float(s['eps_ref']), eligible=bool(s['eligible']),
                        reference_sha256=s['reference_sha256'], store_version=s['store_version'])
    return out


def offline_q1(run, name, R):
    """Recompute every Q1 distance from the saved W (independent decoder) and the label."""
    z = np.load(Path(run) / 'arrays' / f'q1_{name}.npz')
    ids = [str(x) for x in z['case_ids']]
    dev = [str(x) for x in z['dev_ids']]
    n0 = R['setup']['n0']
    rows, dis = {}, []
    r_job = R['q1'][name]['rows']
    Rp = z['W'].shape[-1]
    kdev = 0
    for j, cid in enumerate(ids):
        f, ft = NC.decode257(z['W'][j], Rp), NC.decode257(z['W_tight'][j], Rp)
        d = NC.fdist(f, ft, n0[cid])
        if not agree(d, r_job[cid]['dist_tight']):
            dis.append((cid, 'dist_tight', d, r_job[cid]['dist_tight']))
        row = dict(valid=r_job[cid]['valid'], eta=z['eta'][j], eta_rec=z['eta_rec'][j],
                   tight_valid=r_job[cid]['tight_valid'], dist_tight=d)
        if cid in dev:
            ftt = NC.decode257(z['W_tighter'][kdev], Rp)
            kdev += 1
            dtt = NC.fdist(ft, ftt, n0[cid])
            if not agree(dtt, r_job[cid]['dist_tt']):
                dis.append((cid, 'dist_tt', dtt, r_job[cid]['dist_tt']))
            row.update(tighter_valid=r_job[cid]['tighter_valid'], dist_tt=dtt)
        rows[cid] = row
    lab = HL.q1_label(rows, dev)
    return dict(label=lab['label'] if not dis else 'fail (offline disagreement)', reasons=lab['reasons'][:10],
                nonconv_frac=lab['nonconv_frac'], disagreements=dis, job_label=R['q1'][name]['label'],
                it_mean=float(np.mean([r_job[c]['it_mean'] for c in ids])),
                budget_exits=int(sum(r_job[c]['budget_exits'] for c in ids)))


def offline_q2(run, name, R, TB):
    """Recompute the rollout distances from saved W / Ws and the binding rho on dev6 with NumPy tables."""
    q = R['q2'][name]
    z = np.load(Path(run) / 'arrays' / f'q2_{name}.npz')
    ids = [str(x) for x in z['case_ids']]
    n0 = R['setup']['n0']
    Rp = z['W'].shape[-1]
    dis = []
    for j, cid in enumerate(ids):
        d = NC.fdist(NC.decode257(z['W'][j], Rp), NC.decode257(z['Ws'][j], Rp), n0[cid])
        if not agree(d, q['rows'][cid]['dist_succ']):
            dis.append((cid, 'dist_succ', d, q['rows'][cid]['dist_succ']))
    own, succ = q['own'], q['succ']
    proj = q['proj']
    fit = None
    if '_k' in name and name.split('_')[0] in ('F3a', 'F3b', 'F2F3', 'C2'):
        fam, k = name.split('_k')
        fit = f'{fam}_k{k}'
        own = 'gauss128'
    if fit is not None or name.startswith(('C1', 'C2', 'P6')):
        return dict(disagreements=dis, rho_checked=False)        # fitted rho: checked through the fit tables below
    tq, tp = TB.get(proj, own), TB.get(proj, succ)
    checked = 0
    for j, cid in enumerate(ids):
        if not cid.startswith('dev6'):
            continue
        for k in range(6):
            r_off = HL.rho(tested(z['W'][j][k], tq), tested(z['W'][j][k], tp))
            r_job = q['rows'][cid]['rho'][k]
            ok = (np.isnan(r_off) and np.isnan(r_job)) or (np.isfinite(r_off) and abs(r_off - r_job) <= 1e-8 * max(abs(r_job), 1e-300) + 1e-12)
            checked += 1
            if not ok:
                dis.append((cid, f'rho{k}', r_off, r_job))
    return dict(disagreements=dis, rho_checked=checked)


def offline_rho_fit(run, name, R, TB):
    q = R['q2'][name]
    z = np.load(Path(run) / 'arrays' / f'q2_{name}.npz')
    ids = [str(x) for x in z['case_ids']]
    tq = TB.get(q['proj'], 'gauss128', fit=name)
    tp = TB.get(q['proj'], 'gauss128')
    dis, checked = [], 0
    for j, cid in enumerate(ids):
        if not cid.startswith('dev6'):
            continue
        for k in range(6):
            r_off = HL.rho(tested(z['W'][j][k], tq), tested(z['W'][j][k], tp))
            r_job = q['rows'][cid]['rho'][k]
            checked += 1
            if not ((np.isnan(r_off) and np.isnan(r_job)) or abs(r_off - r_job) <= 1e-8 * max(abs(r_job), 1e-300) + 1e-12):
                dis.append((cid, f'rho{k}', r_off, r_job))
    return dict(disagreements=dis, rho_checked=checked)


def errors(sc, common):
    e = {c: sc[c]['e'] for c in common}
    worst_c = max(e, key=lambda c: e[c])
    return dict(worst=e[worst_c], worst_case=worst_c, median=float(np.median(list(e.values()))), n=len(e))


def no_worse(x, y, scx, scy):
    """A1.2: worst(X) - worst(Y) <= 2 eps*, eps* = max(eps_ref of X's worst case, of Y's worst case)."""
    eps = max(scx[x['worst_case']]['eps_ref'], scy[y['worst_case']]['eps_ref'])
    return bool(x['worst'] - y['worst'] <= 2 * eps), eps


def per_case_counts(scx, scy, common):
    w = t = i = 0
    for c in common:
        d = scx[c]['e'] - scy[c]['e']
        tie = abs(d) <= 2 * scx[c]['eps_ref']
        t += tie
        w += (not tie) and d > 0
        i += (not tie) and d < 0
    return dict(worse=int(w), tie=int(t), better=int(i))


def main():
    run = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / 'analysis.json'
    R = json.loads((run / 'result.json').read_text())
    Rp = int(R['setup']['Rp'])
    TB = Tables(run, Rp)
    A = dict(source=str(run), q1={}, q2={}, scores={}, arms={}, controls={})
    # ---- C8 scoring identity
    try:
        z = np.load(run / 'arrays' / 'q1_R.npz')
        f = NC.decode257(z['W'][0], Rp)
        PS.score(f, dict(case_id='dev6|1', time_tag=TIME_TAG, scheme='CN', dt=.005, requested=('dev6', 0)), 'ST')
        A['controls']['C8'] = dict(fired=False)
    except PS.IdentityError as e:
        A['controls']['C8'] = dict(fired=True, msg=str(e))
    # ---- Q1 / scores
    for name in R['q1']:
        if name.startswith('C4'):
            continue
        A['q1'][name] = offline_q1(run, name, R)
        A['scores'][name] = score_arm(run, name, R)
    # ---- Q2 offline checks
    for name in R['q2']:
        if name.startswith(('F3a', 'F3b', 'F2F3')):
            A['q2'][name] = offline_rho_fit(run, name, R, TB)
        else:
            A['q2'][name] = offline_q2(run, name, R, TB)
    # ---- arm labels
    def qual(name):
        q1 = A['q1'].get(name, {}).get('label', 'not run')
        q2 = R['q2'].get(name, {}).get('label', 'not run')
        q2p = R['q2'].get(name, {}).get('label_projected', 'not run')
        dis = bool(A['q2'].get(name, {}).get('disagreements'))
        return dict(q1=q1, q2=q2 if not dis else 'fail (offline disagreement)', q2_projected=q2p,
                    qualified=(q1 == 'pass' and q2 == 'pass' and not dis),
                    qualified_projected=(q1 == 'pass' and q2p == 'pass' and not dis))
    prod = [n for n in R['q1'] if not n.startswith('C4')]
    common = sorted(set.intersection(*[{c for c, v in A['scores'][n].items() if v['eligible']} for n in prod]))
    tim = R.get('timing', {})
    for n in prod:
        a = dict(qual(n), m=R['q1'][n]['m'], rule=R['q1'][n]['rule'], proj=R['q1'][n]['proj'])
        valid_all = all(r_['valid'] for r_ in R['q1'][n]['rows'].values())
        a['errors'] = errors(A['scores'][n], common) if valid_all else dict(worst=float('inf'), worst_case=None)
        if n in tim:
            a['timing'] = {k: tim[n][k] for k in ('e2e_median', 'evo_median', 'e2e_ci', 'evo_ci')}
        A['arms'][n] = a
    A['common_cases'] = common
    ref = A['arms']['R']
    for n, a in A['arms'].items():
        if n == 'R' or a['errors'].get('worst_case') is None:
            continue
        nw, eps = no_worse(a['errors'], ref['errors'], A['scores'][n], A['scores']['R'])
        a['vs_R'] = dict(no_worse=nw, eps_star=eps, counts=per_case_counts(A['scores'][n], A['scores']['R'], common))
        if 'timing' in a:
            t = a['timing']
            a['vs_R'].update(faster=bool(t['e2e_median'] > 1 and t['e2e_ci'][0] > 1),
                             gain15=bool(t['e2e_median'] >= GAIN_F3 and t['e2e_ci'][0] > 1))
    A['stop'] = stop_rules(R, A, binding=True)
    A['stop_projected'] = stop_rules(R, A, binding=False)
    out.write_text(json.dumps(A, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(json.dumps(dict(stop=A['stop'], stop_projected=A['stop_projected']), indent=1))


def stop_rules(R, A, binding=True):
    """A2.3 / A3.1 / A4.1 outcome precedence. binding=False: the non-binding Q2-projected reading."""
    key = 'qualified' if binding else 'qualified_projected'
    ref = A['arms']['R']
    if not ref[key]:
        return dict(F2='undecided: reference unqualified', F3='undecided: reference unqualified')
    out = {}
    # ---- F2
    F2 = R.get('F2')
    if F2 is None or 'candidates' not in F2:
        out['F2'] = 'fired (incomplete evidence): F2 not run / not built'
    else:
        lab2 = 'label' if binding else 'label_projected'
        mstar, unresolved = None, False
        for q in F2['candidates']:
            n = f'F2_{q}'
            l2 = R['q2'][n][lab2]
            if l2 == 'pass' and n in A['arms']:
                if A['arms'][n][key]:
                    mstar = q
                    break
                continue
            if l2 == 'pass' and n not in A['arms']:
                unresolved = True                   # Q2 pass but Q1 never evaluated
            elif l2 != 'fail':
                unresolved = True
        if mstar is None:
            out['F2'] = 'fired (incomplete evidence)' if unresolved else 'fired (tested failure): no qualified candidate'
        elif unresolved:
            out['F2'] = f'fired (incomplete evidence): qualified rule exists at {mstar} (upper bound on m*)'
        else:
            a = A['arms'][f'F2_{mstar}']
            v = a.get('vs_R', {})
            if 'faster' not in v or 'no_worse' not in v:
                out['F2'] = f'fired (incomplete evidence): m* = {mstar}, accuracy or timing missing'
            elif v['no_worse'] and v['faster']:
                out['F2'] = f'not fired: m* = {mstar}'
            else:
                out['F2'] = f'fired (tested failure): m* = {mstar}, no_worse={v["no_worse"]}, faster={v["faster"]}'
        out['F2_mstar'] = mstar
    # ---- F3
    cands = [f'{f}_k{k}' for f in ('F3a', 'F3b') for k in (4, 8, 16)]
    ok, unresolved, detail = [], False, {}
    for n in cands:
        if n not in A['arms']:
            unresolved = True
            detail[n] = 'not built / not run'
            continue
        a = A['arms'][n]
        v = a.get('vs_R', {})
        lab2 = R['q2'][n]['label' if binding else 'label_projected']
        detail[n] = f'q2={lab2} q1={a["q1"]} no_worse={v.get("no_worse")} gain15={v.get("gain15")}'
        if a[key] and v.get('no_worse') and v.get('gain15'):
            ok.append(n)
            continue
        q_failed = lab2 == 'fail' or a['q1'].startswith('fail')
        evaluated = (lab2 in ('pass', 'fail') and a['q1'] in ('pass', 'fail') and 'gain15' in v and 'no_worse' in v)
        if not (q_failed or evaluated):
            unresolved = True                       # a candidate that could still pass was not fully evaluated
    out['F3'] = ('not fired: ' + ', '.join(ok)) if ok else ('fired (incomplete evidence)' if unresolved else 'fired (tested failure)')
    out['F3_detail'] = detail
    return out


if __name__ == '__main__':
    main()
