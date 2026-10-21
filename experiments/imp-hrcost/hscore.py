"""imp-hrcost offline scoring, qualification and stop rules (CPU only, no JAX): DESIGN 3-5, A1.2-A1.4, A2.1-A2.3, A3.1,
A4.1 (code audit 1 fixes).

    python hscore.py <run dir> [<out json>]

* identities: the E0 store version and metric are pinned (checks/E0-PIN.json); the local checkpoint / rotation used by
  the independent decoder must equal the job's;
* every in-job distance (Q1 tight / tighter, Q2 successor and check) is recomputed from the saved coefficients with the
  independent NumPy decoder (npcheck.decode257) and must agree (1e-9 relative + 1e-13);
* the binding tested-vector rho and the projected indicator are recomputed on the dev6 states with independent NumPy
  tables (npcheck bank evaluator, NumPy points, saved fitted weights) and must agree (same tolerance);
* Q1 and Q2 labels are rebuilt from the verified evidence (hlabel), including the parent-convergence downgrade;
* E0 scoring through p1score.score (ST; identity-asserted); C8 (swapped case id) must raise;
* errors on the common eligible set (E0 eligibility of the registered population; every production arm must cover it);
* stop-rule outcomes (binding Q2, and the non-binding Q2-projected reading). Non-production populations are labelled
  "not a verdict".
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import hlabel as HL
import npcheck as NC
import p1score as PS

HERE = Path(__file__).resolve().parent
TIME_TAG = PS.TIME_TAG
REL, ABS = 1e-9, 1e-13
GAIN_F3 = 1.5
PROD = dict(mesh=1024, Rp=384, dt=.005, cases='all38', q2_cases='sub16')
PIN = json.loads((HERE / 'checks' / 'E0-PIN.json').read_text())
assert PIN['e0_store_version'] == PS.STORE_VERSION and PIN['metric_id'] == PS.METRIC_ID, 'E0 store / metric differ from pin'
FIT_FAMS = ('F3a', 'F3b', 'F2F3')


def agree(a, b):
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    if not (np.isfinite(a) and np.isfinite(b)):
        return bool(a == b)
    return abs(a - b) <= REL * abs(b) + ABS


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rule_points(rule):
    if rule.startswith('gauss'):
        return NC.gauss_rule(int(rule[5:]))
    sys.path.insert(0, str(HERE / 'vendor' / 'quad2d' / 'vendor'))
    import hari_quadrature as HQ
    a, b, k = 1, 1, 2
    n = int(rule[3:])
    while b < n:
        a, b, k = b, a + b, k + 1
    X, w = HQ.fibonacci_lattice(k, seed=0)
    return np.asarray(X, float), np.asarray(w, float)


class Tables:
    """Independent NumPy tested-vector tables per (projection, rule, fitted subset) plus the projection operator of the
    Q2-projected indicator (sine: Q^T with Q = A Rm^-1; bank: L^-1 with Mass = L L^T)."""

    def __init__(self, run, Rp, L):
        self.run, self.Rp, self.L = Path(run), Rp, L
        z = np.load(self.run / 'arrays' / 'bank_ops.npz')
        self.kx, self.ky = z['kx'], z['ky']
        self.Qt = np.linalg.solve(z['Rm'].T, z['A'].T)                 # Q^T = Rm^-T A^T
        self.Linv = np.linalg.inv(np.linalg.cholesky(z['Mass'])) if 'Mass' in z.files else None
        self.cache = {}

    def get(self, proj, rule, fit=None):
        key = (proj, rule, fit)
        if key in self.cache:
            return self.cache[key]
        X, w = rule_points(rule)
        Gq, Gx, Gy = NC.values_grads(X, self.Rp)
        if proj == 'sine':
            x, y = X[:, 0:1], X[:, 1:2]
            T = self.L * w[:, None] * 2. * np.sin(self.kx[None] * np.pi * x) * np.sin(self.ky[None] * np.pi * y)
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

    def proj(self, proj, v):
        return self.Qt @ v if proj == 'sine' else self.Linv @ v


def tested(c, t):
    return t['T'].T @ ((t['Gq'] @ c) * (t['Gs'] @ c))


# ---------------------------------------------------------------------------------------------- Q1 ----

def offline_q1(run, name, R):
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
                demonstrated=lab['demonstrated'] and not dis, nonconv_frac=lab['nonconv_frac'], disagreements=dis,
                job_label=R['q1'][name]['label'], case_ids=ids,
                it_mean=float(np.mean([r_job[c]['it_mean'] for c in ids])),
                budget_exits=int(sum(r_job[c]['budget_exits'] for c in ids)))


# ---------------------------------------------------------------------------------------------- Q2 ----

def tables_of(R, name, TB):
    """(own tables, successor tables) for the independent rho: classical rules by name; fitted rules by their saved
    weights over the parent Gauss 128^2."""
    q = R['q2'][name]
    proj = q['proj']
    fam = name.split('_k')[0]
    if fam in FIT_FAMS:
        return TB.get(proj, 'gauss128', fit=name), TB.get(proj, 'gauss128')
    return TB.get(proj, q['own']), TB.get(proj, q['succ'])


def offline_q2(run, name, R, TB, rho_check=True):
    q = R['q2'][name]
    z = np.load(Path(run) / 'arrays' / f'q2_{name}.npz')
    ids, dev = [str(x) for x in z['case_ids']], [str(x) for x in z['dev_ids']]
    n0 = R['setup']['n0']
    Rp = z['W'].shape[-1]
    dis, dsucc, dchk = [], {}, {}
    for j, cid in enumerate(ids):
        d = NC.fdist(NC.decode257(z['W'][j], Rp), NC.decode257(z['Ws'][j], Rp), n0[cid])
        dsucc[cid] = d
        if not agree(d, q['rows'][cid]['dist_succ']):
            dis.append((cid, 'dist_succ', d, q['rows'][cid]['dist_succ']))
    for j, cid in enumerate(dev):
        js = ids.index(cid)
        d = NC.fdist(NC.decode257(z['Ws'][js], Rp), NC.decode257(z['Wc'][j], Rp), n0[cid])
        dchk[cid] = d
        if not agree(d, q['chk'][cid]):
            dis.append((cid, 'dist_chk', d, q['chk'][cid]))
    checked = 0
    if rho_check:
        tq, tp = tables_of(R, name, TB)
        for j, cid in enumerate(ids):
            if cid not in dev:
                continue
            for k in range(6):
                c = z['W'][j][k]
                a, b = tested(c, tq), tested(c, tp)
                rb, rp = HL.rho(a, b), HL.rho(TB.proj(q['proj'], a), TB.proj(q['proj'], b))
                checked += 1
                if not agree(rb, q['rows'][cid]['rho'][k]):
                    dis.append((cid, f'rho{k}', rb, q['rows'][cid]['rho'][k]))
                if not agree(rp, q['rows'][cid]['rho_proj'][k]):
                    dis.append((cid, f'rho_proj{k}', rp, q['rows'][cid]['rho_proj'][k]))
    args = (list(z['own_valid']), list(z['succ_valid']), [dsucc[c] for c in ids])
    chk = ([bool(z['succ_valid'][ids.index(c)]) for c in dev], list(z['chk_valid']), [dchk[c] for c in dev])
    lab = HL.q2_label(*args, np.asarray([q['rows'][c]['rho'] for c in ids]), *chk)['label']
    labp = HL.q2_label(*args, np.asarray([q['rows'][c]['rho_proj'] for c in ids]), *chk)['label']
    return dict(label=lab, label_projected=labp, disagreements=dis, rho_checked=checked, case_ids=ids,
                job_label=q['label'], job_label_projected=q['label_projected'])


# ------------------------------------------------------------------------------------------- scoring ----

def score_arm(run, name, R):
    z = np.load(Path(run) / 'arrays' / f'q1_{name}.npz')
    ids = [str(x) for x in z['case_ids']]
    out = {}
    for j, cid in enumerate(ids):
        coh, c = cid.split('|')
        f = NC.decode257(z['W'][j], z['W'].shape[-1])
        rec = dict(case_id=cid, time_tag=TIME_TAG, scheme='CN', dt=float(R['setup']['dt']), requested=(coh, int(c)))
        s = PS.score(f, rec, 'ST')
        assert s['kind'] == 'ST' and s['metric_id'] == PIN['metric_id'] and s['store_version'] == PIN['e0_store_version']
        out[cid] = dict(e=float(s['e']), eps_ref=float(s['eps_ref']), eligible=bool(s['eligible']), arm=name,
                        rule=R['q1'][name]['rule'], kind=s['kind'], metric_id=s['metric_id'],
                        reference_sha256=s['reference_sha256'], store_version=s['store_version'])
    return out


def errors(sc, common, valid):
    """Worst / median over the common eligible set; an invalid run on an eligible case -> +inf at that case."""
    e = {c: (sc[c]['e'] if valid[c] else float('inf')) for c in common}
    worst_c = max(e, key=lambda c: e[c])
    return dict(worst=e[worst_c], worst_case=worst_c, median=float(np.median(list(e.values()))), n=len(e))


def no_worse(x, y, scx, scy):
    """A1.2: worst(X) - worst(Y) <= 2 eps*, eps* = max(eps_ref at X's worst case, at Y's worst case)."""
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


# ---------------------------------------------------------------------------------------------- main ----

def main():
    run = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / 'analysis.json'
    R = json.loads((run / 'result.json').read_text())
    cfg = R['config']
    su = R['setup']
    assert su['ckpt_sha256'] == fsha(NC.CKPT) and su['rotation_sha256'] == fsha(NC.ROT), 'decoder identity differs from job'
    production = cfg.get('cases') == PROD['cases'] and cfg.get('q2_cases') == PROD['q2_cases'] and \
        int(su['L']) == PROD['mesh'] and int(su['Rp']) == PROD['Rp'] and abs(su['dt'] - PROD['dt']) < 1e-15
    Rp = int(su['Rp'])
    TB = Tables(run, Rp, int(su['L']))
    A = dict(source=str(run), production=bool(production), q1={}, q2={}, scores={}, arms={}, controls={}, pin=PIN)
    try:
        z = np.load(run / 'arrays' / 'q1_R.npz')
        PS.score(NC.decode257(z['W'][0], Rp), dict(case_id='dev6|1', time_tag=TIME_TAG, scheme='CN', dt=.005,
                                                   requested=('dev6', 0)), 'ST')
        A['controls']['C8'] = dict(fired=False)
    except PS.IdentityError as e:
        A['controls']['C8'] = dict(fired=True, msg=str(e))
    prod = [n for n in R['q1'] if not n.startswith('C4')]
    for name in prod:
        A['q1'][name] = offline_q1(run, name, R)
        A['scores'][name] = score_arm(run, name, R)
    for name in R['q2']:
        A['q2'][name] = offline_q2(run, name, R, TB, rho_check=not name.startswith(('C1', 'C2', 'P6')))
    for name in A['q2']:                                       # parent-convergence downgrade (A1.4), per reading
        fam = name.split('_k')[0]
        if fam in FIT_FAMS:
            par = 'R' if fam in ('F3a', 'F3b') else 'F2_gauss128'
            for key in ('label', 'label_projected'):
                if A['q2'][name][key] == 'pass' and A['q2'].get(par, {}).get(key) != 'pass':
                    A['q2'][name][key] = 'unverified (parent not converged)'
    for name, v in A['q2'].items():
        v['agrees_job'] = bool(v['label'] == v['job_label'] and v['label_projected'] == v['job_label_projected'])
    expect = [f'dev6|{c}' for c in range(6)] + [f'val32|{c}' for c in range(32)]
    pop = expect if production else A['q1']['R']['case_ids']
    eligible = [c for c in pop if A['scores']['R'].get(c, {}).get('eligible')]
    A['common_cases'] = eligible
    A['population_ok'] = {n: A['q1'][n]['case_ids'] == pop for n in prod}
    tim = R.get('timing', {})
    for n in prod:
        q2 = A['q2'].get(n, {})
        dis = bool(q2.get('disagreements')) or bool(A['q1'][n]['disagreements'])
        a = dict(q1=A['q1'][n]['label'], q1_demonstrated=A['q1'][n]['demonstrated'], q2=q2.get('label', 'not run'),
                 q2_projected=q2.get('label_projected', 'not run'), m=R['q1'][n]['m'], rule=R['q1'][n]['rule'],
                 proj=R['q1'][n]['proj'], population_ok=A['population_ok'][n], offline_disagreement=dis)
        ok = a['q1'] == 'pass' and a['population_ok'] and not dis
        a['qualified'] = bool(ok and a['q2'] == 'pass')
        a['qualified_projected'] = bool(ok and a['q2_projected'] == 'pass')
        valid = {c: R['q1'][n]['rows'][c]['valid'] for c in A['q1'][n]['case_ids']}
        a['errors'] = errors(A['scores'][n], eligible, valid) if a['population_ok'] else None
        if n in tim:
            t = tim[n]
            a['timing'] = {k: t[k] for k in ('e2e_median', 'evo_median', 'e2e_ci', 'evo_ci', 'e2e_ratio_of_medians',
                                             'evo_ratio_of_medians', 'timed_valid_all', 'timed_vs_scored_max',
                                             'evo_vs_fused_max', 'device', 'reps', 'burn')}
            a['timing']['outputs_tied'] = bool(t['timed_valid_all'] and np.isfinite(t['timed_vs_scored_max'])
                                               and t['timed_vs_scored_max'] <= 1e-12 and t['evo_vs_fused_max'] <= 1e-12)
        A['arms'][n] = a
    ref = A['arms']['R']
    for n, a in A['arms'].items():
        if n == 'R' or a['errors'] is None or ref['errors'] is None:
            continue
        nw, eps = no_worse(a['errors'], ref['errors'], A['scores'][n], A['scores']['R'])
        a['vs_R'] = dict(no_worse=nw, eps_star=eps, counts=per_case_counts(A['scores'][n], A['scores']['R'], eligible))
        if 'timing' in a and a['timing']['outputs_tied']:
            t = a['timing']
            a['vs_R'].update(faster=bool(t['e2e_median'] > 1 and t['e2e_ci'][0] > 1),
                             gain15=bool(t['e2e_median'] >= GAIN_F3 and t['e2e_ci'][0] > 1))
    if 'AA' in tim:
        A['C6'] = dict(e2e_median=tim['AA']['e2e_median'], fired=bool(.9 <= tim['AA']['e2e_median'] <= 1.1))
    A['stop'] = stop_rules(R, A, binding=True)
    A['stop_projected'] = stop_rules(R, A, binding=False)
    ms = R.get('F2', {}).get('first_qualified')
    if ms and f'F2_{ms}' in A['arms']:                         # F2+F3 against F2(m*) (descriptive, A2.3)
        b = A['arms'][f'F2_{ms}']
        for n, a in A['arms'].items():
            if n.startswith('F2F3') and a['errors'] and b['errors']:
                nw, eps = no_worse(a['errors'], b['errors'], A['scores'][n], A['scores'][f'F2_{ms}'])
                a['vs_F2mstar'] = dict(no_worse=nw, eps_star=eps)
    if not production:
        A['stop'] = {k: f'NOT A VERDICT (non-production population): {v}' for k, v in A['stop'].items()}
        A['stop_projected'] = {k: f'NOT A VERDICT (non-production population): {v}' for k, v in A['stop_projected'].items()}
    out.write_text(json.dumps(A, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(json.dumps(dict(production=production, stop=A['stop'], stop_projected=A['stop_projected']), indent=1, default=str))


def stop_rules(R, A, binding=True):
    """A2.3 / A3.1 / A4.1 outcome precedence. binding=False: the non-binding Q2-projected reading."""
    key = 'qualified' if binding else 'qualified_projected'
    lab2 = 'label' if binding else 'label_projected'
    ref = A['arms']['R']
    if not ref[key]:
        return dict(F2='undecided: reference unqualified', F3='undecided: reference unqualified')
    out = {}
    F2 = R.get('F2')
    if F2 is None or 'candidates' not in F2:
        out['F2'] = 'fired (incomplete evidence): F2 not run / not built'
    else:
        mstar, unresolved = None, False
        for q in F2['candidates']:
            n = f'F2_{q}'
            l2 = A['q2'][n][lab2]
            arm = A['arms'].get(n)
            if l2 == 'pass' and arm is not None and arm[key]:
                mstar = q
                break
            if l2 == 'fail' or (l2 == 'pass' and arm is not None and arm['q1'] != 'pass' and arm['q1_demonstrated']):
                continue                                         # resolved: demonstrated failure
            unresolved = True                                    # unverified Q2, or Q1 not evaluated / not demonstrated
        out['F2_mstar'] = mstar
        if mstar is None:
            out['F2'] = 'fired (incomplete evidence)' if unresolved else 'fired (tested failure): no qualified candidate'
        elif unresolved:
            out['F2'] = f'fired (incomplete evidence): qualified rule exists at {mstar} (upper bound on m*)'
        else:
            v = A['arms'][f'F2_{mstar}'].get('vs_R', {})
            if 'faster' not in v or 'no_worse' not in v:
                out['F2'] = f'fired (incomplete evidence): m* = {mstar}, accuracy or tied timing missing'
            elif v['no_worse'] and v['faster']:
                out['F2'] = f'not fired: m* = {mstar}'
            else:
                out['F2'] = f'fired (tested failure): m* = {mstar}, no_worse={v["no_worse"]}, faster={v["faster"]}'
    cands = [f'{f}_k{k}' for f in ('F3a', 'F3b') for k in (4, 8, 16)]
    ok, unresolved, detail = [], False, {}
    for n in cands:
        a = A['arms'].get(n)
        if a is None:
            unresolved = True
            detail[n] = 'not built / not run (unresolved)'
            continue
        v = a.get('vs_R', {})
        l2 = A['q2'][n][lab2]
        detail[n] = f'q2={l2} q1={a["q1"]} no_worse={v.get("no_worse")} gain15={v.get("gain15")}'
        if a[key] and v.get('no_worse') and v.get('gain15'):
            ok.append(n)
            continue
        demonstrated = (l2 == 'fail') or (a['q1'] != 'pass' and a['q1_demonstrated']) or \
            (a[key] and 'gain15' in v and 'no_worse' in v)
        if not demonstrated:
            unresolved = True
            detail[n] += ' (unresolved)'
    out['F3'] = ('not fired: ' + ', '.join(ok)) if ok else ('fired (incomplete evidence)' if unresolved else 'fired (tested failure)')
    out['F3_detail'] = detail
    return out


if __name__ == '__main__':
    main()
