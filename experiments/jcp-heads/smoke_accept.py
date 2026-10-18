"""jcp-heads smoke acceptance (DESIGN 7, A1.3, A1.7, A2.6; CPU only). Acceptance requires BOTH the job's own per-row
pass flags and this script's checks; where the raw arrays allow it (budget-0 gamma, zero-update, decode, coefficients)
the predicate is recomputed independently here, otherwise the job's per-row values are checked for finiteness, coverage
and bars. Written to checks/smoke-accept-<attempt>.json.

    python smoke_accept.py smk1        (training, accuracy controls, rule-ladder probe; 1024^2)
    python smoke_accept.py smk2        (timing controls at 1024^2, 4096^2 memory pilot)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import npcheck as NC
import nphead as NH

HERE = Path(__file__).resolve().parent
INC_RECON, INC_ORACLE = 0.007507219122474576, 0.005200570538097191          # hfit_dn256b_full.json arms.mid
INC = HERE.parents[0] / 'separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
BAR_Q1, BAR_Q2, BAR_RHO, GAMMA = 2.5e-4, 2.5e-4, .116, 1e-3


def out_dir(att, drv, stem):
    return HERE / 'runs' / att / 'archive' / 'output' / f'{drv}_{stem}'


def rel(a, b):
    return abs(a - b) / abs(b)


def rows_ok(rows, key, bar, n):
    """Every row present (n), finite and within bar; NaN can never hide behind max()."""
    v = np.array([r[key] for r in rows], float)
    return bool(len(v) == n and np.isfinite(v).all() and np.all(v <= bar)), (float(np.max(v)) if len(v) else None)


def smk1():
    res = {}
    # ---- training pipeline (DESIGN 2.3, A1.7, A2.2)
    tr = json.loads((out_dir('smk1', 'htrain', 'smk1train') / 'result.json').read_text())
    ident = tr['arms']['identity']
    res['train'] = dict(
        whitening=bool(tr['gates']['whitening_round_trip'] < 1e-10 and tr['gates']['q_equals_LT_h'] < 1e-10),
        disjointness=bool(all(tr['gates']['disjointness'][c]['min_maxabs_distance'] > 1e-9 for c in ('dev6', 'val32'))),
        identity_recon_rel=rel(ident['recon_train']['mean'], INC_RECON),
        identity_oracle_rel=rel(ident['oracle_test']['mean'], INC_ORACLE),
        complete=bool(tr['complete']), npz_sha256=tr['config']['npz_sha256'])
    res['train']['identity_pass'] = bool(res['train']['identity_recon_rel'] <= 1e-9 and res['train']['identity_oracle_rel'] <= 1e-3)
    # compact archive defines the same head as the emitted checkpoint (checked in-job, A5); loadable here by NumPy
    comp = {}
    for name in ('smk_k4_s0', 'smk_k32_s0'):
        cz = out_dir('smk1', 'htrain', 'smk1train') / f'head_{name}.npz'
        hc = NH.Head(cz)
        comp[name] = dict(k=hc.K, recon=tr['arms'][name]['recon_train']['mean'], oracle=tr['arms'][name]['oracle_test']['mean'],
                          compact_vs_emitted=tr['arms'][name]['compact_vs_emitted'],
                          ok=bool(hc.K == int(name.split('_k')[1].split('_')[0]) and tr['arms'][name]['compact_vs_emitted'] == 0.))
    res['train']['short_heads'] = comp
    # ---- accuracy job: in-job controls, recomputed where possible
    acc_dir = out_dir('smk1', 'hacc', 'smk1acc')
    acc = json.loads((acc_dir / 'result.json').read_text())
    c = acc['controls']
    ok, w = rows_ok(c['repro_head']['rows'], 'rel', 1e-10, 4)
    res['repro_head'] = dict(worst=w, passed=bool(ok and c['repro_head']['passed']),
                             fixtures=sorted({(r['rule'], r['case']) for r in c['repro_head']['rows']}))
    ok, w = rows_ok(c['repro_lin']['rows'], 'rel', 1e-10, 12)
    sch = sorted({r['scheme'] for r in c['repro_lin']['rows']})
    res['repro_lin'] = dict(worst=w, schemes=sch, passed=bool(ok and sch == ['BE', 'CNR'] and c['repro_lin']['passed']))
    jac = {k: v['jacobian'] for k, v in acc['arms'].items() if 'jacobian' in v}
    jok = {k: rows_ok(v['rows'], 'rel', 1e-10, 4) for k, v in jac.items()}
    res['jacobian'] = dict(per_head={k: v[1] for k, v in jok.items()},
                           passed=bool(sorted(jac) == ['Hinc', 'Hk32', 'Hk4'] and all(v[0] for v in jok.values())
                                       and all(v['passed'] for v in jac.values())))
    res['wrong_mesh'] = dict(passed=bool(c['wrong_mesh']['passed']), raised=c['wrong_mesh']['raised'])
    # budget-0 Q1 negative: in-job pooled fraction > 1 %, and an independent NumPy gamma > 1e-3 on > 3 steps
    b0 = np.load(acc_dir / 'CTRL_budget0.npz')
    head = NH.Head(INC)
    kx, ky = None, None
    Rhinc = np.load(acc_dir / 'R_Hinc_BE.npz')
    t = NC.tables(*NC.gauss_rule(int(str(Rhinc['rule'])[5:])), 512, *_modes(Rhinc), 1024)
    gnp = NH.gammas(head, b0['states'][0][:6], b0['A'], b0['lam'], t, 'BE', .005, float(b0['nu'][0]))
    res['budget0'] = dict(pooled_fraction=c['budget0']['pooled_fraction'], numpy_gamma_first5=gnp.tolist(),
                          numpy_over_bar=int(np.sum(gnp > GAMMA)))
    res['budget0']['passed'] = bool(res['budget0']['pooled_fraction'] > .01 and res['budget0']['numpy_over_bar'] > 3)
    # zero-update: constant trajectory vs the tight anchor, bounded below via the prod fields (triangle inequality)
    j0 = [str(x) for x in Rhinc['case_ids']].index('dev6|0')
    f = Rhinc['f257'][j0]
    n0 = float(Rhinc['n0'][j0])
    d_const_prod = max(float(np.linalg.norm(f[k] - f[0])) / n0 for k in range(1, 6))
    row0 = [r for r in acc['rows'] if r['arm'] == 'Hinc' and r['scheme'] == 'BE' and r['case'] == 'dev6|0'][0]
    res['zero_update'] = dict(lower_bound=d_const_prod - row0['dist_tight'], passed=bool(d_const_prod - row0['dist_tight'] > BAR_Q1))
    # under-resolved rule: rollout bar AND rho bar fail, finer solves valid
    ur = c['under_resolved']
    res['under_resolved'] = dict(fails_rollout=bool(ur['fails_rollout_bar']), rho_max=ur['rho_max'],
                                 successors_valid=bool(ur['successors_valid']))
    res['under_resolved']['passed'] = bool(ur['fails_rollout_bar'] and ur['rho_max'] > BAR_RHO and ur['successors_valid'])
    # independent decode of the saved head coefficients and of the latent states (NumPy head) on two cases
    dec = []
    for j in range(2):
        C = np.array([head.c(w) for w in Rhinc['W'][j]])
        dec.append(dict(coef_rel=float(np.max(np.abs(C - Rhinc['C'][j])) / np.max(np.abs(C))),
                        field_rel=float(np.max(np.abs(NC.decode257(C, 512) - Rhinc['f257'][j])) / np.max(np.abs(Rhinc['f257'][j])))))
    res['independent_decode'] = dict(rows=dec, passed=bool(all(d['coef_rel'] <= 1e-9 and d['field_rel'] <= 1e-9 for d in dec)))
    # G-cert and Q1/Q2 summaries of the real-size configuration (reported, no required outcome)
    rows = [r for r in acc['rows'] if r['arm'] == 'Hinc']
    res['hinc_real_size'] = {sc: dict(gcert_all=all(r['gcert_ok'] for r in rows if r['scheme'] == sc),
                                      valid=sum(r['valid'] for r in rows if r['scheme'] == sc),
                                      n=sum(1 for r in rows if r['scheme'] == sc),
                                      noncert=sum(r['noncertified'] for r in rows if r['scheme'] == sc),
                                      max_dist_tight=max(r['dist_tight'] for r in rows if r['scheme'] == sc),
                                      tight_verified=sum(bool(r['tight_verified']) for r in rows if r['scheme'] == sc),
                                      tighter_verified=sum(bool(r.get('tighter_verified')) for r in rows if r['scheme'] == sc and 'tighter_verified' in r),
                                      max_dist_qp=max((r['dist_qp'] for r in rows if r['scheme'] == sc and 'dist_qp' in r), default=None),
                                      rho_max=max((r['rho_max'] for r in rows if r['scheme'] == sc and 'rho_max' in r), default=None),
                                      seconds=sum(r['seconds'] for r in rows if r['scheme'] == sc))
                             for sc in ('BE', 'CNR')}
    res['elapsed'] = acc['elapsed']
    # ---- rule ladder probe
    ru = json.loads((out_dir('smk1', 'hrule', 'smk1rule') / 'result.json').read_text())
    res['rule_probe'] = {g: dict(chosen=v['chosen'], status=v['status'], seconds=v['seconds']) for g, v in ru['groups'].items()}
    res['rule_probe_note'] = ('plumbing probe: group Hk4 holds the SAME short head twice (min_pass 2), so it exercises the '
                              'counting path, not independent-seed behaviour; the 2-of-3 threshold is unit-tested in test_h')
    res['gcert'] = dict(passed=bool(all(v['gcert_all'] for v in res['hinc_real_size'].values())
                                    and all(r['gcert_ok'] for r in acc['rows'])))
    res['train']['short_heads_ok'] = bool(all(v['ok'] for v in comp.values()))
    res['all_controls_pass'] = bool(res['train']['identity_pass'] and res['train']['whitening'] and res['train']['disjointness']
                                    and res['train']['short_heads_ok'] and res['gcert']['passed']
                                    and res['repro_head']['passed'] and res['repro_lin']['passed'] and res['jacobian']['passed']
                                    and res['wrong_mesh']['passed'] and res['budget0']['passed'] and res['zero_update']['passed']
                                    and res['under_resolved']['passed'] and res['independent_decode']['passed'])
    return res


def _modes(R):
    """(kx, ky) of a saved configuration from its A's row count: the frozen 1024^2 list prefix (p1common.mode_list)."""
    import p1common as P
    kx, ky = P.mode_list('frozen', 1024)
    M = R['A'].shape[0]
    return kx[:M], ky[:M]


def smk2():
    res = {}
    td = out_dir('smk2', 'htime', 'smk2time')
    r = json.loads((td / 'result.json').read_text())
    res['equivalence'] = dict(passed=bool(all(e['passed'] for e in r['equivalence'])), n=len(r['equivalence']))
    bad = '1024|L16|gauss24|BE'
    inv_bad = [x for x in r['invocations'] if x['B'] == bad] + [x for x in r['fused'] if x['B'] == bad]
    res['identity_fault'] = dict(n=len(inv_bad), all_invalid=bool(inv_bad and not any(x['valid'] for x in inv_bad)))
    good = [x for x in r['invocations'] + r['fused'] if x['B'] != bad]
    res['other_valid_fraction'] = float(np.mean([x['valid'] for x in good]))
    # fused coverage, independently of split validity: every non-fault subject x timed case has >= 1 valid fused repeat
    cov = {}
    for lb, v in r['subjects'].items():
        if lb == bad or lb.split('|')[1] == 'A_R384':
            continue
        for c_ in v['timed_case_ids']:
            cov[(lb, c_)] = any(x['valid'] for x in r['fused'] if x['B'] == lb and x['case'] == c_)
    res['fused_coverage'] = dict(missing=[list(k) for k, ok in cov.items() if not ok], n=len(cov), passed=bool(cov and all(cov.values())))
    dbl = r['doubled']
    res['doubled'] = dict(ratios=[x['ratio'] for x in dbl], passed=bool(dbl and all(1.8 <= x['ratio'] <= 2.2 and x['identity_ok']
                                                                                     and x['same_iterations'] and x['drift_ok']
                                                                                     for x in dbl)))
    res['subjects'] = {k: dict(cert=[(c_['valid'], c_['gcert_ok'], c_['tight_verified'], c_['tighter_verified']) for c_ in v['certification']])
                       for k, v in r['subjects'].items()}
    by = {}
    for x in r['fused']:
        if x['valid']:
            by.setdefault(x['B'], []).append(x['ratio'])
    res['fused_median_ratio'] = {k: float(np.median(v)) for k, v in by.items()}
    import timing as TM
    res['panel_validity_without_fault'] = TM.panel_validity([x for x in r['invocations'] if x['B'] != bad])['valid']
    pl = json.loads((out_dir('smk2', 'hpilot', 'smk2pilot') / 'result.json').read_text())
    res['pilot'] = dict(peak_bytes=pl['peak_bytes'], within_bar=bool(pl['within_bar']), stages=[(s_['stage'], s_['mem']['peak'])
                                                                                                 for s_ in pl['stages']])
    res['certification_ok'] = bool(all(all(t_[:2]) for v in res['subjects'].values() for t_ in v['cert']))
    res['all_controls_pass'] = bool(res['equivalence']['passed'] and res['identity_fault']['all_invalid']
                                    and res['doubled']['passed'] and res['pilot']['within_bar']
                                    and res['panel_validity_without_fault'] and res['fused_coverage']['passed']
                                    and res['certification_ok'])
    return res


if __name__ == '__main__':
    att = sys.argv[1]
    out = {'smk1': smk1, 'smk2': smk2}[att]()
    p = HERE / 'checks' / f'smoke-accept-{att}.json'
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps(out, indent=1, default=float)[:4000])
