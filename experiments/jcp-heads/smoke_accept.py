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


BAND = (1 / 1.1, 1.1)
BAD = '1024|L16|gauss24|BE'
PILOT_BAR, PILOT_ALLOW = 120e9, 1.10


def lab_(sp):
    return f"{sp['mesh']}|{sp['name']}|{sp['rule']}|{sp['scheme']}"


def close(a, b, rel_=1e-9):
    return bool(np.isfinite([a, b]).all() and abs(a - b) <= rel_ * abs(b))


def smk1():
    res = {}
    res['steps_complete'] = bool(all((out_dir('smk1', d_, st_) / 'COMPLETE').exists() and
                                     json.loads((out_dir('smk1', d_, st_) / 'result.json').read_text()).get('complete')
                                     for d_, st_ in (('htrain', 'smk1train'), ('hacc', 'smk1acc'), ('hrule', 'smk1rule'))))
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
                          ok=bool(hc.K == int(name.split('_k')[1].split('_')[0]) and tr['arms'][name]['compact_vs_emitted'] == 0.
                                  and np.isfinite([tr['arms'][name]['recon_train']['mean'], tr['arms'][name]['oracle_test']['mean']]).all()))
    res['train']['short_heads'] = comp
    # ---- accuracy job: in-job controls, recomputed where possible
    acc_dir = out_dir('smk1', 'hacc', 'smk1acc')
    acc = json.loads((acc_dir / 'result.json').read_text())
    c = acc['controls']
    cfg = json.loads((HERE / 'configs' / 'smk1acc.json').read_text())          # coverage bound to the committed config
    blk = [b for b in cfg['blocks'] if b['kind'] == 'controls'][0]
    want_h = sorted((rl, cname) for rl, fx in blk['repro_head'].items() for cname in fx)
    got_h = sorted((r['rule'], r['case']) for r in c['repro_head']['rows'])
    ok, w = rows_ok(c['repro_head']['rows'], 'rel', 1e-10, len(want_h))
    res['repro_head'] = dict(worst=w, fixtures=got_h, passed=bool(ok and got_h == want_h and c['repro_head']['passed']))
    want_l = sorted((sc, f'dev6|{i}') for sc in blk['repro_lin']['fixtures'] for i in range(6))
    got_l = sorted((r['scheme'], r['case']) for r in c['repro_lin']['rows'])
    ok, w = rows_ok(c['repro_lin']['rows'], 'rel', 1e-10, len(want_l))
    res['repro_lin'] = dict(worst=w, passed=bool(ok and got_l == want_l and c['repro_lin']['passed']))
    jac = {k: v['jacobian'] for k, v in acc['arms'].items() if 'jacobian' in v}
    want_j = sorted(a['name'] for a in cfg['arms'] if a['kind'] == 'head')
    jok = {k: rows_ok(v['rows'], 'rel', 1e-10, 4) for k, v in jac.items()}
    fix_ok = all(sorted((r['case'], r['step']) for r in v['rows']) == [('dev6|0', 1), ('dev6|0', 25), ('dev6|1', 1), ('dev6|1', 25)]
                 for v in jac.values())
    res['jacobian'] = dict(per_head={k: v[1] for k, v in jok.items()},
                           passed=bool(sorted(jac) == want_j and fix_ok and all(v[0] for v in jok.values())
                                       and all(v['passed'] for v in jac.values())))
    wr = c['wrong_mesh']['raised']
    res['wrong_mesh'] = dict(raised=wr, passed=bool(len(wr) == 3 and all(isinstance(x, str) and x for x in wr) and c['wrong_mesh']['passed']))
    # accuracy coverage: every configured arm x scheme x case has exactly one row
    want_rows = sorted((a['name'], sc, f'{x[0]}|{x[1]}') for a in cfg['arms'] for sc in a['schemes']
                       for x in (([['dev6', i] for i in range(6)] + [['val32', i] for i in range(32)]) if a['cases'] == 'all38' else a['cases']))
    got_rows = sorted((r['arm'], r['scheme'], r['case']) for r in acc['rows'])
    res['accuracy_coverage'] = dict(passed=bool(got_rows == want_rows), n=len(got_rows), expected=len(want_rows))
    # negative controls: coverage of the dev6 fixture cases and summaries recomputed from the rows
    dev6 = [f'dev6|{i}' for i in range(6)]
    b0r, urr = c['budget0']['rows'], c['under_resolved']['rows']
    pooled = sum(x['noncertified'] for x in b0r) / max(1, sum(x['steps'] for x in b0r))
    res['control_coverage'] = dict(budget0=bool(sorted(x['case'] for x in b0r) == dev6 and close(pooled, c['budget0']['pooled_fraction'])),
                                   under_resolved=bool(sorted(x['case'] for x in urr) == dev6
                                                       and bool(c['under_resolved']['fails_rollout_bar']) == any(x['dist'] > 2.5e-4 for x in urr)
                                                       and close(max(max(x['rho']) for x in urr), c['under_resolved']['rho_max'])))
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
    rcfg = json.loads((HERE / 'configs' / 'smk1rule.json').read_text())
    res['rule_coverage'] = bool(sorted(ru['groups']) == sorted(g['name'] for g in rcfg['groups'])
                                and all(v['trail'] and v['status'] in ('selected', 'not quadrature-qualified (ladder exhausted)')
                                        and (v['chosen'] is None) == (v['status'] != 'selected')
                                        and sorted(v['trail'][-1]['arms']) == sorted(a['name'] for a in
                                                                                     [g for g in rcfg['groups'] if g['name'] == k][0]['arms'])
                                        for k, v in ru['groups'].items()))
    res['rule_probe_note'] = ('plumbing probe: group Hk4 holds the SAME short head twice (min_pass 2), so it exercises the '
                              'counting path, not independent-seed behaviour; the 2-of-3 threshold is unit-tested in test_h')
    res['gcert'] = dict(passed=bool(all(v['gcert_all'] for v in res['hinc_real_size'].values())
                                    and all(r['gcert_ok'] for r in acc['rows'])))
    res['train']['short_heads_ok'] = bool(all(v['ok'] for v in comp.values()))
    res['all_controls_pass'] = bool(res['steps_complete'] and res['train']['identity_pass'] and res['train']['whitening'] and res['train']['disjointness']
                                    and res['train']['short_heads_ok'] and res['gcert']['passed']
                                    and res['repro_head']['passed'] and res['repro_lin']['passed'] and res['jacobian']['passed']
                                    and res['wrong_mesh']['passed'] and res['budget0']['passed'] and res['zero_update']['passed']
                                    and res['under_resolved']['passed'] and res['independent_decode']['passed']
                                    and res['accuracy_coverage']['passed'] and res['control_coverage']['budget0']
                                    and res['control_coverage']['under_resolved'] and res['rule_coverage'])
    return res


def _modes(R):
    """(kx, ky) of a saved configuration from its A's row count: the frozen 1024^2 list prefix (p1common.mode_list)."""
    import p1common as P
    kx, ky = P.mode_list('frozen', 1024)
    M = R['A'].shape[0]
    return kx[:M], ky[:M]


def accept_timing(r, cfg, pl):
    """Pure acceptance of a timing smoke (result r, its committed config cfg, pilot result pl). Coverage of every
    configured identity is required (a missing record fails); stored ratios and flags are recomputed from the raw times."""
    res = {}
    cases = sorted(f'{a}|{b}' for a, b in cfg['cases'])
    cert_cases = sorted(f'{a}|{b}' for a, b in cfg['cert_cases'])
    reps = cfg['reps']
    labels = [lab_(sp) for sp in cfg['subjects']]
    all_lab = sorted({lab_(cfg['A'])} | set(labels))
    res['complete'] = bool(r.get('complete'))
    res['subjects_present'] = bool(sorted(r['subjects']) == all_lab)
    eq = {e['subject']: e for e in r['equivalence']}
    res['equivalence_unique'] = bool(len(eq) == len(r['equivalence']))
    res['equivalence'] = dict(passed=bool(res['equivalence_unique'] and sorted(eq) == all_lab and all(e['passed'] and np.isfinite(e['fields_rel'])
                                                                         and e['fields_rel'] <= 1e-12 and e['same_iterations']
                                                                         for e in eq.values())), n=len(eq))
    cert_ok = True
    for lb in all_lab:
        rows = r['subjects'].get(lb, {}).get('certification', [])
        cert_ok &= sorted(x['case'] for x in rows) == cert_cases
        cert_ok &= all(x['valid'] and x['gcert_ok'] and np.isfinite([x['dist_tight'], x['dist_tight_tighter']]).all() for x in rows)
    res['certification_ok'] = bool(cert_ok)
    want = sorted((lb, c, k) for lb in labels for c in cases for k in range(int(reps[c.split('|')[0]])))

    def ident(xs):
        return sorted((x['B'], x['case'], x['rep']) for x in xs)
    res['split_coverage'] = bool(ident(r['invocations']) == want)
    res['fused_coverage_ids'] = bool(ident(r['fused']) == want)

    def att_ok(a):
        """One recorded split attempt: finite positive times, consistent drift and ratio."""
        t = [a['tA1'], a['tA2'], a['t_evolve'], a['t_init'], a['t_decode']]
        return bool(np.isfinite(t).all() and min(t) > 0 and close(a['drift'], a['tA2'] / a['tA1'])
                    and close(a['ratio'], a['t_evolve'] / (.5 * (a['tA1'] + a['tA2']))))

    def split_ok(x):
        atts = x.get('attempts') or []
        return bool(atts and all(att_ok(a) for a in atts) and x['all_hashes_ok'] == all(a['hashes_ok'] for a in atts)
                    and att_ok(x) and x['all_hashes_ok'] and BAND[0] <= x['drift'] <= BAND[1])

    # identity failures are never tolerated outside the injected-fault subject; drift-invalid repeats are (coverage below)
    res['identity_clean'] = bool(all(x['all_hashes_ok'] and all(a['hashes_ok'] for a in x.get('attempts') or [{'hashes_ok': False}])
                                     for x in r['invocations'] if x['B'] != BAD)
                                 and all(x.get('all_hashes_ok', x['hash_ok']) and all(a['hash_ok'] for a in x.get('attempts') or [{'hash_ok': False}])
                                         for x in r['fused'] if x['B'] != BAD))

    def fused_ok(x):
        t = [x['tA1'], x['tA2'], x['t_fused']]
        return bool(np.isfinite(t).all() and min(t) > 0 and close(x['ratio'], x['t_fused'] / (.5 * (x['tA1'] + x['tA2'])))
                    and close(x['drift'], x['tA2'] / x['tA1'])
                    and x['hash_ok'] and x.get('all_hashes_ok', x['hash_ok']) and BAND[0] <= x['tA2'] / x['tA1'] <= BAND[1])
    res['split_flag_agrees'] = bool(all(split_ok(x) == bool(x['valid']) for x in r['invocations']))
    res['fused_valid_flag_agrees'] = bool(all(fused_ok(x) == bool(x['valid']) for x in r['fused']))
    bad_x = [x for x in r['invocations'] + r['fused'] if x['B'] == BAD]
    res['identity_fault'] = dict(n=len(bad_x), all_invalid=bool(bad_x and not any(x['valid'] for x in bad_x)),
                                 failed_on_identity=bool(bad_x and all(not x.get('all_hashes_ok', x.get('hash_ok')) for x in bad_x)))
    cov = {(lb, c): any(fused_ok(x) for x in r['fused'] if x['B'] == lb and x['case'] == c) for lb in labels if lb != BAD
           for c in cases}
    res['fused_coverage'] = dict(missing=[list(k) for k, ok in cov.items() if not ok], passed=bool(cov and all(cov.values())))
    import timing as TM
    res['panel_validity_without_fault'] = TM.panel_validity([x for x in r['invocations'] if x['B'] != BAD])['valid']
    dwant = sorted((lab_(sp), f'{a}|{b}', k) for sp in cfg.get('doubled', []) for a, b in sp['cases'] for k in range(3))
    dbl = r['doubled']
    dok = all(np.isfinite([x['t_single1'], x['t_double'], x['t_single2']]).all() and min(x['t_single1'], x['t_double'], x['t_single2']) > 0
              and close(x['ratio'], x['t_double'] / (.5 * (x['t_single1'] + x['t_single2'])))
              and close(x['drift'], x['t_single2'] / x['t_single1']) and bool(x['drift_ok']) == bool(BAND[0] <= x['drift'] <= BAND[1])
              and 1.8 <= x['ratio'] <= 2.2 and x['identity_ok'] and x['same_iterations'] and BAND[0] <= x['drift'] <= BAND[1]
              for x in dbl)
    res['doubled'] = dict(ratios=[x['ratio'] for x in dbl],
                          passed=bool(sorted((x['subject'], x['case'], x['rep']) for x in dbl) == dwant and dok))
    stages = [s_['mem']['peak'] for s_ in pl['stages']]
    names = [s_['stage'] for s_ in pl['stages']]
    pcfg = json.loads((HERE / 'configs' / 'smk2pilot.json').read_text())
    kfirst = {}
    for h in pcfg['heads']:                          # hpilot rolls out the first head of each distinct K
        kfirst.setdefault(int(h.get('K', 16)), h['name'])
    big = kfirst[max(kfirst)]
    want_st = (['bank_mesh', 'bank_1024_for_A', 'A', 'heads', 'linear', 'successor_check_tables_on_largest_M', 'inputs_resident']
               + [f'rollout_{n}' for n in kfirst.values()] + [f"rollout_L{pcfg['lin'][-1]}"]
               + [f'rollout_{big}_tight', f'rollout_{big}_tighter', 'split_programs', 'rollout_A'])
    res['pilot_stages'] = dict(missing=sorted(set(want_st) - set(names)),
                               passed=bool(set(want_st) <= set(names) and all(np.isfinite(x) and x > 0 for x in stages)))
    peak = max(stages) if stages and all(np.isfinite(x) for x in stages) else float('nan')
    res['pilot'] = dict(peak_bytes=peak, recomputed_within_bar=bool(np.isfinite(peak) and peak > 0 and peak * PILOT_ALLOW <= PILOT_BAR),
                        stored_within_bar=bool(pl.get('within_bar')), stored_peak_agrees=bool(pl.get('peak_bytes') == peak),
                        complete=bool(pl.get('complete')))
    res['pilot_note'] = ('measurement probe (A7): it does not reproduce every allocation of the timing driver; time4096 (first '
                         'in the defer order) is reported "not reached (memory)" if it fails on memory')
    by = {}
    for x in r['fused']:
        if fused_ok(x):
            by.setdefault(x['B'], []).append(x['ratio'])
    res['fused_median_ratio'] = {k: float(np.median(v)) for k, v in by.items()}
    res['all_controls_pass'] = bool(res['complete'] and res['subjects_present'] and res['equivalence']['passed']
                                    and res['identity_clean'] and res['identity_fault']['failed_on_identity']
                                    and res['pilot_stages']['passed']
                                    and res['certification_ok'] and res['split_coverage'] and res['fused_coverage_ids']
                                    and res['split_flag_agrees'] and res['fused_valid_flag_agrees']
                                    and res['identity_fault']['all_invalid'] and res['fused_coverage']['passed']
                                    and res['panel_validity_without_fault'] and res['doubled']['passed']
                                    and res['pilot']['recomputed_within_bar'] and res['pilot']['stored_within_bar']
                                    and res['pilot']['stored_peak_agrees'] and res['pilot']['complete'])
    return res


def smk2():
    r = json.loads((out_dir('smk2', 'htime', 'smk2time') / 'result.json').read_text())
    cfg = json.loads((HERE / 'configs' / 'smk2time.json').read_text())
    pl = json.loads((out_dir('smk2', 'hpilot', 'smk2pilot') / 'result.json').read_text())
    assert (out_dir('smk2', 'htime', 'smk2time') / 'COMPLETE').exists() and (out_dir('smk2', 'hpilot', 'smk2pilot') / 'COMPLETE').exists()
    return accept_timing(r, cfg, pl)


if __name__ == '__main__':
    att = sys.argv[1]
    out = {'smk1': smk1, 'smk2': smk2}[att]()
    p = HERE / 'checks' / f'smoke-accept-{att}.json'
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps(out, indent=1, default=float)[:4000])
