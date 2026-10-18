"""CPU regression tests of the timing-smoke acceptance (code audit 4): a complete synthetic passing fixture built from the
committed configs/smk2time.json, and one mutation per confirmed counterexample, each of which must be rejected.

    python test_accept.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import smoke_accept as SA

HERE = Path(__file__).resolve().parent


def passing():
    cfg = json.loads((HERE / 'configs' / 'smk2time.json').read_text())
    cases = [f'{a}|{b}' for a, b in cfg['cases']]
    cert = [f'{a}|{b}' for a, b in cfg['cert_cases']]
    labels = [SA.lab_(sp) for sp in cfg['subjects']]
    allb = [SA.lab_(cfg['A'])] + labels
    r = dict(complete=True, subjects={}, equivalence=[], invocations=[], fused=[], doubled=[])
    for lb in allb:
        qp = any(SA.lab_(sp) == lb and sp.get('qp') for sp in cfg['subjects'])
        r['subjects'][lb] = dict(certification=[dict(case=c, valid=True, gcert_ok=True, dist_tight=1e-6, dist_tight_tighter=1e-8,
                                                     tight_valid=True, tighter_valid=True, tight_verified=True, tighter_verified=False,
                                                     **(dict(dist_qp=1e-5, dist_qp_qpp=1e-6, qp_valid=True, qpp_valid=True) if qp else {}))
                                                for c in cert])
        r['equivalence'].append(dict(subject=lb, fields_rel=0., same_iterations=True, passed=True))
    for lb in labels:
        ok = lb != SA.BAD
        for c in cases:
            for k in range(cfg['reps'][c.split('|')[0]]):
                att = dict(B=lb, case=c, rep=k, attempt=0, tA1=1., tA2=1., t_init=.1, t_evolve=.5, t_decode=.1, drift=1.,
                           ratio=.5, hashes_ok=ok)
                r['invocations'].append(dict(att, all_hashes_ok=ok, valid=ok, retries=0, drifts=[1.], drift_ok=True,
                                             attempts=[dict(att)]))
                fat = dict(tA1=1., tA2=1., t_fused=.7, drift=1., hash_ok=ok)
                r['fused'].append(dict(B=lb, case=c, rep=k, tA1=1., tA2=1., t_fused=.7, ratio=.7, drift=1., hash_ok=ok,
                                       all_hashes_ok=ok, retries=0, attempts=[fat], valid=ok))
    for sp in cfg['doubled']:
        for a, b in sp['cases']:
            for k in range(3):
                at = dict(t_single1=1., t_double=2., t_single2=1., drift=1., identity_ok=True, same_iterations=True)
                r['doubled'].append(dict(subject=SA.lab_(sp), case=f'{a}|{b}', rep=k, t_single1=1., t_double=2., t_single2=1.,
                                         ratio=2., drift=1., drift_ok=True, same_iterations=True, identity_ok=True, attempts=[at]))
    pc = json.loads((HERE / 'configs' / 'smk2pilot.json').read_text())
    kf = {}
    for h in pc['heads']:
        kf.setdefault(int(h.get('K', 16)), h['name'])
    big = kf[max(kf)]
    names = (['bank_mesh', 'bank_1024_for_A', 'A', 'heads', 'linear', 'successor_check_tables_on_largest_M', 'inputs_resident']
             + [f'rollout_{n}' for n in kf.values()] + [f"rollout_L{pc['lin'][-1]}"]
             + [f'rollout_{big}_tight', f'rollout_{big}_tighter', 'split_programs', 'rollout_A'])
    pl = dict(stages=[dict(stage=n, mem=dict(peak=40e9 + i * 1e8)) for i, n in enumerate(names)], within_bar=True, complete=True)
    pl['peak_bytes'] = max(x['mem']['peak'] for x in pl['stages'])
    return r, cfg, pl


def main():
    r, cfg, pl = passing()
    base = SA.accept_timing(r, cfg, pl)
    assert base['all_controls_pass'], base
    print('ok  complete passing fixture accepted')
    muts = {
        'no equivalence records': lambda r, pl: r.update(equivalence=[]),
        'empty certification lists': lambda r, pl: [v.update(certification=[]) for v in r['subjects'].values()],
        'only one split invocation + fault': lambda r, pl: r.update(invocations=[x for x in r['invocations'] if x['B'] == SA.BAD][:1]
                                                                   + [x for x in r['invocations'] if x['B'] != SA.BAD][:1]),
        'one of three doubled repetitions': lambda r, pl: r.update(doubled=r['doubled'][:1]),
        'timing complete=False': lambda r, pl: r.update(complete=False),
        'fused ratio inconsistent with times': lambda r, pl: [x.update(ratio=123.) for x in r['fused'] if x['valid']],
        'pilot stage peak NaN': lambda r, pl: pl['stages'][3]['mem'].update(peak=float('nan')),
        'pilot over the bar with within_bar=True': lambda r, pl: (pl['stages'][-1]['mem'].update(peak=115e9), pl.update(peak_bytes=115e9)),
        'pilot stage missing': lambda r, pl: pl.update(stages=pl['stages'][:-2] + pl['stages'][-1:]),
        'duplicate equivalence record': lambda r, pl: r['equivalence'].append(dict(r['equivalence'][0])),
        'doubled drift inconsistent': lambda r, pl: r['doubled'][0].update(drift=1.05),
        'identity failure on a normal subject (retried)': lambda r, pl: next(x for x in r['invocations'] if x['B'] != SA.BAD)['attempts'].insert(
            0, dict(next(x for x in r['invocations'] if x['B'] != SA.BAD)['attempts'][0], hashes_ok=False, attempt=-1)),
        'injected fault fails on drift only': lambda r, pl: [x.update(all_hashes_ok=True, hashes_ok=True, hash_ok=True, drift=1.5, tA2=1.5,
                                                                     attempts=[dict(a, hashes_ok=True, hash_ok=True) for a in x['attempts']])
                                                            for x in r['invocations'] + r['fused'] if x['B'] == SA.BAD],
        'missing CN-R subject': lambda r, pl: [r['subjects'].pop(k) for k in list(r['subjects']) if k.endswith('CNR') and 'Hinc' in k],
        'fused ratios NaN': lambda r, pl: [x.update(ratio=float('nan')) for x in r['fused']],
        'a fused case missing for one subject': lambda r, pl: r.update(fused=[x for x in r['fused']
                                                                             if not (x['B'] == '1024|L384|gauss128|BE' and x['case'] == 'val32|0')]),
        'split record differs from its final attempt': lambda r, pl: r['invocations'][0].update(t_evolve=.6, ratio=.6),
        'non-finite time in an invalid fused repeat': lambda r, pl: next(x for x in r['fused'] if x['B'] == SA.BAD)['attempts'][0].update(t_fused=float('nan')),
        'certification lacks successor fields': lambda r, pl: [x.pop('dist_qp', None) for v in r['subjects'].values() for x in v['certification']],
        'doubled attempt identity failure hidden': lambda r, pl: r['doubled'][0]['attempts'][0].update(identity_ok=False),
        'doubled attempt time NaN': lambda r, pl: r['doubled'][1]['attempts'][0].update(t_double=float('nan')),
        'fault not injected': lambda r, pl: [x.update(valid=True, all_hashes_ok=True, hash_ok=True, hashes_ok=True,
                                                      attempts=[dict(a, hashes_ok=True, hash_ok=True) for a in x['attempts']])
                                             for x in r['invocations'] + r['fused']],
    }
    for name, f in muts.items():
        r2, pl2 = copy.deepcopy(r), copy.deepcopy(pl)
        f(r2, pl2)
        out = SA.accept_timing(r2, cfg, pl2)
        assert not out['all_controls_pass'], f'mutation accepted: {name}'
        print(f'ok  rejected: {name}')
    smk1_tests()
    combined_gate_tests()
    print('test_accept: all checks passed')


def smk1_tests():
    # rule trails: a selected group (rung 0 fails rollout, rung 1 passes everything) and an exhausted group
    pop = [f'dev6|{i}' for i in range(6)] + [f'val32|{i}' for i in range(10)]
    chk = pop[:6]
    ladder = ['gauss16', 'gauss24', 'gauss32']
    g = dict(name='G', min_pass=1, arms=[dict(name='a')])

    def arm(passing):
        if not passing:
            return dict(dist=[1e-3] * 16, valid_q=[True] * 16, valid_qp=[True] * 16, dist_check=[], valid_qpp=[], rho_max=None,
                        rollout_pass=False, rho_pass=False, check_pass=False, passed=False, max_dist=1e-3)
        return dict(dist=[1e-5] * 16, valid_q=[True] * 16, valid_qp=[True] * 16, dist_check=[1e-5] * 6, valid_qpp=[True] * 6,
                    rho_max=.01, rollout_pass=True, rho_pass=True, check_pass=True, passed=True, max_dist=1e-5)
    rung = lambda i, ok: dict(rule=ladder[i], successor=f'gauss{[24, 32, 48][i]}', check=f'gauss{[32, 48, 64][i]}',
                              arms=dict(a=arm(ok)), n_pass=int(ok), min_pass=1, group_pass=ok)
    sel = dict(population=pop, check_population=chk, trail=[rung(0, False), rung(1, True)], status='selected',
               chosen=dict(main='gauss24', qp='gauss32', qpp='gauss48'))
    exh = dict(population=pop, check_population=chk, trail=[rung(i, False) for i in range(3)],
               status='not quadrature-qualified (ladder exhausted)', chosen=None)
    assert SA.rule_trail_ok(g, sel, ladder, pop, chk) and SA.rule_trail_ok(g, exh, ladder, pop, chk)
    print('ok  rule trails: selected and exhausted accepted')
    import copy as cp
    bad = {
        'population differs': lambda v: v.update(population=pop[:15]),
        'arm missing at a rung': lambda v: v['trail'][0]['arms'].clear(),
        'rollout flag contradicts distances': lambda v: v['trail'][0]['arms']['a'].update(rollout_pass=True),
        'check evidence missing': lambda v: v['trail'][1]['arms']['a'].update(dist_check=[]),
        'selected rung not first passing': lambda v: v['trail'].insert(1, rung(1, True)),
        'chosen successor wrong': lambda v: v['chosen'].update(qp='gauss48'),
        'claimed exhaustion without full trail': lambda v: (v.update(status='not quadrature-qualified (ladder exhausted)', chosen=None),
                                                            v['trail'].pop()),
    }
    for name, f in bad.items():
        v = cp.deepcopy(sel)
        f(v)
        assert not SA.rule_trail_ok(g, v, ladder, pop, chk), name
        print(f'ok  rule trail rejected: {name}')
    # accuracy rows: configured evidence required
    cfg = json.loads((HERE / 'configs' / 'smk1acc.json').read_text())
    rows = []
    exp = lambda cs: ([f'dev6|{i}' for i in range(6)] + [f'val32|{i}' for i in range(32)]) if cs == 'all38' else \
        ([f'dev6|{i}' for i in range(6)] + [f'val32|{i}' for i in range(10)]) if cs == 'sub16' else [f'{a}|{b}' for a, b in cs]
    for a in cfg['arms']:
        for sc in a['schemes']:
            for c in exp(a['cases']):
                r = dict(arm=a['name'], scheme=sc, case=c, dist_tight=1e-6, seconds=1., tight_valid=True, tight_verified=False,
                         gcert_ok=True, valid=True)
                if c.startswith('dev6'):
                    r.update(dist_tight_tighter=1e-8, tighter_verified=False, tighter_valid=True)
                if c in exp(a['q2_cases']):
                    r.update(dist_qp=1e-5, rho_max=.01, qp_valid=True)
                    if c.startswith('dev6'):
                        r.update(dist_qp_qpp=1e-6, qpp_valid=True)
                rows.append(r)
    assert SA.acc_rows_ok(cfg, rows)
    print('ok  accuracy rows accepted')
    for name, f in {'a row missing': lambda R: R.pop(0), 'successor metric missing': lambda R: [r.pop('dist_qp', None) for r in R],
                    'anchor distance NaN': lambda R: R[3].update(dist_tight=float('nan')),
                    'duplicate row': lambda R: R.append(dict(R[0])),
                    'tighter_valid missing': lambda R: [r.pop('tighter_valid', None) for r in R]}.items():
        R = cp.deepcopy(rows)
        f(R)
        assert not SA.acc_rows_ok(cfg, R), name
        print(f'ok  accuracy rows rejected: {name}')



def combined_gate_tests():
    """Structural (smoke_accept.rule_trail_ok) + numerical (hscore.rule_verify) gates together on synthetic archives built
    from a real bank-free linear arm: passing, failed-then-passing (incl. null rho / null distances / non-finite w0) and
    exhausted trails."""
    import tempfile
    import numpy as np
    import hscore as HSC
    import npcheck as NC
    pop, chk = ['dev6|0', 'dev6|1'], ['dev6|0']
    ladder = ['gauss16', 'gauss24', 'gauss32']
    dim, M = 4, 16
    rng = np.random.default_rng(3)
    base = rng.normal(size=(6, dim)) * .3

    def arch(tmp, gname, rows):
        keys, W, valid, n0, ex, sg, sr, icr, icg, ff, w0 = [], [], [], [], [], [], [], [], [], [], []
        for (rule, case), (Wv, ok, w0v) in rows.items():
            keys.append(f'a|{rule}|{case}'); W.append(Wv); valid.append(ok); n0.append(1.)
            ex.append(np.zeros(5, int)); sg.append(np.zeros(50)); sr.append(np.full(50, 4)); icr.append(-1); icg.append(np.nan)
            ff.append(True); w0.append(w0v)
        np.savez(Path(tmp) / f'SOL_{gname}.npz', keys=np.array(keys), W=np.array(W), valid=np.array(valid), n0=np.array(n0),
                 exits=np.array(ex), step_gn=np.array(sg), step_reason=np.array(sr), ic_reason=np.array(icr), ic_gn=np.array(icg),
                 fields_finite=np.array(ff), w0=np.array(w0))

    def arm_entry(Wq, Wp, Wpp, okq, rho_val):
        d = [NC.fdist(NC.decode257(Wq[c], dim), NC.decode257(Wp[c], dim), 1.) for c in pop]
        roll = bool(okq and np.isfinite(d).all() and max(d) <= 2.5e-4)
        rho = roll and rho_val is not None and rho_val <= .116
        dc = [NC.fdist(NC.decode257(Wp[c], dim), NC.decode257(Wpp[c], dim), 1.) for c in chk] if rho else []
        ck = bool(rho and max(dc) <= 2.5e-4)
        return dict(dist=[None if not np.isfinite(x) else x for x in d], valid_q=[okq] * 2, valid_qp=[True] * 2,
                    dist_check=dc, valid_qpp=[True] * len(dc), rho_max=rho_val if roll else None, rollout_pass=roll,
                    rho_pass=bool(rho), check_pass=ck, passed=bool(roll and rho and ck), max_dist=None)
    # identical solutions at every rule -> distances 0 -> rollout passes; rho is recomputed from tables by rule_verify, so
    # we let the job value be the recomputed one (computed below)
    g = dict(name='G', min_pass=1, arms=[dict(name='a', kind='lin', Rp=dim, M=M)])
    with tempfile.TemporaryDirectory() as tmp:
        same = {c: base for c in pop}
        bad = {c: base * np.nan for c in pop}
        rows = {}
        for rl in ladder + ['gauss48', 'gauss64']:
            for c in pop:
                rows[(rl, c)] = (bad[c] if rl == 'gauss16' else base, rl != 'gauss16', np.zeros(dim) if rl != 'gauss16' else np.full(dim, np.nan))
        arch(tmp, 'G', rows)
        tq, tp = HSC.tables('gauss24', dim, M, 1024), HSC.tables('gauss32', dim, M, 1024)
        rr = max(NC.rho(ck, tq, tp) for ck in base)
        t0 = dict(rule='gauss16', successor='gauss24', check='gauss32', arms=dict(a=arm_entry(bad, same, same, False, None)),
                  n_pass=0, min_pass=1, group_pass=False)
        e1 = arm_entry(same, same, same, True, rr)
        t1 = dict(rule='gauss24', successor='gauss32', check='gauss48', arms=dict(a=e1), n_pass=int(e1['passed']), min_pass=1,
                  group_pass=e1['passed'])
        v = dict(population=pop, check_population=chk, trail=[t0, t1], status='selected' if e1['passed'] else 'x',
                 chosen=dict(main='gauss24', qp='gauss32', qpp='gauss48'))
        res = dict(mesh=1024, groups=dict(G=v))
        st = SA.rule_trail_ok(g, v, ladder, pop, chk)
        nv = HSC.rule_verify(res, tmp, dict(groups=[g]))['G']
        assert e1['passed'] and st and nv['ok'], (st, nv)
        print('ok  combined gate: failed (null distances, non-finite w0) then passing trail accepted by both checkers')
        # null rho on a rollout-passing rung recorded as a failure is accepted structurally
        e_nr = dict(e1, rho_max=None, rho_pass=False, check_pass=False, passed=False, dist_check=[], valid_qpp=[])
        v2 = dict(v, trail=[t0, dict(t1, arms=dict(a=e_nr), n_pass=0, group_pass=False)], status='not quadrature-qualified (ladder exhausted)',
                  chosen=None)
        assert SA.rule_trail_ok(g, v2, ladder[:2], pop, chk)
        print('ok  structural gate accepts a recorded null-rho failure and an exhausted trail')


if __name__ == '__main__':
    main()
