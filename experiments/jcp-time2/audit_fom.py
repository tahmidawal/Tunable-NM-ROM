"""Independent audit of a pulled fomrun job (DESIGN section 9, A1.10, A2.21, A3.5, A9, A10). NumPy only except the
required independent BE re-run. Every inventory is derived from the staged config (authenticated by the staging
provenance). One validation function decides acceptance; the rejection tests feed MUTATED copies of the evidence
through that same function and each must be rejected.

    /home/tahmid/Dev/.venv/bin/python experiments/jcp-time2/audit_fom.py runs/<attempt>/archive [--no-rerun]
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_t2 as AT  # noqa: E402

BANDS = dict(BE=(.8, 1.25), CN=(1.7, 2.3), CNR=(1.7, 2.3), BDF2=(1.7, 2.3))
FIN = lambda x: x is not None and np.isfinite(x)


def within(a, b, rtol, atol=0.):
    return FIN(a) and FIN(b) and abs(a - b) <= atol + rtol * abs(b)


class Evidence:
    def __init__(self, arc):
        self.res = json.loads((arc / 'output/result.json').read_text())
        self.O = dict(np.load(arc / 'output/order_fields.npz'))
        self.CAL = dict(np.load(arc / 'output/calibration_fields_f32.npz'))
        self.F32 = dict(np.load(arc / 'output/fom_fields_f32.npz'))
        self.F64 = dict(np.load(arc / 'output/fom_fields_timing_f64.npz'))
        self.W = dict(np.load(arc / 'output/rom_W.npz'))


def validate(ev, cfg, refs, dec, a1k_rows):
    """Returns (checks, info). No early exits: every check is evaluated."""
    res = ev.res
    ck, info = {}, {}
    cases = [(coh, c) for coh in cfg['cohorts'] for c in range(dict(dev6=6, val32=32)[coh])]
    tidx = [i for i, x in enumerate(cases) if x[0] == 'dev6'][:cfg['timing']['cases']]
    ck['timing_case_indices'] = res.get('timing_case_indices') == tidx

    def n0r(coh, c):
        return np.linalg.norm(AT.initial257(AT.cohort(coh)[c]))

    def errs(fr, coh, c):
        rST, rS = refs[(coh, c, 'ST')], refs[(coh, c, 'S')]
        return {f'e_{t}': max(np.linalg.norm(fr[j] - R_[j]) for j in range(1, 6)) / n0r(coh, c)
                for t, R_ in (('ST', rST), ('S', rS), ('TX', (16 * rST - rS) / 15))}

    # ---- order check (L = 256)
    ok, orders = True, {}
    recs = res['order']
    keyset = {(r['scheme'], r['cohort'], r['case']) for r in recs}
    ok = ok and len(recs) == len(keyset) == len(cfg['schemes']) * len(cfg['order_cases']) and \
        keyset == {(sc, coh, c) for sc in cfg['schemes'] for coh, c in cfg['order_cases']}
    for r in recs:
        sc, coh, c = r['scheme'], r['cohort'], r['case']
        n0 = n0r(coh, c)
        try:
            F = {f: ev.O[f'{sc}__{coh}__{c}__{f:g}'] for f in (2, 1, .5, .25, .125)}
        except KeyError:
            ok = False
            continue
        d = {f: max(np.linalg.norm(F[f][j] - F[f / 2][j]) for j in range(1, 6)) / n0 for f in (2, 1, .5, .25)}
        good_d = all(FIN(v) and v > 0 for v in d.values())
        p = float(np.log2(d[.5] / d[.25])) if good_d else float('nan')
        orders[f'{sc}|{coh}{c}'] = p
        lo, hi = BANDS[sc]
        nf = r.get('nfail', {})
        gate = sc != 'CN'          # A11.1: undamped CN is reported, not gated (stiff-mode transient); CN-R is gated
        ok = ok and good_d and set(nf) == {'2', '1', '0.5', '0.25', '0.125'} and all(v == 0 for v in nf.values()) \
            and (lo <= p <= hi or not gate) and within(p, r['orders'].get('0.5'), 1e-9)
    ck['fom_order_check'] = bool(ok)
    info['fom_orders_finest'] = orders
    # ---- calibration (re-applied from saved float32 fields)
    dev = [x for x in cases if x[0] == 'dev6']
    want_cfg = [(sc, f) for sc in cfg['schemes'] for f in cfg['dt_factors']]
    ok, bad = True, []
    chosen = {}
    for sc, f in want_cfg:
        e_ = res['calibration'].get(f'{sc}|{f:g}')
        if e_ is None or [r['case'] for r in e_['cases']] != [f'{coh}{c}' for coh, c in dev]:
            ok = False
            bad.append((sc, f, 'cases'))
            continue
        tight_ok = all(r['tight_verified'] for r in e_['cases'])
        passes = {nt: True for nt in cfg['ntols']}
        for r, (coh, c) in zip(e_['cases'], dev):
            t_ = ev.CAL[f'{sc}__{f:g}__tight__{coh}__{c}'].astype(float)
            e_t = errs(t_, coh, c)['e_ST']
            ok = ok and FIN(r['tight_e_ST']) and abs(e_t - r['tight_e_ST']) <= 1e-6 * max(r['tight_e_ST'], 1e-3)
            for nt in cfg['ntols']:
                dd_job = r['diffs'][f'{nt:g}']['diff']
                fr = ev.CAL[f'{sc}__{f:g}__{nt:g}__{coh}__{c}'].astype(float)
                dd = max(np.linalg.norm(fr[j] - t_[j]) for j in range(1, 6)) / n0r(coh, c)
                ok = ok and FIN(dd_job) and dd_job >= 0 and abs(dd - dd_job) <= 1e-6        # float32 evidence
                passes[nt] = passes[nt] and r['diffs'][f'{nt:g}']['verified'] and dd_job < .01 * r['tight_e_ST']
        exp = None if not tight_ok else ([max(nt for nt in cfg['ntols'] if passes[nt]), 1e-8] if any(passes.values())
                                          else [1e-10, 1e-12])
        got = e_['chosen'] if e_['chosen'] is None else list(e_['chosen'])
        stat = 'unresolved' if exp is None else ('fallback_tight' if exp == [1e-10, 1e-12] else 'calibrated')
        if got != exp or e_['status'] != stat:
            ok = False
            bad.append((sc, f, got, exp))
        chosen[f'{sc}|{f:g}'] = exp
    ck['calibration_rule'] = bool(ok)
    info['calibration'] = {k: v for k, v in chosen.items()}
    info['calibration_mismatch'] = bad[:6]
    ck['timing_chosen_matches'] = {k: (None if v is None else list(v)) for k, v in res['timing'].get('chosen', {}).items()} == chosen
    # ---- FOM evaluation
    fruns = [f'fom|{k}' for k, v in chosen.items() if v is not None]
    rows = {}
    for r in res['rows']:
        rows[(r['run'], r['cohort'], r['case'])] = r
    ck['fom_inventory'] = len(rows) == len(res['rows']) and set(rows) == {(k, coh, c) for k in fruns for coh, c in cases}
    ok, bad = True, []
    rehash = {}
    for k in fruns:
        pair = chosen[k[4:]]
        for coh, c in cases:
            r = rows.get((k, coh, c))
            if r is None:
                ok = False
                continue
            st = r['stats']
            nst = int(st['steps'])
            srel = np.array(st['step_rel'][:nst], float)
            ver = bool(np.all(np.isfinite(srel)) and np.all(srel <= pair[0]))
            ok = ok and [r['ntol'], r['ltol']] == pair and r['verified'] == ver and int(st['nfail']) == int(np.sum(~(np.isfinite(srel) & (srel <= pair[0]))))
            f32 = ev.F32[f'{k}|{coh}|{c}'.replace('|', '__')].astype(float)
            e = errs(f32, coh, c)
            for t in ('ST', 'S', 'TX'):
                if not (FIN(r[f'e_{t}']) and abs(e[f'e_{t}'] - r[f'e_{t}']) <= 1e-6 * max(r[f'e_{t}'], 1e-3)):
                    ok = False
                    bad.append((k, coh, c, t))
        for ci in tidx:
            coh, c = cases[ci]
            f64 = ev.F64[f'{k}|{coh}|{c}'.replace('|', '__')]
            f32 = ev.F32[f'{k}|{coh}|{c}'.replace('|', '__')]
            rehash[(k, ci)] = AT.sha(f64)
            ok = ok and rehash[(k, ci)] == rows[(k, coh, c)]['restricted_sha256'] and np.array_equal(f64.astype(np.float32), f32)
    ck['fom_evaluation'] = bool(ok)
    info['fom_mismatch'] = bad[:6]
    info['fom_unverified'] = sorted({k for (k, coh, c), r in rows.items() if not r['verified']})
    # ---- ROM evaluation (coefficients -> independent NumPy decode)
    rkeys = [f"rom|{x['setting']}|{x['rule']}|{x['form']}|{x['scheme']}|{x['dt_factor']:g}" for x in cfg['rom_arms']]
    rrows = {(r['run'], r['cohort'], r['case']): r for r in res['rom_rows']}
    ck['rom_inventory'] = len(rrows) == len(res['rom_rows']) and set(rrows) == {(k, coh, c) for k in rkeys for coh, c in cases}
    ok, bad, xjob = True, [], []
    for k, x in zip(rkeys, cfg['rom_arms']):
        for coh, c in cases:
            r = rrows.get((k, coh, c))
            Wk = f'{k}|{coh}|{c}'.replace('|', '__')
            if r is None or Wk not in ev.W:
                ok = False
                continue
            ok = ok and r['verified'] == (int(r['stats']['nfail']) == 0)
            e = errs(dec.fields(ev.W[Wk]), coh, c)
            for t in ('ST', 'S', 'TX'):
                if not within(e[f'e_{t}'], r[f'e_{t}'], 1e-9, 1e-13):
                    ok = False
                    bad.append((k, coh, c, t))
            a = a1k_rows.get((x['setting'], f"main|{x['form']}|{x['scheme']}|{.005 * x['dt_factor']:.8g}|prod", coh, c))
            if a is not None:
                xjob.append(abs(a['e_ST'] - r['e_ST']))
    ck['rom_evaluation'] = bool(ok)
    info['rom_mismatch'] = bad[:6]
    info['rom_unverified'] = sorted({k for (k, coh, c), r in rrows.items() if not r['verified']})
    info['cross_job_max_abs_diff'] = max(xjob) if xjob else None
    info['cross_job_pairs'] = len(xjob)
    ck['cross_job_rom_agreement'] = bool(xjob) and max(xjob) <= 1e-9
    for ci in tidx:
        coh, c = cases[ci]
        for k in rkeys:
            rehash[(k, ci)] = AT.sha(ev.W[f'{k}|{coh}|{c}'.replace('|', '__')])
    # ---- timing
    inv = res['timing'].get('invocations', [])
    A_ = 'fom|BE|1'
    subj = fruns + rkeys
    want = {(b, ci, r_) for b in subj if b != A_ for ci in tidx for r_ in range(cfg['timing']['reps'])}
    got = [(i['B'], i['case_index'], i['rep']) for i in inv]
    ck['timing_inventory'] = res['timing'].get('A') == A_ and len(got) == len(set(got)) == len(want) and set(got) == want
    ok = True
    for i in inv:
        ts = [i['tA1'], i['tB'], i['tA2']]
        ok = ok and all(FIN(t) and t > 0 for t in ts) and within(i['ratio'], i['tB'] / (.5 * (i['tA1'] + i['tA2'])), 1e-12) \
            and within(i['drift'], i['tA2'] / i['tA1'], 1e-12) \
            and i['A_sha'][0] == i['A_sha'][1] == rehash.get((A_, i['case_index'])) and i['B_sha'] == rehash.get((i['B'], i['case_index']))
    ck['timing_values_and_hashes'] = bool(ok) and bool(inv)
    # ---- reproduction vs the 2D lane
    lane = json.loads((AT.LANE_DV / f"dv{res['mesh']}/archive/output/result.json").read_text())
    lr = {(r['cohort'], r['case']): r for r in lane['fom_rows']}
    dd = [max(abs(x['e_ST'] - lr[(x['cohort'], x['case'])]['ref_ST_evolved']), abs(x['e_S'] - lr[(x['cohort'], x['case'])]['ref_S_evolved']))
          for x in res['repro']]
    info['repro_max_abs_diff'] = max(dd) if dd else None
    ck['fom_reproduction'] = len(dd) == len(cases) and all(FIN(v) for v in dd) and max(dd) <= 1e-6
    return ck, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive')
    ap.add_argument('--no-rerun', action='store_true')
    a = ap.parse_args()
    arc = Path(a.archive)
    ev = Evidence(arc)
    res = ev.res
    logs = ''.join(f.read_text() for f in (arc / 'logs').glob('*.out'))
    att = res['config']['attempt']
    scfg = arc / f'experiments/jcp-time2/configs/{att}.json'
    prov = json.loads((arc / 'PROVENANCE.json').read_text())
    psha = next(x['sha256'] for x in prov if x['source'].endswith(f'configs/{att}.json'))
    cfg = json.loads(scfg.read_text())
    head = dict(complete=bool(res.get('complete')),
                backend_precision=res['backend'] == 'gpu' and 'jax_backend=gpu' in logs and 'precision=highest' in logs,
                config_authenticated=psha == hashlib.sha256(scfg.read_bytes()).hexdigest() and cfg == res['config'])
    man = json.loads((AT.REFSRC / 'result.json').read_text())
    ent = {(x['cohort'], x['case'], x['ref']): x['f257_sha256'] for x in man['cases']}
    cases = [(coh, c) for coh in cfg['cohorts'] for c in range(dict(dev6=6, val32=32)[coh])]
    refs = {}
    for coh, c in cases:
        for tag in ('ST', 'S'):
            z = np.load(AT.REFSRC / f'ref_{tag}_{coh}_{c:03d}.npz')['f257']
            assert AT.sha(z) == ent[(coh, c, tag)]
            refs[(coh, c, tag)] = z
    head['references_pinned'] = all(res['references'].get(f'{t}|{coh}|{c}') == ent[(coh, c, t)] for coh, c in cases for t in ('ST', 'S'))
    a1k = {}
    for f in (HERE / 'runs').glob('a1k*/archive/output/result.json'):
        R = json.loads(f.read_text())
        for r in R['rows']:
            a1k[(r['setting'], r['run'], r['cohort'], r['case'])] = r
    dec = AT.Decoder()
    ck, info = validate(ev, cfg, refs, dec, a1k)
    ck.update(head)
    # ---- rejection tests through the same validator
    rej = {}

    def mut(name, fn, key):
        e2 = copy.copy(ev)
        e2.res = copy.deepcopy(ev.res)
        for at in ('O', 'CAL', 'F32', 'F64', 'W'):
            setattr(e2, at, dict(getattr(ev, at)))
        fn(e2)
        c2, _ = validate(e2, cfg, refs, dec, a1k)
        rej[name] = not c2[key]
    r0 = next(i for i, r in enumerate(res['rows']))
    mut('fom_error_perturbed', lambda e2: e2.res['rows'][r0].__setitem__('e_ST', 1.05 * e2.res['rows'][r0]['e_ST'] + 1e-3), 'fom_evaluation')
    mut('fom_error_nan', lambda e2: e2.res['rows'][r0].__setitem__('e_ST', float('nan')), 'fom_evaluation')
    mut('fom_tolerance_wrong', lambda e2: e2.res['rows'][r0].__setitem__('ntol', 1e-3), 'fom_evaluation')
    mut('fom_verified_flag_flipped', lambda e2: e2.res['rows'][r0].__setitem__('verified', not e2.res['rows'][r0]['verified']), 'fom_evaluation')
    k1 = next(iter(ev.F32))
    mut('fom_field_swapped', lambda e2: e2.F32.__setitem__(k1, e2.F32[k1][::-1].copy()), 'fom_evaluation')
    mut('rom_coefficients_swapped', lambda e2: e2.W.__setitem__(next(iter(e2.W)), -e2.W[next(iter(e2.W))]), 'rom_evaluation')
    mut('calibration_duplicate_case', lambda e2: e2.res['calibration'][next(iter(e2.res['calibration']))]['cases'].__setitem__(
        1, e2.res['calibration'][next(iter(e2.res['calibration']))]['cases'][0]), 'calibration_rule')
    mut('calibration_negative_diff', lambda e2: next(iter(next(iter(e2.res['calibration'].values()))['cases'][0]['diffs'].values())).__setitem__('diff', -1.), 'calibration_rule')
    mut('timing_record_dropped', lambda e2: e2.res['timing']['invocations'].pop(0), 'timing_inventory')
    mut('timing_nan', lambda e2: e2.res['timing']['invocations'][0].__setitem__('tB', float('nan')), 'timing_values_and_hashes')
    mut('timing_hash_disconnected', lambda e2: e2.res['timing']['invocations'][0].__setitem__('B_sha', '0' * 64), 'timing_values_and_hashes')
    mut('order_nfail_missing', lambda e2: e2.res['order'][0].__setitem__('nfail', {}), 'fom_order_check')
    ck['rejections_fire'] = all(rej.values())
    info['rejections'] = rej
    # ---- independent BE re-run (engines.make_fom, not fom2.py)
    if a.no_rerun:
        ck['independent_BE_rerun'] = False
        info['independent_BE_rerun'] = 'skipped (audit incomplete)'
    else:
        sys.path.insert(0, str(AT.ROOT / 'experiments/mr-burgers2d'))
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        import engines as e
        L = res['mesh']
        ph = AT.cohort('dev6')[0]
        q, _ = e.make_fom(L, .005)
        nt, lt = res['calibration']['BE|1']['chosen']
        f = np.asarray(q(jnp.asarray(e.initial(L, ph)), float(ph[4]), nt, lt)[0])[:, ::L // 256, ::L // 256]
        d = max(np.linalg.norm(f[j] - ev.F64['fom__BE__1__dev6__0'][j]) for j in range(6)) / np.linalg.norm(AT.initial257(ph))
        info['independent_BE_rerun_diff'] = float(d)
        ck['independent_BE_rerun'] = d <= 1e-8
    allp = all(ck.values())
    out = dict(attempt=att, job_id=res['job_id'], commit=res['commit'], checks=ck, info=info, all_pass=allp)
    (HERE / 'checks' / f'audit-{att}.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    for k, v in ck.items():
        print(('PASS ' if v else 'FAIL ') + k)
    print(json.dumps(info, default=str)[:2500])
    print('AUDIT ALL PASS' if allp else 'AUDIT FAILED')
    sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
