"""Independent audit of a pulled fomrun job (DESIGN section 9, A1.10, A2.21, A3.5, A9). NumPy only except the
optional independent BE re-run. Everything required is derived from the staged config (authenticated by the staging
provenance), not from the job's own lists.

Checks: completion, backend and precision; config authentication; reference pins; FOM order check (re-computed from the
saved L=256 fields: BE in [0.8, 1.25], CN/CN-R/BDF2 in [1.7, 2.3] at the finest pair, every trajectory verified);
calibration rule re-applied (six cases, tight solves verified, unresolved configurations excluded); FOM evaluation
inventory and verification; FOM errors re-scored from the saved float32 fields (tolerance 1e-6 relative) and the timing
cases' float64 fields re-hashed; ROM errors re-scored from the saved coefficients (npbank decode, tolerance 1e-9);
timing inventory and hashes against the re-hashed evaluation outputs; FOM reproduction against the 2D lane's FOM rows;
an independent local re-run of BE at dt0 with engines.make_fom (not fom2.py) on dev6 case 0.
A skipped check (--no-rerun) makes the audit incomplete (all_pass false). Rejection tests: a perturbed FOM error, a
swapped FOM field, a dropped timing record must each be rejected.

    /home/tahmid/Dev/.venv/bin/python experiments/jcp-time2/audit_fom.py runs/<attempt>/archive [--no-rerun]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_t2 as AT  # noqa: E402

BANDS = dict(BE=(.8, 1.25), CN=(1.7, 2.3), CNR=(1.7, 2.3), BDF2=(1.7, 2.3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive')
    ap.add_argument('--no-rerun', action='store_true')
    a = ap.parse_args()
    arc = Path(a.archive)
    res = json.loads((arc / 'output/result.json').read_text())
    logs = ''.join(f.read_text() for f in (arc / 'logs').glob('*.out'))
    att = res['config']['attempt']
    ck, info = {}, {}
    ck['complete'] = bool(res.get('complete'))
    ck['backend_precision'] = res['backend'] == 'gpu' and 'jax_backend=gpu' in logs and 'precision=highest' in logs
    scfg = arc / f'experiments/jcp-time2/configs/{att}.json'
    prov = json.loads((arc / 'PROVENANCE.json').read_text())
    psha = next(x['sha256'] for x in prov if x['source'].endswith(f'configs/{att}.json'))
    ck['config_authenticated'] = psha == hashlib.sha256(scfg.read_bytes()).hexdigest() and json.loads(scfg.read_text()) == res['config']
    cfg = json.loads(scfg.read_text())
    man = json.loads((AT.REFSRC / 'result.json').read_text())
    ent = {(x['cohort'], x['case'], x['ref']): x['f257_sha256'] for x in man['cases']}
    cases = [(coh, c) for coh in cfg['cohorts'] for c in range(dict(dev6=6, val32=32)[coh])]
    refs = {}
    for coh, c in cases:
        for tag in ('ST', 'S'):
            z = np.load(AT.REFSRC / f'ref_{tag}_{coh}_{c:03d}.npz')['f257']
            assert AT.sha(z) == ent[(coh, c, tag)]
            refs[(coh, c, tag)] = z
    ck['references_pinned'] = all(res['references'].get(f'{t}|{coh}|{c}') == ent[(coh, c, t)] for coh, c in cases for t in ('ST', 'S'))

    def errs(fr, coh, c):
        n0r = np.linalg.norm(AT.initial257(AT.cohort(coh)[c]))
        rST, rS = refs[(coh, c, 'ST')], refs[(coh, c, 'S')]
        return {f'e_{t}': max(np.linalg.norm(fr[j] - R_[j]) / n0r for j in range(1, 6))
                for t, R_ in (('ST', rST), ('S', rS), ('TX', (16 * rST - rS) / 15))}

    # ---- order check
    O = np.load(arc / 'output/order_fields.npz')
    ordbad, orders = [], {}
    for coh, c in cfg['order_cases']:
        n0 = np.linalg.norm(AT.initial257(AT.cohort(coh)[c])[1:-1, 1:-1])     # L = 256: the 257 grid is the mesh
        for sc in cfg['schemes']:
            F = {f: O[f'{sc}__{coh}__{c}__{f:g}'] for f in (2, 1, .5, .25, .125)}
            d = {f: max(np.linalg.norm(F[f][j] - F[f / 2][j]) for j in range(1, 6)) / n0 for f in (2, 1, .5, .25)}
            p = float(np.log2(d[.5] / d[.25]))
            orders[f'{sc}|{coh}{c}'] = p
            row = next(r for r in res['order'] if r['scheme'] == sc and r['cohort'] == coh and r['case'] == c)
            lo, hi = BANDS[sc]
            if not (lo <= p <= hi) or any(v != 0 for v in row['nfail'].values()) or not AT.close(p, row['orders']['0.5']):
                ordbad.append((sc, coh, c, p))
    ck['fom_order_check'] = not ordbad and len(orders) == len(cfg['schemes']) * len(cfg['order_cases'])
    info['fom_orders_finest'] = orders
    # ---- calibration
    calbad = []
    want_cfg = [(sc, f) for sc in cfg['schemes'] for f in cfg['dt_factors']]
    for sc, f in want_cfg:
        e_ = res['calibration'].get(f'{sc}|{f:g}')
        if e_ is None or len(e_['cases']) != 6:
            calbad.append((sc, f, 'missing'))
            continue
        if not all(r['tight_verified'] for r in e_['cases']):
            exp = None
        else:
            ok = [nt for nt in sorted(cfg['ntols'], reverse=True)
                  if all(r['diffs'][f'{nt:g}']['verified'] and r['diffs'][f'{nt:g}']['diff'] < .01 * r['tight_e_ST'] for r in e_['cases'])]
            exp = [ok[0], 1e-8] if ok else [1e-10, 1e-12]
        if (e_['chosen'] if e_['chosen'] is None else list(e_['chosen'])) != exp:
            calbad.append((sc, f))
    ck['calibration_rule'] = not calbad
    info['calibration'] = {k: v['status'] for k, v in res['calibration'].items()}
    resolved = [(sc, f) for sc, f in want_cfg if res['calibration'].get(f'{sc}|{f:g}', {}).get('chosen')]
    fruns = [f'fom|{sc}|{f:g}' for sc, f in resolved]
    rows = {(r['run'], r['cohort'], r['case']): r for r in res['rows']}
    ck['fom_inventory'] = len(rows) == len(res['rows']) and set(rows) == {(k, coh, c) for k in fruns for coh, c in cases}
    info['fom_unverified'] = [k for k, r in rows.items() if not r['verified']]
    # ---- FOM re-scoring
    F32 = np.load(arc / 'output/fom_fields_f32.npz')
    F64 = np.load(arc / 'output/fom_fields_timing_f64.npz')
    bad = []
    rehash = {}
    tcases = [x for x in cases if x[0] == 'dev6'][:cfg['timing']['cases']]
    for k in fruns:
        for coh, c in cases:
            fr = F32[f'{k}|{coh}|{c}'.replace('|', '__')].astype(float)
            e = errs(fr, coh, c)
            for t in ('ST', 'S', 'TX'):
                if abs(e[f'e_{t}'] - rows[(k, coh, c)][f'e_{t}']) > 1e-6 * max(rows[(k, coh, c)][f'e_{t}'], 1e-3):
                    bad.append((k, coh, c, t))
        for ci, (coh, c) in enumerate(tcases):
            f64 = F64[f'{k}|{coh}|{c}'.replace('|', '__')]
            rehash[(k, ci)] = AT.sha(f64)
            if rehash[(k, ci)] != rows[(k, coh, c)]['restricted_sha256']:
                bad.append((k, coh, c, 'sha'))
    ck['fom_rescored'] = not bad
    info['fom_rescore_mismatches'] = bad[:10]
    # ---- ROM re-scoring (coefficients -> npbank decode)
    Wz = np.load(arc / 'output/rom_W.npz')
    dec = AT.Decoder()
    rrows = {(r['run'], r['cohort'], r['case']): r for r in res['rom_rows']}
    rkeys = [f"rom|{x['setting']}|{x['rule']}|{x['form']}|{x['scheme']}|{x['dt_factor']:g}" for x in cfg['rom_arms']]
    ck['rom_inventory'] = len(rrows) == len(res['rom_rows']) and set(rrows) == {(k, coh, c) for k in rkeys for coh, c in cases}
    rbad = []
    for k in rkeys:
        for coh, c in cases:
            F = dec.fields(Wz[f'{k}|{coh}|{c}'.replace('|', '__')])
            e = errs(F, coh, c)
            for t in ('ST', 'S', 'TX'):
                if not AT.close(e[f'e_{t}'], rrows[(k, coh, c)][f'e_{t}']):
                    rbad.append((k, coh, c, t))
    ck['rom_rescored'] = not rbad
    info['rom_rescore_mismatches'] = rbad[:10]
    info['rom_unverified'] = [k for k, r in rrows.items() if not r['verified']]
    # ---- timing
    inv = res['timing']['invocations']
    subj = fruns + rkeys
    A_ = 'fom|BE|1'
    want = {(b, ci, r_) for b in subj if b != A_ for ci in range(len(tcases)) for r_ in range(cfg['timing']['reps'])}
    got = [(i['B'], i['case_index'], i['rep']) for i in inv]
    ck['timing_inventory'] = len(got) == len(set(got)) == len(want) and set(got) == want

    def tim_hash_ok(inv_):
        okk = True
        for i in inv_:
            ci = i['case_index']
            coh, c = tcases[ci]
            expB = rehash.get((i['B'], ci)) if i['B'].startswith('fom|') else rrows[(i['B'], coh, c)]['restricted_sha256']
            okk = okk and i['A_sha'][0] == i['A_sha'][1] == rehash[(A_, ci)] and i['B_sha'] == expB
        return okk
    ck['timing_hashes'] = tim_hash_ok(inv)
    # ---- FOM reproduction vs the 2D lane
    lane = json.loads((AT.LANE_DV / f"dv{res['mesh']}/archive/output/result.json").read_text())
    lr = {(r['cohort'], r['case']): r for r in lane['fom_rows']}
    dd = [max(abs(x['e_ST'] - lr[(x['cohort'], x['case'])]['ref_ST_evolved']), abs(x['e_S'] - lr[(x['cohort'], x['case'])]['ref_S_evolved']))
          for x in res['repro']]
    info['repro_max_abs_diff'] = max(dd)
    ck['fom_reproduction'] = len(dd) == len(cases) and max(dd) <= 1e-6
    # ---- rejection tests
    rej = {}
    k0 = fruns[0]
    r0 = dict(rows[(k0, 'dev6', 0)])
    e0 = errs(F32[f'{k0}|dev6|0'.replace('|', '__')].astype(float), 'dev6', 0)['e_ST']
    rej['perturbed_error'] = abs(e0 - 1.01 * r0['e_ST']) > 1e-6 * max(r0['e_ST'], 1e-3)
    e_sw = errs(F32[f'{k0}|dev6|1'.replace('|', '__')].astype(float), 'dev6', 0)['e_ST']
    rej['swapped_field'] = abs(e_sw - r0['e_ST']) > 1e-6 * max(r0['e_ST'], 1e-3)
    inv_d = inv[1:]
    got_d = [(i['B'], i['case_index'], i['rep']) for i in inv_d]
    rej['dropped_timing_record'] = not (len(got_d) == len(want) and set(got_d) == want)
    ck['rejections_fire'] = all(rej.values())
    info['rejections'] = rej
    # ---- independent BE re-run
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
        n0r = np.linalg.norm(AT.initial257(ph))
        d = max(np.linalg.norm(f[j] - F64['fom__BE__1__dev6__0'][j]) / n0r for j in range(6))
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
