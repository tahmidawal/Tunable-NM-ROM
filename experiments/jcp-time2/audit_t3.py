"""Audit of a pulled t3run job (DESIGN A10.2). Re-scores every saved coefficient trajectory through a separate
evaluation path (bank features evaluated at the 65-node lattice coordinates with tables.feature_rows, instead of the
job's mesh rows), against the pinned refined reference; checks inventories (from the staged, provenance-authenticated
config), verification flags against the saved statistics, the anchor/self-difference metrics, and timing records.
Rejection tests feed mutated copies through the same validator.

    /home/tahmid/Dev/.venv/bin/python experiments/jcp-time2/audit_t3.py runs/<attempt>/archive
"""
import argparse
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
import jax  # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
from offmesh import C, TB  # noqa: E402
from qpanel import lattice65_coords  # noqa: E402

REF3 = HERE.parents[2] / '2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/runs/ref1/code/output'
DT0 = .01


def expected_runs(g):
    dts = [DT0 * f for f in g['dt_factors']]
    tig = [DT0 * f for f in g['tight_factors']]
    out = []
    for form in g['forms']:
        for sc in g['schemes']:
            out += [(form, sc, d, 'prod') for d in dts] + [(form, sc, d, 'tight') for d in tig] + [(form, sc, d, 'tighter') for d in tig]
        for sc in g['control_schemes']:
            out += [(form, sc, d, 'tight') for d in tig] + [(form, sc, d, 'tighter') for d in tig]
        if form == 'GAL':
            for sc in g['anchor_schemes']:
                out += [(form, sc, DT0 / 16, 'tight'), (form, sc, DT0 / 16, 'tighter')]
    return [f'{f}|{s}|{d:.8g}|{l}' for f, s, d, l in out]


def close(a, b, rtol=1e-8, atol=1e-12):
    return a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and abs(a - b) <= atol + rtol * abs(b)


def validate(res, Wz, cfg, G, REF):
    ck, info = {}, {}
    er = expected_runs(cfg['grid'])
    names = [f"{x['rule']}_R{x['Rp']}" for x in cfg['rules']]
    cases = list(range(cfg['cohort_count']))
    rows = {(r['arm'], r['run'], r['case']): r for r in res['rows']}
    ck['inventory'] = res.get('runs') == er and len(rows) == len(res['rows']) == len(names) * len(er) * len(cases)
    bad, miss = [], []
    for nm in names:
        W = Wz[nm]['W']
        ok_shape = W.shape == (len(cases), len(er), 6, G[nm].shape[0]) and list(Wz[nm]['runs']) == er
        if not ok_shape:
            bad.append((nm, 'shape'))
            continue
        for j in cases:
            Fc = W[j] @ G[nm]                               # (runs, 6, 63^3), one case at a time
            n0 = np.linalg.norm(REF[j][0])
            Fj = {k: Fc[i] for i, k in enumerate(er)}
            dist = lambda a, b: float((np.linalg.norm(Fj[a] - Fj[b], axis=1) / n0)[1:].max())
            anc = 'GAL|BDF2|0.000625|tight'
            for i, k in enumerate(er):
                r = rows.get((nm, k, j))
                if r is None:
                    bad.append((nm, k, j, 'missing'))
                    continue
                e = float((np.linalg.norm(Fc[i] - REF[j], axis=1) / n0)[1:].max())
                if not close(e, r['e_ref'], 1e-7, 1e-10):
                    bad.append((nm, k, j, 'e_ref', e, r['e_ref']))
                if r['verified'] != (int(r['stats']['nfail']) == 0 and np.isfinite(W[j, i]).all()):
                    bad.append((nm, k, j, 'verified_flag'))
                form, sc, dt, lev = k.split('|')
                dt = float(dt)
                checks = [('anchor', anc)]
                if f'{form}|{sc}|{dt / 2:.8g}|{lev}' in Fj:
                    checks.append(('d_half', f'{form}|{sc}|{dt / 2:.8g}|{lev}'))
                if lev == 'prod' and f'{form}|{sc}|{dt:.8g}|tight' in Fj:
                    checks.append(('prod_vs_tight', f'{form}|{sc}|{dt:.8g}|tight'))
                if lev == 'tight' and f'{form}|{sc}|{dt:.8g}|tighter' in Fj:
                    checks.append(('s_h', f'{form}|{sc}|{dt:.8g}|tighter'))
                for tag, other in checks:
                    if tag not in r:
                        miss.append((nm, k, j, tag))
                    elif not close(dist(k, other), r[tag], 1e-6, 1e-12):
                        bad.append((nm, k, j, tag))
    ck['rescored'] = not bad
    ck['mandatory_metrics'] = not miss
    info['mismatches'] = [list(map(str, b)) for b in bad[:10]]
    info['missing'] = [list(map(str, m)) for m in miss[:10]]
    vr = {(v['arm'], v['case']) for v in res['vendor_rows']}
    ck['vendor_inventory'] = vr == {(nm, j) for nm in names for j in cases} and len(res['vendor_rows']) == len(vr)
    tc = cfg['timing']
    ok = True
    for nm in names:
        t = res['timing'].get(nm, {})
        cands = [k for k in er if k.endswith('|prod') and float(k.split('|')[2]) >= DT0 / 2 - 1e-15]
        inv = t.get('invocations', [])
        got = [(i['B'], i['case'], i['rep']) for i in inv]
        want = {(b, j, r_) for b in cands for j in range(tc['cases']) for r_ in range(tc['reps'])}
        ok = ok and len(got) == len(set(got)) == len(want) and set(got) == want
        for i in inv:
            ts = [i['tA1'], i['tB'], i['tA2']]
            ok = ok and all(np.isfinite(x) and x > 0 for x in ts) and close(i['ratio'], i['tB'] / (.5 * (i['tA1'] + i['tA2'])), 1e-12) \
                and i['A_expected_sha'] and i['B_expected_sha'] and i['A_sha'][0] == i['A_sha'][1] == i['A_expected_sha'] \
                and i['B_sha'] == i['B_expected_sha']
    ck['timing'] = bool(ok)
    info['unverified'] = sorted({f"{r['arm']}|{r['run']}" for r in res['rows'] if not r['verified']})
    return ck, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive')
    a = ap.parse_args()
    arc = Path(a.archive)
    res = json.loads((arc / 'output/result.json').read_text())
    logs = ''.join(f.read_text() for f in (arc / 'logs').glob('*.out'))
    att = res['config']['attempt']
    scfg = arc / f'experiments/jcp-time2/configs/{att}.json'
    prov = json.loads((arc / 'PROVENANCE.json').read_text())
    psha = next(x['sha256'] for x in prov if x['source'].endswith(f'configs/{att}.json'))
    cfg = json.loads(scfg.read_text())
    head = dict(complete=bool(res.get('complete')),
                backend_precision=res['backend'] == 'gpu' and 'jax_backend=gpu' in logs and 'precision=highest' in logs,
                config_authenticated=psha == hashlib.sha256(scfg.read_bytes()).hexdigest() and cfg == res['config'])
    r3 = REF3 / f"ref_{cfg['cohort_seed']}.npz"
    dn = json.loads(r3.with_suffix('.done').read_text())
    head['reference_pinned'] = hashlib.sha256(r3.read_bytes()).hexdigest() == dn['sha256'] == res['reference']['sha256']
    z = np.load(r3)
    REF = {j: np.asarray(z[f'c{j}']) for j in range(cfg['cohort_count'])}
    b = pickle.loads((HERE / 'vendor/quad3d/inputs/model_M2/bank.pkl').read_bytes())
    bank = jax.tree_util.tree_map(jnp.asarray, b['params'])
    T = np.asarray(b['rotation'])
    X65 = lattice65_coords()
    G = {}
    for x in cfg['rules']:
        G[f"{x['rule']}_R{x['Rp']}"] = np.asarray(TB.feature_rows(bank, T[:, :x['Rp']], X65))
    Wz = {nm: dict(np.load(arc / f'output/W_{nm}.npz')) for nm in G}
    ck, info = validate(res, Wz, cfg, G, REF)
    ck.update(head)
    rej = {}

    def mut(name, fn, keyc):
        r2, W2 = copy.deepcopy(res), {k: dict(v) for k, v in Wz.items()}
        fn(r2, W2)
        c2, _ = validate(r2, W2, cfg, G, REF)
        rej[name] = not c2[keyc]
    nm0 = next(iter(G))
    mut('error_perturbed', lambda r2, W2: r2['rows'][0].__setitem__('e_ref', 1.01 * r2['rows'][0]['e_ref'] + 1e-4), 'rescored')
    mut('coefficients_case_swap', lambda r2, W2: W2[nm0].__setitem__('W', W2[nm0]['W'][[1, 0] + list(range(2, len(W2[nm0]['W'])))]), 'rescored')
    mut('verified_flag_flipped', lambda r2, W2: r2['rows'][0].__setitem__('verified', not r2['rows'][0]['verified']), 'rescored')
    mut('timing_dropped', lambda r2, W2: r2['timing'][nm0]['invocations'].pop(0), 'timing')
    ck['rejections_fire'] = all(rej.values())
    info['rejections'] = rej
    allp = all(ck.values())
    out = dict(attempt=att, job_id=res['job_id'], commit=res['commit'], checks=ck, info=info, all_pass=allp)
    (HERE / 'checks' / f'audit-{att}.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    for k, v in ck.items():
        print(('PASS ' if v else 'FAIL ') + k)
    print(json.dumps(info, default=str)[:2000])
    print('AUDIT ALL PASS' if allp else 'AUDIT FAILED')
    sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
