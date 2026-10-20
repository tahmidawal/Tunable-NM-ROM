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


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()

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


sys.path.insert(0, str(HERE))
import audit_t2 as AT2  # noqa: E402
GTOL = dict(prod=1e-3, tight=1e-7, tighter=1e-8)


def rom_consistent(r, form, dt, lev):
    v, c = AT2.rom_steps_ok(r['stats'], form, dt, GTOL[lev])
    return c and r['verified'] == v


def validate(res, Wz, cfg, G, REF, U0):
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
            w0 = U0[(nm, j)]                                # independent L2 projection, computed once (main)
            if not np.allclose(W[j][:, 0], w0[None], rtol=0, atol=1e-7 * np.abs(w0).max()):
                bad.append((nm, j, 'initial_projection'))
            Fj = {k: Fc[i] for i, k in enumerate(er)}
            dist = lambda a, b: float((np.linalg.norm(Fj[a] - Fj[b], axis=1) / n0)[1:].max())
            anc = 'GAL|BDF2|0.000625|tight'
            for i, k in enumerate(er):
                r = rows.get((nm, k, j))
                if r is None:
                    bad.append((nm, k, j, 'missing'))
                    continue
                pe = np.linalg.norm(Fc[i] - REF[j], axis=1) / n0
                e = float(pe[1:].max())
                if not close(e, r['e_ref'], 1e-7, 1e-10) or not all(close(a_, b_, 1e-7, 1e-10) for a_, b_ in zip(pe, r['e_ref_per_time'])) \
                        or len(r['e_ref_per_time']) != 6:
                    bad.append((nm, k, j, 'e_ref', e, r['e_ref']))
                form, sc, dt, lev = k.split('|')
                dt = float(dt)
                if not rom_consistent(r, form, dt, lev) or not np.isfinite(W[j, i]).all():
                    bad.append((nm, k, j, 'verification'))
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
    vr = {(v['arm'], v['case']): v for v in res['vendor_rows']}
    ck['vendor_inventory'] = set(vr) == {(nm, j) for nm in names for j in cases} and len(res['vendor_rows']) == len(vr)
    vbad = []
    for nm in names:
        Wv, W = Wz[nm]['W_vendor'], Wz[nm]['W']
        er_i = {k: i for i, k in enumerate(er)}
        for j in cases:
            v = vr.get((nm, j))
            if v is None or Wv.shape[1:] != (6, G[nm].shape[0]) or not np.isfinite(Wv[j]).all():
                vbad.append((nm, j, 'missing'))
                continue
            n0 = np.linalg.norm(REF[j][0])
            Fv = Wv[j] @ G[nm]
            Fa = W[j, er_i['GAL|BDF2|0.000625|tight']] @ G[nm]
            Fb = W[j, er_i['LSPG|BE|0.01|prod']] @ G[nm]
            d = lambda X: float((np.linalg.norm(Fv - X, axis=1) / n0)[1:].max())
            if not (close(d(REF[j]), v['e_ref'], 1e-7, 1e-10) and close(d(Fa), v['anchor'], 1e-6, 1e-12)
                    and close(d(Fb), v['vs_generic_BE'], 1e-6, 1e-12)):
                vbad.append((nm, j, 'metrics'))
            rs_ = np.array(v.get('reasons_per_step', []), float)
            gn_ = np.array(v.get('gn_per_step', []), float)
            it_ = np.array(v.get('it_per_step', []), float)
            ints = lambda a: bool(np.all(np.isfinite(a)) and np.all(a == np.round(a)))
            if not (ints(rs_) and ints(it_) and it_.shape == (25,) and np.all((it_[:3] >= 0) & (it_[:3] <= 50))
                    and np.all(it_[3:] == 1) and int(v['it_sum']) == int(it_.sum())
                    and list(v['reasons']) == list(np.bincount(rs_.astype(int), minlength=5))):
                vbad.append((nm, j, 'vendor_iterations_or_histogram'))
            pe = np.linalg.norm(Fv - REF[j], axis=1) / n0
            # eligibility of the deployed fixed-sweep output (A12.3): every step finite with a valid exit code, no
            # non-finite exit (3); the fixed sweep does not promise the adaptive stopping test, so reason 0 is allowed
            if not (rs_.shape == (25,) and np.all((rs_ >= 0) & (rs_ <= 4)) and 3 not in rs_ and gn_.shape == (25,)
                    and np.all(np.isfinite(gn_)) and len(v['e_ref_per_time']) == 6
                    and all(close(a_, b_, 1e-7, 1e-10) for a_, b_ in zip(pe, v['e_ref_per_time'])) and int(v['it_sum']) >= 25):
                vbad.append((nm, j, 'vendor_solver_or_errors'))
    ck['vendor_rescored'] = not vbad
    info['vendor_mismatch'] = [list(map(str, b)) for b in vbad[:6]]
    tc = cfg['timing']
    ok = True
    for nm in names:
        t = res['timing'].get(nm, {})
        cands = [k for k in er if k.endswith('|prod') and float(k.split('|')[2]) >= DT0 / 2 - 1e-15]
        inv = t.get('invocations', [])
        got = [(i['B'], i['case'], i['rep']) for i in inv]
        want = {(b, j, r_) for b in cands for j in range(tc['cases']) for r_ in range(tc['reps'])}
        ok = ok and len(got) == len(set(got)) == len(want) and set(got) == want
        er_i = {k: i_ for i_, k in enumerate(er)}
        for i in inv:
            ts = [i['tA1'], i['tB'], i['tA2']]
            hA = sha(Wz[nm]['W_vendor'][i['case']])
            hB = sha(Wz[nm]['W'][i['case'], er_i[i['B']]]) if i['B'] in er_i else None
            ok = ok and all(np.isfinite(x) and x > 0 for x in ts) and close(i['ratio'], i['tB'] / (.5 * (i['tA1'] + i['tA2'])), 1e-12) \
                and close(i['drift'], i['tA2'] / i['tA1'], 1e-12) and i.get('A') == 'vendor' \
                and i['A_sha'][0] == i['A_sha'][1] == hA and i['B_sha'] == hB
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
    bank_path = HERE / 'vendor/quad3d/inputs/model_M2/bank.pkl'
    head['model_authenticated'] = hashlib.sha256(bank_path.read_bytes()).hexdigest() == res['model_sha256'] == cfg['expected_model_sha256']
    head['rules_authenticated'] = hashlib.sha256((HERE / 'vendor/quad3d/rules/rules.npz').read_bytes()).hexdigest() == cfg['expected_rules_sha256']
    import subprocess
    cert = json.loads(subprocess.check_output(['git', '-C', str(HERE), 'show', f"{res['commit']}:experiments/jcp-time2/checks/test_lmm.json"]))
    t2sha = next(x['sha256'] for x in prov if x['source'].endswith('jcp-time2/t2core.py'))
    head['G2a_certificate'] = bool(cert['all_pass'] and cert['source_sha256']['t2core.py'] == t2sha)
    b = pickle.loads(bank_path.read_bytes())
    bank = jax.tree_util.tree_map(jnp.asarray, b['params'])
    T = np.asarray(b['rotation'])
    X65 = lattice65_coords()
    G = {}
    for x in cfg['rules']:
        G[f"{x['rule']}_R{x['Rp']}"] = np.asarray(TB.feature_rows(bank, T[:, :x['Rp']], X65))
    Wz = {nm: dict(np.load(arc / f'output/W_{nm}.npz')) for nm in G}
    tab = C.table(cfg['cohort_seed'], cfg['cohort_count'])
    U0 = {}
    for nm, Gm in G.items():        # independent L2 projections: one least-squares solve for all cases (normal equations)
        Uall = np.stack([np.asarray(C.initial_interior(65, tab, j)) for j in range(cfg['cohort_count'])], 1)
        Wls = np.linalg.solve(Gm @ Gm.T, Gm @ Uall)
        for j in range(cfg['cohort_count']):
            U0[(nm, j)] = Wls[:, j]
    ck, info = validate(res, Wz, cfg, G, REF, U0)
    ck.update(head)
    rej = {}

    def mut(name, fn, keyc):
        r2, W2 = copy.deepcopy(res), {k: dict(v) for k, v in Wz.items()}
        fn(r2, W2)
        c2, _ = validate(r2, W2, cfg, G, REF, U0)
        rej[name] = not c2[keyc]
    nm0 = next(iter(G))
    mut('error_perturbed', lambda r2, W2: r2['rows'][0].__setitem__('e_ref', 1.01 * r2['rows'][0]['e_ref'] + 1e-4), 'rescored')
    mut('coefficients_case_swap', lambda r2, W2: W2[nm0].__setitem__('W', W2[nm0]['W'][[1, 0] + list(range(2, len(W2[nm0]['W'])))]), 'rescored')
    mut('verified_flag_flipped', lambda r2, W2: r2['rows'][0].__setitem__('verified', not r2['rows'][0]['verified']), 'rescored')
    mut('timing_dropped', lambda r2, W2: r2['timing'][nm0]['invocations'].pop(0), 'timing')
    mut('time_shift', lambda r2, W2: W2[nm0].__setitem__('W', np.roll(W2[nm0]['W'], 1, axis=2)), 'rescored')
    mut('vendor_error_perturbed', lambda r2, W2: r2['vendor_rows'][0].__setitem__('e_ref', 1.01 * r2['vendor_rows'][0]['e_ref'] + 1e-4), 'vendor_rescored')
    mut('timing_hash_disconnected', lambda r2, W2: r2['timing'][nm0]['invocations'][0].__setitem__('B_sha', '0' * 64), 'timing')
    mut('timing_drift_nan', lambda r2, W2: r2['timing'][nm0]['invocations'][0].__setitem__('drift', float('nan')), 'timing')
    ck['rejections_fire'] = all(rej.values())
    info['rejections'] = rej
    allp = all(ck.values())
    out = dict(attempt=att, job_id=res['job_id'], commit=res['commit'], checks=ck, info=info, all_pass=allp,
               result_sha256=hashlib.sha256((arc / 'output/result.json').read_bytes()).hexdigest())
    (HERE / 'checks' / f'audit-{att}.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    for k, v in ck.items():
        print(('PASS ' if v else 'FAIL ') + k)
    print(json.dumps(info, default=str)[:2000])
    print('AUDIT ALL PASS' if allp else 'AUDIT FAILED')
    sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
