"""Independent NumPy audit of one burgers-bank-knob attempt (no JAX import anywhere in this file).

From the SAVED outputs, the checkpoint and the committed rotation only, it
  * recomputes the same-grid evolved error of every (arm, case) on the restricted grid (every arm) and the full
    grid (audit case), against the job's numbers;
  * re-derives every certificate status from the saved per-state rho, and re-computes rho itself in NumPy for
    spot states (the rotated, truncated bank evaluated from the checkpoint's feature network, times T);
  * checks the parity gate, the certificate control, the repetition gates and the pre-registered order-effect gate;
  * builds the per-arm table, each arm's speedup against the fastest FOM at least as accurate as itself, and applies
    the pre-registered setting rule (DESIGN.md section 6);
  * runs two injected controls (a swapped case, a perturbed error) through the same comparators and requires both
    to be DETECTED.

    python audit_bankknob.py runs/<attempt>/archive --checkpoint <pkl> --rotation inputs/rotation_R512.npz --out checks/<attempt>-summary.json
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

BAR = 0.116


def med(x):
    return float(np.median(np.asarray(x, float)))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def silu(x):
    return x / (1. + np.exp(-x))


def features_np(params, p):
    ang = 2. * np.pi * (p @ np.asarray(params['B']))
    x = np.concatenate((np.sin(ang), np.cos(ang)), -1)
    for w, b in params['g'][:-1]:
        x = silu(x @ np.asarray(w) + np.asarray(b))
    w, b = params['g'][-1]
    x = x @ np.asarray(w) + np.asarray(b)
    bc = 16. * p[:, 0] * (1 - p[:, 0]) * p[:, 1] * (1 - p[:, 1])
    return float(np.asarray(params['out_scale'])) * bc[:, None] * x


def fields_np(params, Tp, coefs, L, chunk=65536):
    """(G T_p) coefs^T on the interior grid, row-major (i, j), without holding the bank: (S, L-1, L-1)."""
    x = np.arange(1, L) / L
    xy = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    W = Tp @ coefs.T                                                  # (R, S)
    out = np.empty((len(xy), coefs.shape[0]))
    for s in range(0, len(xy), chunk):
        out[s:s + chunk] = features_np(params, xy[s:s + chunk]) @ W
    return out.T.reshape(-1, L - 1, L - 1)


def modes_np(L, M):
    k = np.arange(1, L)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    lam = 4 * L ** 2 * (np.sin(np.pi * kx / (2 * L)) ** 2 + np.sin(np.pi * ky / (2 * L)) ** 2)
    ind = np.argsort(lam.ravel(), kind='stable')[:M]
    return kx.ravel()[ind], ky.ravel()[ind]


def advection_np(U, L):
    p = np.pad(U, 1)
    c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    return c * L * (np.where(c > 0, c - xm, xp - c) + np.where(c > 0, c - ym, yp - c))


def rho_np(U, L, kx, ky, ij, w):
    a = advection_np(U, L)
    x = np.arange(1, L) / L
    sx, sy = np.sin(np.pi * x[:, None] * kx), np.sin(np.pi * x[:, None] * ky)
    exact = (2. / L) * np.sum(sx * (a @ sy), axis=0)
    rows = (2. / L) * np.sin(np.pi * (ij[:, 0] / L)[:, None] * kx) * np.sin(np.pi * (ij[:, 1] / L)[:, None] * ky)
    samp = (w * a[ij[:, 0] - 1, ij[:, 1] - 1]) @ rows
    return float(np.linalg.norm(samp - exact) / np.linalg.norm(exact))


def head_np(params, z):
    x = z
    for w, b in params['h'][:-1]:
        x = silu(x @ np.asarray(w) + np.asarray(b))
    w, b = params['h'][-1]
    return x @ np.asarray(w) + np.asarray(b) + z @ np.asarray(params['h_lin'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rotation', required=True)
    p.add_argument('--directions', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--spot-random', type=int, default=2)
    p.add_argument('--no-rho-spot', action='store_true')
    a = p.parse_args()
    arc = Path(a.archive)
    o = arc / 'output'
    r = json.loads((o / 'result.json').read_text())
    cfg = r['config']
    L = r['intervals']
    R = r['R']
    K = r['K']
    tight = cfg['same_grid_reference']
    gates = {}

    def gate(name, ok, **kw):
        gates[name] = dict(passed=bool(ok), **kw)

    log = ''.join(x.read_text(errors='replace') for x in sorted((arc / 'logs').glob('*.out')))
    gate('log_says_backend_gpu', 'jax_backend=gpu' in log)
    gate('complete', r.get('complete', False))
    gate('x64_and_highest', r['x64'] and r['matmul_precision'] == 'highest')
    gate('cohort_hash', r['gates']['evaluation_cohort_hash']['passed'] in (True, None),
         expected=r['gates']['evaluation_cohort_hash']['expected'], got=r['physical_sha256'])
    gate('rotation_file_is_the_committed_one', r['rotation']['file_sha256'] == cfg['rotation_sha256'] == sha(a.rotation))
    for k in ('rotated_operator_matches_parent_times_T', 'phi_free_operator_parity', 'repetition_output_identical',
              'retained_repetitions'):
        if k in r['gates']:
            gate(k, r['gates'][k]['passed'], job=r['gates'][k])

    # ---- errors: restricted grid, every (arm, case); full grid for the audit case --------------------------
    cases = sorted({x['case'] for x in r['quick']})
    truth_r = {c: np.load(o / f'restricted_{tight}_case{c}.npz')['fields'] for c in cases}
    exact_restriction = max(1, L // cfg.get('restrict_to', 256)) == 1

    def compare(xrows, truth):
        gap = 0.
        for x in xrows:
            f = np.load(o / f"restricted_{x['name']}_case{x['case']}.npz")['fields']
            b = truth[x['case']]
            sg = np.linalg.norm((f - b).reshape(len(f), -1), axis=1) / np.linalg.norm(b[0])
            if x['same_grid_evolved'] > 1e-6:
                gap = max(gap, abs(sg[1:].max() / x['same_grid_evolved'] - 1))
        return gap
    bar_r = 1e-9 if exact_restriction else 0.05
    worst_gap = compare(r['quick'], truth_r)
    gate('restricted_recomputation_tracks_job', worst_gap < bar_r, worst_relative_gap=worst_gap,
         restriction_is_identity=exact_restriction, bar=bar_r)
    full = []
    for c in cfg['audit_cases']:
        tp = o / f'full_{tight}_case{c}.npy'
        if not tp.exists():
            continue
        t = np.load(tp)
        n0 = float(np.linalg.norm(t[0]))
        for fp in sorted(o.glob(f'full_*_case{c}.npy')):
            name = fp.name[len('full_'):-len(f'_case{c}.npy')]
            f = np.load(fp)
            sg = [float(np.linalg.norm(f[k] - t[k])) / n0 for k in range(len(f))]
            job = next(x for x in r['quick'] if x['name'] == name and x['case'] == c)
            full.append(dict(name=name, case=c, recomputed_evolved=max(sg[1:]), job_evolved=job['same_grid_evolved'],
                             abs_diff=abs(max(sg[1:]) - job['same_grid_evolved'])))
            del f
    gate('full_grid_errors_recomputed', bool(full) and all(x['abs_diff'] <= 1e-12 for x in full),
         worst_abs_diff=max([x['abs_diff'] for x in full] or [None]), arms=len(full))

    # ---- injected controls: the comparators must DETECT a swapped case and a perturbed reported error ------
    roms = [x for x in r['quick'] if x['family'] == 'rom']
    probe = [x for x in roms if x['same_grid_evolved'] > 1e-6][:6]
    if len(cases) > 1:
        swapped = {c: truth_r[cases[(i + 1) % len(cases)]] for i, c in enumerate(cases)}
        gap_swap = compare(probe, swapped)
    else:
        gap_swap = None
    pert = [dict(x, same_grid_evolved=x['same_grid_evolved'] * (1 + 1e-3)) for x in probe]
    gap_pert = compare(pert, truth_r)
    full_pert = [abs(x['recomputed_evolved'] - x['job_evolved'] * (1 + 1e-6)) for x in full
                 if x['job_evolved'] > 1e-6]
    full_pert_detected = bool(full_pert) and min(full_pert) > 1e-12
    gate('controls_detected', (gap_swap is None or gap_swap >= bar_r) and gap_pert >= min(bar_r, 1e-4)
         and full_pert_detected, swapped_case_gap=gap_swap, perturbed_error_gap=gap_pert, bar=bar_r,
         full_grid_perturbation_1em6_detected=full_pert_detected,
         note='the same comparator that accepts the job, fed a swapped truth or an error perturbed by 0.1 %, must '
              'report a gap above its bar (perturbed: above 1e-4 when the restriction is not the identity)')

    # ---- parity -------------------------------------------------------------------------------------------
    par = r['parity']
    gate('parity_rotated_vs_unrotated', (not cfg.get('parity_pairs')) or (bool(par) and all(x['passed'] for x in par)),
         applicable=bool(cfg.get('parity_pairs')),
         pairs=[{k: x[k] for k in ('rotated', 'parent', 'worst_relative', 'integers_identical', 'passed')} for x in par],
         bar=cfg['parity_bar'])

    # ---- certificates, recomputed from the saved per-state rho ------------------------------------------
    setup = {x['arm']: x for x in r['arm_setup']}
    ps = r['population_source']
    nd, ci = ps['certification_draws'], ps.get('confirmation_draw_index')
    cert, mism = {}, []
    for name, c_ in r['certificates'].items():
        z = np.load(o / f'deployed_{name}.npz')
        j = setup[name]['exact_steps']
        per = []
        for d in range(len(ps['draws'])):
            sel = (z['meta'][:, 0] == d) & (z['meta'][:, 2] >= j)
            per.append(float(z['rho'][sel].max()))
        npass = sum(v <= BAR for v in per[:nd])
        conf = (per[ci] <= BAR) if ci is not None else None
        status = ('confirmed' if npass == nd and conf is not False else
                  'fails' if npass == 0 else f'not confirmed ({npass}/{nd} draws, confirmation {conf})')
        cert[name] = dict(status=status, rho_max_cert=max(per[:nd]), rho_max_confirmation=per[ci] if ci is not None else None,
                          per_draw=per, job_status=c_['status'])
        if status != c_['status']:
            mism.append(name)
    gate('certificate_status_recomputed_matches_job', not mism and bool(cert), mismatches=mism)
    ctrl = [n for n in cert if setup[n].get('control')]
    gate('certificate_control_fails', bool(ctrl) and all(cert[n]['status'] != 'confirmed' for n in ctrl),
         controls={n: cert[n]['status'] for n in ctrl})

    # ---- rho itself in NumPy for spot states ----------------------------------------------------------
    spot = []
    if not a.no_rho_spot and cert:
        ck = pickle.load(open(a.checkpoint, 'rb'))
        params = ck['params']
        rot = np.load(a.rotation)
        T, Lr = rot['T'], rot['L']
        rng = np.random.default_rng(20260923)
        want = [n for n in cert if setup[n]['model'] in ('trunc', 'lin', 'parent')]
        # every per-draw argmax of a spread of arms, plus random states; bounded so the grid pass stays cheap
        pick_arms = [n for n in want if setup[n]['R_prime'] in (512, 64, 32) or setup[n].get('control')][:8]
        todo = []
        for n in pick_arms:
            z = np.load(o / f'deployed_{n}.npz')
            idx = {int(np.argmax(np.where(z['meta'][:, 0] == d, z['rho'], -1))) for d in range(min(nd, 2))}
            idx |= set(rng.choice(len(z['rho']), a.spot_random, replace=False).tolist())
            for i in sorted(idx):
                todo.append((n, i, z['coefficients'][i], float(z['rho'][i]), z['meta'][i].tolist()))
        by_T = {}
        for n, i, c_, rj, meta in todo:
            st = setup[n]
            Rp = st['R_prime'] if st['model'] != 'parent' else R
            key = 'I' if st['model'] == 'parent' else Rp
            by_T.setdefault(key, []).append((n, i, c_, rj, meta))
        for key, items in by_T.items():
            Tp = np.eye(R) if key == 'I' else T[:, :key]
            U = fields_np(params, Tp, np.stack([x[2] for x in items]), L)
            for (n, i, c_, rj, meta), Ui in zip(items, U):
                st = setup[n]
                rz = np.load(o / f"rule_L{L}_{st['rule']}.npz")
                kx, ky = modes_np(L, st['M'])
                got = rho_np(Ui, L, kx, ky, rz['ij'], rz['weights'])
                spot.append(dict(arm=n, state=i, meta=meta, job=rj, numpy=got, rel_diff=abs(got - rj) / max(rj, 1e-300)))
            del U
    wr = max([x['rel_diff'] for x in spot] or [None]) if spot else None
    gate('rho_recomputed_in_numpy', a.no_rho_spot or (bool(spot) and wr <= 1e-7), worst_relative_diff=wr,
         states=len(spot), skipped=a.no_rho_spot)

    # ---- order effect (pre-registered, DESIGN section 5) -----------------------------------------------
    inv = r['invocations']
    arm_case_med = {}
    for x in inv:
        arm_case_med.setdefault((x['name'], x['case']), []).append(x['gpu_seconds'])
    arm_case_med = {k: med(v) for k, v in arm_case_med.items()}
    arm_med = {}
    for x in inv:
        arm_med.setdefault(x['name'], []).append(x['gpu_seconds'])
    arm_med = {k: med(v) for k, v in arm_med.items()}
    oe_long, oe_short, per_arm = [], [], {}
    legacy = {}
    for ph in dict.fromkeys(x['phase'] for x in inv):
        seq = sorted([x for x in inv if x['phase'] == ph], key=lambda x: x['seq'])
        for k in range(1, len(seq)):
            x, pv = seq[k], seq[k - 1]
            ratio = x['gpu_seconds'] / arm_case_med[(x['name'], x['case'])]
            longn = arm_med[pv['name']] >= 4 * arm_med[x['name']]
            (oe_long if longn else oe_short).append(ratio)
            d = per_arm.setdefault(x['name'], dict(long=[], short=[]))
            d['long' if longn else 'short'].append(ratio)
            lg = legacy.setdefault(x['name'], dict(long=[], short=[]))
            lg['long' if pv['gpu_seconds'] >= 1. else 'short'].append(x['gpu_seconds'])
    pooled = (med(oe_long) / med(oe_short) - 1) if oe_long and oe_short else None
    arms_oe = {n: dict(n_long=len(d['long']), n_short=len(d['short']),
                       gap=(med(d['long']) / med(d['short']) - 1) if d['long'] and d['short'] else None)
               for n, d in per_arm.items()}
    evaluable = {n: v for n, v in arms_oe.items() if v['n_long'] >= 6 and v['n_short'] >= 6}
    worst_arm = max([v['gap'] for v in evaluable.values()] or [0.])
    gate('no_order_effect', (pooled is None or pooled <= .03) and worst_arm <= .05, pooled_gap=pooled,
         pooled_n_long=len(oe_long), pooled_n_short=len(oe_short), worst_evaluable_arm_gap=worst_arm,
         evaluable_arms=len(evaluable), per_arm=arms_oe,
         legacy_repanel_definition={n: dict(n_long=len(v['long']), n_short=len(v['short']),
                                            gap=(med(v['long']) / med(v['short']) - 1) if v['long'] and v['short'] else None)
                                    for n, v in legacy.items()},
         note='pre-registered: per phase, each invocation normalised by its (arm, case) median; long neighbour = '
              'predecessor arm median >= 4x own; pooled gap <= 3 %, per arm (>= 6 samples each side) <= 5 %')

    # ---- per-arm table ---------------------------------------------------------------------------------
    quick = {}
    for x in r['quick']:
        quick.setdefault(x['name'], {})[x['case']] = x
    table = {}
    for n, pc in quick.items():
        xs = [x for x in inv if x['name'] == n]
        fam = next(iter(pc.values()))['family']
        t = dict(name=n, family=fam, cases=len(pc), timed=bool(xs),
                 reps=len(xs) // max(len(pc), 1),
                 median_gpu_ms=1e3 * med([x['gpu_seconds'] for x in xs]) if xs else None,
                 median_host_ms=1e3 * med([x['host_seconds'] for x in xs]) if xs else None,
                 worst_evolved_percent=100 * max(x['same_grid_evolved'] for x in pc.values()),
                 median_evolved_percent=100 * med([x['same_grid_evolved'] for x in pc.values()]),
                 worst_all_times_percent=100 * max(x['same_grid_all'] for x in pc.values()),
                 worst_current_relative_evolved_percent=100 * max(x['current_relative_evolved'] for x in pc.values()))
        if fam == 'rom':
            st = setup[n]
            t.update(model=st['model'], R_prime=st['R_prime'], q=st['q'], M=st['M'], m=st['m'], rule=st['rule'],
                     unknowns=st['unknowns'], gtol=st['gtol'], exact_steps=st['exact_steps'],
                     arm_family=st.get('arm_family'),
                     control=st.get('control'),
                     total_iterations_median=med([x['total_iterations'] for x in pc.values()]),
                     stalled_exits=int(sum(x['stalled_exits'] for x in pc.values())),
                     damping_retries=int(sum(x.get('damping_retries_total', 0) for x in pc.values())),
                     certificate=cert.get(n, {}).get('status'),
                     rho_max_cert=cert.get(n, {}).get('rho_max_cert'),
                     rho_max_confirmation=cert.get(n, {}).get('rho_max_confirmation'),
                     condition=st.get('correction_elimination_condition_number'))
        else:
            fs = next(f for f in cfg['fom_settings'] if f['name'] == n)
            t.update(dt=fs['dt'], ntol=fs['ntol'], ltol=fs['ltol'], impl=fs.get('impl', 'audited'),
                     stalled_steps=int(sum(x['stalled_steps'] for x in pc.values())),
                     nonlinear_converged=all(x['nonlinear_converged'] for x in pc.values()))
        table[n] = t
    foms = {n: t for n, t in table.items() if t['family'] == 'fom' and t['timed']}

    def fom_for(err, names=None):
        ok = [f for f in foms if (names is None or f in names) and foms[f]['nonlinear_converged']
              and foms[f]['worst_evolved_percent'] <= err]
        return min(ok, key=lambda f: foms[f]['median_gpu_ms']) if ok else None

    for n, t in table.items():
        if t['family'] != 'rom' or not t['timed']:
            continue
        f = fom_for(t['worst_evolved_percent'])
        t.update(fom_at_least_as_accurate=f, fom_gpu_ms=foms[f]['median_gpu_ms'] if f else None,
                 fom_worst_evolved_percent=foms[f]['worst_evolved_percent'] if f else None,
                 speedup_gpu=(foms[f]['median_gpu_ms'] / t['median_gpu_ms']) if f else None,
                 speedup_host=(foms[f]['median_host_ms'] / t['median_host_ms']) if f else None)
        t['by_grid'] = {}
        for gname, names in (cfg.get('fom_subsets') or {}).items():
            g = fom_for(t['worst_evolved_percent'], names)
            t['by_grid'][gname] = dict(fom=g, speedup_gpu=(foms[g]['median_gpu_ms'] / t['median_gpu_ms']) if g else None)

    # ---- the pre-registered setting rule (DESIGN section 6) ---------------------------------------------
    cand = {n: t for n, t in table.items() if t['family'] == 'rom' and t['timed'] and t['model'] != 'parent'
            and not t['control']}
    conf = {n: t for n, t in cand.items() if t['certificate'] == 'confirmed'}
    sel = dict(mesh=L, rule='DESIGN.md section 6', candidates=len(cand), confirmed=len(conf))
    acc = min(conf, key=lambda n: t_err(table[n])) if conf else None
    acc_any = min(cand, key=lambda n: t_err(table[n])) if cand else None
    ref_fast = next((n for n, t in cand.items() if t['model'] == 'trunc' and t['R_prime'] == R and t['q'] == 0), None)
    ref_acc = next((n for n, t in cand.items() if t['model'] == 'trunc' and t['R_prime'] == R and t['q'] > 0), None)
    ferr = table[ref_fast]['worst_evolved_percent'] if ref_fast else None
    fast_ok = {n: t for n, t in conf.items() if ferr is not None and t['worst_evolved_percent'] <= ferr}
    fast = min(fast_ok, key=lambda n: table[n]['median_gpu_ms']) if fast_ok else None
    fast_any_ok = {n: t for n, t in cand.items() if ferr is not None and t['worst_evolved_percent'] <= ferr}
    fast_any = min(fast_any_ok, key=lambda n: table[n]['median_gpu_ms']) if fast_any_ok else None

    def row(n):
        if n is None:
            return None
        t = table[n]
        return {k: t.get(k) for k in ('name', 'model', 'R_prime', 'q', 'M', 'm', 'unknowns', 'rule', 'gtol',
                                      'worst_evolved_percent', 'median_evolved_percent', 'median_gpu_ms',
                                      'median_host_ms', 'certificate', 'rho_max_cert', 'rho_max_confirmation',
                                      'fom_at_least_as_accurate', 'fom_gpu_ms', 'fom_worst_evolved_percent',
                                      'speedup_gpu', 'speedup_host', 'stalled_exits')}
    sel.update(accurate=row(acc), accurate_if_certificates_ignored=row(acc_any) if acc_any != acc else None,
               fast_reference_arm=ref_fast, fast_reference_error_percent=ferr, fast=row(fast),
               fast_if_certificates_ignored=row(fast_any) if fast_any != fast else None,
               current_accurate_arm=row(ref_acc), current_fast_arm=row(ref_fast))
    if acc:
        f = fom_for(table[acc]['worst_evolved_percent'])
        sel['table1'] = dict(fom=f, fom_gpu_ms=foms[f]['median_gpu_ms'] if f else None,
                             rom_gpu_ms=table[acc]['median_gpu_ms'],
                             speedup_gpu=(foms[f]['median_gpu_ms'] / table[acc]['median_gpu_ms']) if f else None)
    # can q be dropped? some (a)/(b) arm with a confirmed rule reaches the current accurate arm's error
    if ref_acc:
        target = table[ref_acc]['worst_evolved_percent']
        noq = {n: t for n, t in conf.items() if t['model'] == 'lin' or t['q'] == 0}
        best = min(noq, key=lambda n: t_err(table[n])) if noq else None
        noq_any = {n: t for n, t in cand.items() if t['model'] == 'lin' or t['q'] == 0}
        best_any = min(noq_any, key=lambda n: t_err(table[n])) if noq_any else None
        sel['q_droppable'] = dict(target_arm=ref_acc, target_percent=target, best_without_q=row(best),
                                  best_without_q_ignoring_certificates=row(best_any),
                                  verdict=bool(best and table[best]['worst_evolved_percent'] <= target))
    failed = [k for k, v in gates.items() if not v['passed']]
    summary = dict(attempt=cfg['attempt'], job_id=r['job_id'], commit=r['commit'], gpu=r['gpu'], intervals=L,
                   cohort=r['cohort_name'], cohort_cases=len(cases), elapsed_seconds=r.get('elapsed_seconds'),
                   rotation=r['rotation'], gates=gates, failed_gates=failed, accepted=not failed, selection=sel,
                   table=table, certificates=cert, rho_spot=spot, full_grid=full, parity=par,
                   phases=r['phases'].get('timed_phases'),
                   sources=dict(result_json_sha256=sha(o / 'result.json')))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1, allow_nan=False, default=float) + '\n')
    print('failed gates:', failed)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('name', 'worst_evolved_percent', 'median_gpu_ms', 'speedup_gpu', 'certificate', 'verdict', 'fom', 'target_percent')})
                      for k, v in sel.items()}, indent=1))


def t_err(t):
    return (t['worst_evolved_percent'], t['median_gpu_ms'] or 0.)


if __name__ == '__main__':
    main()
