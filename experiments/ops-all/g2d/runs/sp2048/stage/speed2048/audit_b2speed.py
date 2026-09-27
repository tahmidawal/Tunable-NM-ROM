"""Independent NumPy audit + pre-registered selection for one burgers2d-speed attempt (no JAX anywhere in this file).
Helpers (features_np, rho_np, ...) are copied from experiments/burgers-bank-knob/audit_bankknob.py @ b5c843ab.

From the SAVED outputs, the checkpoint and the committed rotation only, it
  * recomputes every (arm, case) evolved error on the restricted grid and, for the audit case, on the full grid;
  * re-derives every certificate status from the saved per-state rho (k >= max(j,1) primary; k >= j+1 beside it)
    and recomputes rho in NumPy for spot states (argmax cert, argmax confirmation, 2 random, every certified knob);
  * recomputes the coefficient map of every ROM arm (case 0) in NumPy against the saved field;
  * checks parity, the fast-bar reproduction, repetitions, and recomputes the A-B-A drift and neighbour gates from
    the raw invocations;
  * builds the knob table (time = faster compile mode), the FOM table (fastest setting+mode), applies the
    pre-registered rule of DESIGN.md section 6, and the coordinator's general-path factor X (1024^2);
  * runs injected controls (swapped case, perturbed error, perturbed A2 time) that must be DETECTED.

    python audit_b2speed.py runs/<attempt>/archive --checkpoint <pkl> --rotation <npz> --directions <npz> \
        --out checks/<attempt>-summary.json [--write-selection selection-<L>.json]
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


def fields_multi(params, groups, L, chunk=65536):
    """One pass over the interior grid for several (T_p, coefs) groups: [(S_g, L-1, L-1)]."""
    x = np.arange(1, L) / L
    xy = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    Ws = [Tp @ c.T for Tp, c in groups]
    outs = [np.empty((len(xy), c.shape[0])) for _, c in groups]
    for s in range(0, len(xy), chunk):
        F = features_np(params, xy[s:s + chunk])
        for W, o in zip(Ws, outs):
            o[s:s + chunk] = F @ W
    return [o.T.reshape(-1, L - 1, L - 1) for o in outs]


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



def strip_mode(n):
    return n[:-len('__graphs')] if n.endswith('__graphs') else n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rotation', required=True)
    p.add_argument('--directions', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--write-selection')
    p.add_argument('--spot-random', type=int, default=2)
    p.add_argument('--no-rho-spot', action='store_true')
    a = p.parse_args()
    arc = Path(a.archive)
    o = arc / 'output'
    r = json.loads((o / 'result.json').read_text())
    cfg = r['config']
    L, R, K = r['intervals'], r['R'], r['K']
    tight = cfg['same_grid_reference']
    lim = cfg['gate_limit']
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
    gate('directions_file', r['directions']['sha256'] == cfg['directions_sha256'] == sha(a.directions))
    for k in ('phi_free_operator_parity', 'repetition_output_identical', 'retained_repetitions'):
        if k in r['gates']:
            gate(k, r['gates'][k]['passed'], job=r['gates'][k])
    sl = r['gates'].get('sep_lattice_projection', [])
    gate('sep_lattice_projection', all(x['passed'] for x in sl), rows=sl)
    setup = {x['arm']: x for x in r['arm_setup']}
    inv = r['invocations']

    # ---- repetitions recomputed ----
    cnt = {}
    for x in inv:
        cnt[(x['name'], x['case'])] = cnt.get((x['name'], x['case']), 0) + 1
    need = lambda n: 2 * cfg['rom_reps'] if n in setup else cfg['fom_reps']
    gate('retained_repetitions_recomputed', bool(cnt) and all(v >= need(k[0]) for k, v in cnt.items()),
         minimum=min(cnt.values()) if cnt else 0, phases=sorted({x['phase'] for x in inv}))

    # ---- errors ----
    cases = sorted({x['case'] for x in r['quick']})
    truth_r = {c: np.load(o / f'restricted_{tight}_case{c}.npz')['fields'] for c in cases}
    exact_restriction = max(1, L // cfg.get('restrict_to', 256)) == 1

    def compare(xrows, truth, ratios=None):
        gap = 0.
        for x in xrows:
            fpth = o / f"restricted_{x['name']}_case{x['case']}.npz"
            if not fpth.exists() and cfg.get('save_restricted_skip_graphs_fom') and x['name'].endswith('__graphs'):
                fpth = o / f"restricted_{strip_mode(x['name'])}_case{x['case']}.npz"   # SHA-identical (gated)
            f = np.load(fpth)['fields']
            b = truth[x['case']]
            sg = np.linalg.norm((f - b).reshape(len(f), -1), axis=1) / np.linalg.norm(b[0])
            if x['same_grid_evolved'] > 1e-6:
                gap = max(gap, abs(sg[1:].max() / x['same_grid_evolved'] - 1))
                if ratios is not None:
                    ratios.append(sg[1:].max() / x['same_grid_evolved'])
        return gap
    bar_r = 1e-9 if exact_restriction else 0.05
    ratios = []
    worst_gap = compare(r['quick'], truth_r, ratios)
    ok_r = worst_gap < bar_r if exact_restriction else (bool(ratios) and min(ratios) >= .5 and max(ratios) <= 1.05)
    gate('restricted_recomputation_tracks_job', ok_r, worst_relative_gap=worst_gap, ratio_min=min(ratios or [None]),
         ratio_max=max(ratios or [None]), restriction_is_identity=exact_restriction,
         rule='identity restriction: <= 1e-9; else restricted/job ratio in [0.5, 1.05] (parent amendment A1)')
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
    gate('full_grid_errors_recomputed', bool(full) and all(x['abs_diff'] <= 1e-12 for x in full),
         worst_abs_diff=max([x['abs_diff'] for x in full] or [None]), arms=len(full))

    # ---- parity ----
    par = r['parity']
    gate('parity', (not cfg.get('parity_pairs')) or (len(par) == len(cfg['parity_pairs']) and all(x['passed'] for x in par)),
         pairs=[{k: x[k] for k in ('engineered', 'parent', 'worst_relative', 'integers_identical', 'passed')} for x in par],
         bar=cfg['parity_bar'])

    qsha = {(x['name'], x['case']): x['field_sha256'] for x in r['quick']}
    ncs = sorted({x['case'] for x in r['quick']})
    fsha = {fs['name']: all(qsha.get((fs['name'] + '__graphs', c)) == qsha.get((fs['name'], c)) for c in ncs)
            for fs in cfg['fom_settings'] if (fs['name'] + '__graphs', ncs[0]) in qsha}
    gate('fom_mode_field_sha_identical', (not cfg.get('fom_both_modes', True)) or
         (len(fsha) == len(cfg['fom_settings']) and all(fsha.values())), rows=fsha)
    fp = r.get('fom_mode_parity', [])
    gate('fom_mode_parity', (not cfg.get('fom_both_modes', True)) or (not cfg.get('fom_mode_parity', True)) or (len(fp) == len(cfg['fom_settings']) and
                                                                      all(x['passed'] for x in fp)),
         rows=[{k: x[k] for k in ('setting', 'worst_relative', 'newton_identical', 'passed')} for x in fp])

    # ---- certificates ----
    ps = r['population_source']
    nd, ci = ps['certification_draws'], ps.get('confirmation_draw_index')

    def status_of(z, kmin):
        per = [float(z['rho'][(z['meta'][:, 0] == d) & (z['meta'][:, 2] >= kmin)].max()) for d in range(len(ps['draws']))]
        npass = sum(v <= BAR for v in per[:nd])
        conf = (per[ci] <= BAR) if ci is not None else None
        st = ('confirmed' if npass == nd and conf is not False else
              'fails' if npass == 0 else f'not confirmed ({npass}/{nd} draws, confirmation {conf})')
        return st, per
    cert, mism = {}, []
    for name, c_ in r['certificates'].items():
        z = np.load(o / f'deployed_{name}.npz')
        j = setup[name]['exact_steps']
        st, per = status_of(z, max(j, 1))
        st2, per2 = status_of(z, j + 1)
        cert[name] = dict(status=st, rho_max_cert=max(per[:nd]), rho_max_confirmation=per[ci] if ci is not None else None,
                          per_draw=per, job_status=c_['status'], sensitivity_k_ge_j_plus_1=dict(
                              status=st2, rho_max_cert=max(per2[:nd]), rho_max_confirmation=per2[ci] if ci is not None else None))
        if st != c_['status'] or st2 != c_['sensitivity_k_ge_j_plus_1']['status']:
            mism.append(name)
    CERT_RUN = not cfg.get('skip_certificates')
    gate('certificate_status_recomputed_matches_job', (not CERT_RUN) or (not mism and bool(cert)), mismatches=mism)

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = ck['params']
    rot = np.load(a.rotation)
    T, Lr = rot['T'], rot['L']
    spot = []
    if not a.no_rho_spot and cert:
        rng = np.random.default_rng(20260924)
        todo = []
        for n in cert:
            z = np.load(o / f'deployed_{n}.npz')
            m_ = z['meta']
            idx = {int(np.argmax(np.where(m_[:, 0] < nd, z['rho'], -1)))}
            if ci is not None:
                idx.add(int(np.argmax(np.where(m_[:, 0] == ci, z['rho'], -1))))
            idx |= set(rng.choice(len(z['rho']), a.spot_random, replace=False).tolist())
            for i in sorted(idx):
                todo.append((n, i, z['coefficients'][i], float(z['rho'][i]), m_[i].tolist()))
        by_T = {}
        for n, i, c_, rj, meta in todo:
            by_T.setdefault(setup[n]['R_prime'], []).append((n, i, c_, rj, meta))
        keys = list(by_T)
        Us = fields_multi(params, [(T[:, :k], np.stack([x[2] for x in by_T[k]])) for k in keys], L)
        for key, U in zip(keys, Us):
            for (n, i, c_, rj, meta), Ui in zip(by_T[key], U):
                st = setup[n]
                rz = np.load(o / f"rule_L{L}_{st['rule']}.npz")
                kx, ky = modes_np(L, st['M'])
                got = rho_np(Ui, L, kx, ky, rz['ij'], rz['weights'])
                spot.append(dict(arm=n, state=i, meta=meta, job=rj, numpy=got, rel_diff=abs(got - rj) / max(rj, 1e-300)))
        del Us
    wr = max([x['rel_diff'] for x in spot] or [None]) if spot else None
    gate('rho_recomputed_in_numpy', a.no_rho_spot or (not CERT_RUN) or (bool(spot) and wr <= 1e-7), worst_relative_diff=wr,
         states=len(spot))

    # ---- coefficient map in NumPy (case 0) ----
    Cq = np.asarray(np.load(a.directions)['C'])
    sub = max(1, L // cfg.get('restrict_to', 256))
    xr = np.arange(sub, L, sub) / L
    xyr = np.stack(np.meshgrid(xr, xr, indexing='ij'), -1).reshape(-1, 2)
    Fr = features_np(params, xyr)
    cmap = []
    for x in r['quick']:
        if x['family'] != 'rom' or x['case'] != 0:
            continue
        st = setup[x['name']]
        zf = o / f"restricted_{x['name']}_case0.npz"
        if not zf.exists():
            continue
        z = np.load(zf)
        W = z['internal_latents'][::int(round(.05 / r['dt']))]
        if st['model'] == 'lin':
            coef = W
        else:
            coef = np.stack([head_np(params, w[:K]) + Cq[:, :st['q']] @ w[K:] for w in W]) @ Lr[:st['R_prime']].T
        U = (Fr @ (T[:, :st['R_prime']] @ coef.T)).T.reshape(len(W), len(xr), len(xr))
        saved = z['fields'][:, 1:-1, 1:-1]
        cmap.append(dict(arm=x['name'], relative=float(np.linalg.norm(U - saved) / np.linalg.norm(saved))))
    wc = max([c_['relative'] for c_ in cmap] or [None]) if cmap else None
    gate('coefficient_map_recomputed_in_numpy', (not cmap and not cfg.get('save_restricted', True)) or
         (bool(cmap) and wc <= 1e-9), worst_relative=wc, arms=len(cmap))

    # ---- A-B-A gates recomputed from raw invocations ----
    def medph(nm, ph, rows=inv):
        v = [x['gpu_seconds'] for x in rows if x['name'] == nm and x['phase'] == ph]
        return med(v) if v else None
    rom_timed = sorted({x['name'] for x in inv if x['phase'] == 'romA1'})
    fom_timed = sorted({x['name'] for x in inv if x['phase'] == 'fomB'})

    def drift_rows(rows):
        out = []
        for nm in rom_timed:
            a1, a2 = medph(nm, 'romA1', rows), medph(nm, 'romA2', rows)
            out.append(dict(name=nm, romA1_median=a1, romA2_median=a2, ratio=a2 / a1))
        return out
    drift = drift_rows(inv)
    gate('drift_ABA', bool(drift) and all(1 / lim <= d['ratio'] <= lim for d in drift), limit=lim,
         worst=max((max(d['ratio'], 1 / d['ratio']) for d in drift), default=None), rows=drift)
    def neighbour_rows(rows_inv):
        nb = []
        for label, subs in (('romA1', rom_timed), ('romA2', rom_timed), ('fomB', fom_timed)):
            ivp = [x for x in rows_inv if x['phase'] == label]
            medians = {nm: medph(nm, label, rows_inv) for nm in subs}
            cm = {}
            for x in ivp:
                cm.setdefault((x['name'], x['case']), []).append(x['gpu_seconds'])
            cm = {k: med(v) for k, v in cm.items()}
            cut = np.quantile(list(medians.values()), [1 / 3, 2 / 3])
            for nm in subs:
                mine = [x for x in ivp if x['name'] == nm and x.get('previous') is not None]
                lo = [x['gpu_seconds'] / cm[(nm, x['case'])] for x in mine if medians[x['previous']] <= cut[0]]
                hi = [x['gpu_seconds'] / cm[(nm, x['case'])] for x in mine if medians[x['previous']] >= cut[1]]
                ev = len(lo) >= 3 and len(hi) >= 3
                nb.append(dict(name=nm, phase=label, ratio=float(np.mean(hi) / np.mean(lo)) if ev else None,
                               n_short=len(lo), n_long=len(hi), evaluable=ev))
        return nb
    nb = neighbour_rows(inv)
    evn = [x for x in nb if x['evaluable']]
    gate('neighbour', bool(evn) and all(x['ratio'] <= lim for x in evn), limit=lim,
         worst=max((x['ratio'] for x in evn), default=None), evaluable=len(evn),
         not_evaluable=[(x['name'], x['phase']) for x in nb if not x['evaluable']], rows=nb,
         rule='case-controlled: invocation / its (subject, case, phase) median; mean after long / after short')

    # ---- tables ----
    quick = {}
    for x in r['quick']:
        quick.setdefault(x['name'], {})[x['case']] = x
    table = {}
    for n, pc in quick.items():
        xs = [x for x in inv if x['name'] == n]
        fam = next(iter(pc.values()))['family']
        t = dict(name=n, family=fam, cases=len(pc), timed=bool(xs), invocations=len(xs),
                 median_gpu_ms=1e3 * med([x['gpu_seconds'] for x in xs]) if xs else None,
                 median_host_ms=1e3 * med([x['host_seconds'] for x in xs]) if xs else None,
                 worst_evolved_percent=100 * max(x['same_grid_evolved'] for x in pc.values()),
                 median_evolved_percent=100 * med([x['same_grid_evolved'] for x in pc.values()]),
                 per_case_evolved_percent=[100 * pc[c]['same_grid_evolved'] for c in sorted(pc)])
        if fam == 'rom':
            st = setup[n]
            t.update({k: st.get(k) for k in ('model', 'R_prime', 'q', 'M', 'm', 'rule', 'unknowns', 'gtol', 'exact_steps',
                                             'cap', 'impl', 'graphs', 'role', 'candidate')})
            t['knob'] = st['knob']
            t.update(A1_median_ms=1e3 * medph(n, 'romA1') if xs else None,
                     A2_median_ms=1e3 * medph(n, 'romA2') if xs else None,
                     total_iterations_per_case=[pc[c]['total_iterations'] for c in sorted(pc)],
                     max_iterations_per_step=max(pc[c]['max_iterations'] for c in pc),
                     stalled_exits=int(sum(x['stalled_exits'] for x in pc.values())),
                     budget_exits=int(sum(x['budget_exits'] for x in pc.values())),
                     rejected_steps=int(sum(x.get('damping_retries_total', 0) for x in pc.values())))
        else:
            fs = next(f for f in cfg['fom_settings'] if f['name'] == strip_mode(n))
            t.update(setting=fs['name'], graphs=n.endswith('__graphs'), dt=fs['dt'], ntol=fs['ntol'], ltol=fs['ltol'],
                     impl=fs.get('impl', 'audited'),
                     newton_iterations_per_case=[pc[c]['newton_iterations_total'] for c in sorted(pc)],
                     nonlinear_converged=all(x['nonlinear_converged'] for x in pc.values()))
        table[n] = t
    foms = {n: t for n, t in table.items() if t['family'] == 'fom' and t['timed']}
    fom_err_same = all(abs(table[n]['worst_evolved_percent'] - table[strip_mode(n)]['worst_evolved_percent']) <= 1e-12
                       for n in table if n.endswith('__graphs'))
    gate('fom_modes_same_error', fom_err_same)

    def fom_for(err):
        ok = [f for f in foms if foms[f]['nonlinear_converged'] and foms[f]['worst_evolved_percent'] <= err]
        return min(ok, key=lambda f: foms[f]['median_gpu_ms']) if ok else None

    # knobs: candidates grouped by knob; time = faster mode; error must agree across modes
    knobs = {}
    for n, t in table.items():
        if t['family'] == 'rom' and t.get('candidate') and t['timed']:
            knobs.setdefault(json.dumps(t['knob']), []).append(n)
    ktab, err_mismatch = {}, []
    for k, names in knobs.items():
        errs = {table[n]['worst_evolved_percent'] for n in names}
        if max(errs) - min(errs) > 1e-12:
            err_mismatch.append(k)
        best = min(names, key=lambda n: table[n]['median_gpu_ms'])
        cn = next((n for n in names if n in cert), None)
        t = dict(knob=json.loads(k), arms=names, timed_arm=best, mode='graphs' if table[best]['graphs'] else 'default',
                 median_gpu_ms=table[best]['median_gpu_ms'], median_host_ms=table[best]['median_host_ms'],
                 other_mode_ms={n: table[n]['median_gpu_ms'] for n in names if n != best},
                 worst_evolved_percent=table[best]['worst_evolved_percent'],
                 median_evolved_percent=table[best]['median_evolved_percent'],
                 certified_arm=cn, certificate=cert.get(cn, {}).get('status'),
                 rho_max_cert=cert.get(cn, {}).get('rho_max_cert'),
                 rho_max_confirmation=cert.get(cn, {}).get('rho_max_confirmation'),
                 certificate_k_ge_j_plus_1=cert.get(cn, {}).get('sensitivity_k_ge_j_plus_1', {}).get('status'),
                 **{kk: table[best][kk] for kk in ('model', 'R_prime', 'q', 'M', 'm', 'rule', 'unknowns', 'gtol',
                                                   'exact_steps', 'cap', 'total_iterations_per_case',
                                                   'max_iterations_per_step', 'stalled_exits', 'budget_exits')})
        f = fom_for(t['worst_evolved_percent'])
        t.update(own_fom=f, own_fom_ms=foms[f]['median_gpu_ms'] if f else None,
                 own_speedup=(foms[f]['median_gpu_ms'] / t['median_gpu_ms']) if f else None)
        ktab[best] = t
    gate('knob_error_identical_across_modes', not err_mismatch, mismatches=err_mismatch)

    # ---- fast bar reproduction ----
    pb = cfg['parent_fast_bar']
    twin = next((n for n, t in table.items() if t['family'] == 'rom' and t.get('impl') == 'parent'
                 and t['model'] == 'trunc' and t['q'] == 0 and t['R_prime'] == 512 and not t['cap']
                 and 'budget' not in n), None)
    got = table[twin]['worst_evolved_percent'] if twin else None
    heldout = bool(cfg.get('skip_certificates'))
    if not heldout:
        gate('fast_bar_reproduces', twin is not None and pb['percent'] is not None and abs(got / pb['percent'] - 1) <= 1e-6,
             parent=pb['percent'], this_job=got, arm=twin)
    ferr = got if got is not None else -1.   # DESIGN section 6: the fast bar is the q=0 setting's error re-measured in this job

    if heldout:
        # held-out: the frozen dev6 picks are REPORTED whatever they show; no selection, no dev-cohort checks
        hs = cfg['heldout_selection']
        acc_n, fast_n = hs['accurate'], hs['fast']
        f = fom_for(table[acc_n]['worst_evolved_percent'])
        rowh = lambda n: {k: table[n].get(k) for k in ('name', 'worst_evolved_percent', 'median_evolved_percent',
                                                       'median_gpu_ms', 'A1_median_ms', 'A2_median_ms',
                                                       'total_iterations_per_case', 'max_iterations_per_step',
                                                       'stalled_exits', 'budget_exits')}
        sel = dict(mesh=L, rule='held-out: frozen dev6 selection (selection-<L>.json), reported as measured',
                   accurate=rowh(acc_n), fast=rowh(fast_n),
                   table1=dict(fom=f, fom_gpu_ms=foms[f]['median_gpu_ms'] if f else None,
                               fom_worst_evolved_percent=foms[f]['worst_evolved_percent'] if f else None,
                               accurate_speedup=foms[f]['median_gpu_ms'] / table[acc_n]['median_gpu_ms'] if f else None,
                               fast_speedup=foms[f]['median_gpu_ms'] / table[fast_n]['median_gpu_ms'] if f else None),
                   parent_settings_this_job={n: dict(rowh(n), speedup_vs_table1_fom=(foms[f]['median_gpu_ms'] /
                                                                                       table[n]['median_gpu_ms']) if f else None)
                                             for n, t in table.items() if t['family'] == 'rom' and t.get('impl') == 'parent'},
                   own_fom={n: fom_for(table[n]['worst_evolved_percent']) for n in (acc_n, fast_n)})
    # ---- the pre-registered rule ----
    conf = {n: t for n, t in ktab.items() if t['certificate'] == 'confirmed'} if not heldout else {}
    key_err = lambda n: (ktab[n]['worst_evolved_percent'], ktab[n]['median_gpu_ms'])
    acc = min(conf, key=key_err) if conf else None
    acc_any = min(ktab, key=key_err) if ktab else None
    fast_ok = [n for n in conf if ktab[n]['worst_evolved_percent'] <= ferr]
    fast = min(fast_ok, key=lambda n: ktab[n]['median_gpu_ms']) if fast_ok else None
    fast_any_ok = [n for n in ktab if ktab[n]['worst_evolved_percent'] <= ferr]
    fast_any = min(fast_any_ok, key=lambda n: ktab[n]['median_gpu_ms']) if fast_any_ok else None
    selh = sel if heldout else None
    sel = dict(mesh=L, rule='DESIGN.md section 6', knobs=len(ktab), confirmed=len(conf), fast_bar_percent=ferr,
               accurate=ktab.get(acc), fast=ktab.get(fast),
               accurate_if_certificates_ignored=ktab.get(acc_any) if acc_any != acc else None,
               fast_if_certificates_ignored=ktab.get(fast_any) if fast_any != fast else None,
               fast_near_ties={n: dict(ms=ktab[n]['median_gpu_ms'], err=ktab[n]['worst_evolved_percent'])
                               for n in fast_ok if fast and n != fast and ktab[n]['median_gpu_ms'] <= 1.05 * ktab[fast]['median_gpu_ms']})
    if acc:
        f = fom_for(ktab[acc]['worst_evolved_percent'])
        sel['table1'] = dict(fom=f, fom_gpu_ms=foms[f]['median_gpu_ms'] if f else None,
                             fom_worst_evolved_percent=foms[f]['worst_evolved_percent'] if f else None,
                             accurate_ms=ktab[acc]['median_gpu_ms'], fast_ms=ktab[fast]['median_gpu_ms'] if fast else None,
                             accurate_speedup=(foms[f]['median_gpu_ms'] / ktab[acc]['median_gpu_ms']) if f else None,
                             fast_speedup=(foms[f]['median_gpu_ms'] / ktab[fast]['median_gpu_ms']) if f and fast else None)
    # the parent's Table-1 settings measured in this job (before): parent text, default compile
    def twin_knob(n, t):
        """The engineered knob a parent arm realises: an LM budget of 1 is the cap-1 knob (Codex results audit)."""
        k = list(t['knob'])
        if '_budget1' in n:
            k[-1] = 1
        return k
    before = {}
    for n, t in table.items():
        if t['family'] == 'rom' and t.get('impl') == 'parent':
            f = fom_for(ktab[acc]['worst_evolved_percent']) if acc else None
            before[n] = dict(median_gpu_ms=t['median_gpu_ms'], worst_evolved_percent=t['worst_evolved_percent'],
                             speedup_vs_table1_fom=(foms[f]['median_gpu_ms'] / t['median_gpu_ms']) if f else None,
                             total_iterations_per_case=t['total_iterations_per_case'],
                             engineered_same_knob={m: dict(ms=table[m]['median_gpu_ms'],
                                                           factor=t['median_gpu_ms'] / table[m]['median_gpu_ms'])
                                                   for m in table if table[m]['family'] == 'rom' and m != n
                                                   and table[m].get('impl') == 'eng' and table[m]['timed']
                                                   and table[m].get('knob') == twin_knob(n, t)})
    sel['parent_settings_this_job'] = before
    # coordinator: general path vs fast path (paper section 6.3)
    gen = next((n for n, t in table.items() if t['family'] == 'rom' and t.get('impl') == 'general'), None)
    if gen:
        fastp = next(n for n, t in table.items() if t.get('impl') == 'parent' and t['model'] == 'lin' and 'budget' not in n
                     and t['R_prime'] == table[gen]['R_prime'] and not t['exact_steps'])
        sel['general_path_section_6_3'] = dict(
            general_arm=gen, fast_path_arm=fastp, general_ms=table[gen]['median_gpu_ms'],
            fast_path_ms=table[fastp]['median_gpu_ms'], X=table[gen]['median_gpu_ms'] / table[fastp]['median_gpu_ms'],
            general_worst_evolved_percent=table[gen]['worst_evolved_percent'],
            fast_path_worst_evolved_percent=table[fastp]['worst_evolved_percent'],
            general_per_case=table[gen]['per_case_evolved_percent'], fast_per_case=table[fastp]['per_case_evolved_percent'],
            general_iterations=table[gen]['total_iterations_per_case'],
            fast_iterations=table[fastp]['total_iterations_per_case'],
            same_error_criterion='pre-registered: |worst evolved error (general) / (fast path) - 1| <= 0.05',
            same_error=bool(abs(table[gen]['worst_evolved_percent'] / table[fastp]['worst_evolved_percent'] - 1) <= .05))
    if acc and fast:
        sel['parity_pairs'] = cfg.get('parity_pairs', [])
        sel['accurate_arm'], sel['fast_arm'] = ktab[acc]['timed_arm'], ktab[fast]['timed_arm']

    if heldout:
        sel = selh
    # ---- injected controls ----
    roms = [x for x in r['quick'] if x['family'] == 'rom']
    probe = [x for x in roms if x['same_grid_evolved'] > 1e-6][:6]
    # swapped case through the ACTUAL restricted-error predicate (identity: gap <= 1e-9; else ratio in [0.5, 1.05])
    if len(cases) > 1:
        sw_ratios = []
        gap_swap = compare(probe, {c: truth_r[cases[(i + 1) % len(cases)]] for i, c in enumerate(cases)}, sw_ratios)
        swap_rejected = (gap_swap >= bar_r) if exact_restriction else not (min(sw_ratios) >= .5 and max(sw_ratios) <= 1.05)
    else:
        gap_swap, swap_rejected = None, True
    # a 1e-6 relative perturbation of a reported error through the ACTUAL full-grid predicate (abs diff <= 1e-12)
    pert_rejected = bool(full) and not all(abs(x['recomputed_evolved'] - x['job_evolved'] * (1 + 1e-6)) <= 1e-12
                                           for x in full if x['job_evolved'] > 1e-6)
    # identity restriction: also through the restricted predicate
    if exact_restriction:
        pert_rejected = pert_rejected and compare([dict(x, same_grid_evolved=x['same_grid_evolved'] * (1 + 1e-6))
                                                   for x in probe], truth_r) >= bar_r
    victim = rom_timed[0] if rom_timed else None
    pinv = [dict(x, gpu_seconds=x['gpu_seconds'] * 1.2) if (x['name'] == victim and x['phase'] == 'romA2') else x for x in inv]
    drift_rejected = victim is not None and not all(1 / lim <= d['ratio'] <= lim for d in drift_rows(pinv))
    gate('controls_detected', swap_rejected and pert_rejected and drift_rejected, swapped_case_gap=gap_swap,
         swapped_case_rejected=swap_rejected, perturbed_error_1em6_rejected=pert_rejected,
         perturbed_A2_time_x1p2_rejected=drift_rejected,
         note='each injected control is run through the same predicate that accepts the job and must be rejected')

    failed = [k for k, v in gates.items() if not v['passed']]
    summary = dict(attempt=cfg['attempt'], job_id=r['job_id'], commit=r['commit'], gpu=r['gpu'], host=r.get('host'),
                   intervals=L, cohort=r['cohort_name'], cohort_cases=len(cases), elapsed_seconds=r.get('elapsed_seconds'),
                   gates=gates, failed_gates=failed, accepted=not failed, selection=sel, knobs=ktab, table=table,
                   certificates=cert, rho_spot=spot, full_grid=full, coefficient_map=cmap,
                   sources=dict(result_json_sha256=sha(o / 'result.json')))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1, allow_nan=False, default=float) + '\n')
    print('failed gates:', failed)
    for k in ('accurate', 'fast'):
        v = sel.get(k)
        print(k, v and (v.get('timed_arm', v.get('name')), round(v['worst_evolved_percent'], 4), round(v['median_gpu_ms'], 2),
                        v.get('certificate')))
    print('table1', sel.get('table1'))
    if 'general_path_section_6_3' in sel:
        print('general path', {k: v for k, v in sel['general_path_section_6_3'].items() if 'per_case' not in k})
    if a.write_selection:
        assert not failed and acc and fast, 'selection is written only from an accepted job with both picks'
        tw = [n for n, t in table.items() if t.get('impl') == 'parent']
        pacc = next(n for n in tw if table[n]['model'] == 'lin' and table[n]['R_prime'] == 384)
        pfast = {256: 128, 512: None, 1024: 128}[L]
        pfast = (next(n for n in tw if table[n]['model'] == 'lin' and table[n]['R_prime'] == 128) if pfast else
                 next(n for n in tw if table[n]['model'] == 'trunc'))
        Path(a.write_selection).write_text(json.dumps(dict(
            mesh=L, accurate=ktab[acc]['timed_arm'], fast=ktab[fast]['timed_arm'], parent_accurate_twin=pacc,
            parent_fast_twin=pfast, parity_pairs=cfg.get('parity_pairs', []),
            source_summary_sha256=hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
            source_attempt=cfg['attempt'], source_job=r['job_id']), indent=1) + '\n')


if __name__ == '__main__':
    main()
