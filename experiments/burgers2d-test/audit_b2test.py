"""Independent NumPy audit of one burgers2d-test attempt (no JAX anywhere in this file). It is
experiments/burgers2d-speed/audit_b2speed.py @ fb4a9ff7 (every gate up to the knob table verbatim) with the
dev-cohort selection replaced by the test-panel report of DESIGN.md sections 4-5:
  * the PRIMARY arms (accurate / fast / extra, frozen in the config) are reported whatever they show;
  * the Table-1 FOM = fastest timed, converged (setting, mode) at least as accurate as the accurate arm, same job;
  * reproduction gate vs the earlier held-out job at this mesh (same 64 cases): shared arm and FOM worst errors <= 1e-6
    relative;
  * 2048^2: secondary eng arms vs their parent twins (per-case error <= 1e-8 relative, identical total iterations).

    python audit_b2test.py runs/<attempt>/archive --checkpoint <pkl> --rotation <npz> --directions <npz> \
        --out checks/<attempt>-summary.json [--prior <earlier held-out summary json>]
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
    p.add_argument('--prior', help='earlier held-out summary on the same 64 cases (reproduction gate)')
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

    # ---- test panel (DESIGN.md section 4): frozen arms reported as measured ----
    ta = cfg['test_arms']
    prim = ta['primary']
    f1 = fom_for(table[prim['accurate']]['worst_evolved_percent'])

    def rowt(n):
        t = table[n]
        own = fom_for(t['worst_evolved_percent'])
        return dict({k: t.get(k) for k in ('name', 'R_prime', 'M', 'm', 'rule', 'gtol', 'impl', 'graphs',
                                           'worst_evolved_percent', 'median_evolved_percent', 'median_gpu_ms',
                                           'median_host_ms', 'A1_median_ms', 'A2_median_ms', 'invocations',
                                           'total_iterations_per_case', 'max_iterations_per_step', 'stalled_exits',
                                           'budget_exits', 'per_case_evolved_percent')},
                    table1_fom=f1, table1_speedup_gpu=(foms[f1]['median_gpu_ms'] / t['median_gpu_ms']) if f1 else None,
                    own_fom=own, own_fom_ms=foms[own]['median_gpu_ms'] if own else None,
                    own_speedup_gpu=(foms[own]['median_gpu_ms'] / t['median_gpu_ms']) if own else None)
    sec = {}
    for n in ta.get('secondary', []):
        sec[n] = rowt(n)
    # secondary eng arms: time = faster of the two modes (dev convention), per width
    sec_best = {}
    for n, v in sec.items():
        k = n.split('__')[0]
        if k not in sec_best or v['median_gpu_ms'] < sec_best[k]['median_gpu_ms']:
            sec_best[k] = v
    sel = dict(mesh=L, rule='test panel: frozen Table-1 arms (DESIGN.md section 2), reported as measured',
               primary_impl=ta['primary_impl'], roles={r: rowt(n) for r, n in prim.items()},
               table1=dict(fom=f1, fom_gpu_ms=foms[f1]['median_gpu_ms'] if f1 else None,
                           fom_worst_evolved_percent=foms[f1]['worst_evolved_percent'] if f1 else None,
                           fom_median_evolved_percent=foms[f1]['median_evolved_percent'] if f1 else None,
                           rule='fastest timed converged (setting, mode) with worst error <= the accurate arm, same job'),
               secondary=sec, secondary_faster_mode=sec_best,
               fom_table={n: {k: foms[n][k] for k in ('worst_evolved_percent', 'median_evolved_percent', 'median_gpu_ms',
                                                      'nonlinear_converged', 'invocations')} for n in foms})

    # ---- eng vs parent (2048^2 secondary arms) ----
    if sec:
        rows_ep = []
        for n in sec:
            twin = n.split('__')[0] + '__parent'
            if twin not in quick:
                continue
            per = []
            for c in sorted(quick[n]):
                a_, b_ = quick[n][c], quick[twin][c]
                per.append((abs(a_['same_grid_evolved'] / b_['same_grid_evolved'] - 1),
                            a_['total_iterations'] == b_['total_iterations']))
            rows_ep.append(dict(eng=n, parent=twin, worst_relative_error_diff=max(x[0] for x in per),
                                iterations_identical=all(x[1] for x in per)))
        gate('eng_vs_parent_2048', bool(rows_ep) and all(x['worst_relative_error_diff'] <= 1e-8 and x['iterations_identical']
                                                        for x in rows_ep), rows=rows_ep, bar=1e-8)

    # ---- reproduction of the earlier held-out job on the same 64 cases ----
    if cfg.get('prior_heldout'):
        assert a.prior, 'this mesh has an earlier held-out job: pass --prior'
        pr = json.loads(Path(a.prior).read_text())
        ptab = pr['table']
        base_of = lambda n: n.split('__')[0]
        rep_rows = []
        for n, t in table.items():
            hits = [m for m in ptab if (base_of(m) == base_of(n) if t['family'] == 'rom' else m == n)]
            for m in hits:
                x, y = t['worst_evolved_percent'], ptab[m]['worst_evolved_percent']
                ok = (abs(x - y) <= 1e-9) if max(abs(x), abs(y)) < 1e-9 else abs(x / y - 1) <= 1e-6
                rep_rows.append(dict(this=n, prior=m, this_percent=x, prior_percent=y, passed=bool(ok)))
        need = [n for n in prim.values() if any(base_of(m) == base_of(n) for m in ptab)]
        gate('reproduces_prior_heldout', bool(need) and all(x['passed'] for x in rep_rows),
             prior_job=pr.get('job_id'), prior_sha256=sha(a.prior), rows=rep_rows,
             primary_arms_covered=need, bar='1e-6 relative on the worst evolved error')
    gate('cohort_is_test64', r['physical_sha256'] == 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'
         or bool(cfg.get('local_smoke_waives_cohort_hash')), got=r['physical_sha256'])

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
                   gates=gates, failed_gates=failed, accepted=not failed, selection=sel, table=table,
                   full_grid=full, coefficient_map=cmap,
                   sources=dict(result_json_sha256=sha(o / 'result.json'), config=cfg['attempt']))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1, allow_nan=False, default=float) + '\n')
    print('failed gates:', failed)
    print('table1 FOM', sel['table1'])
    for k, v in sel['roles'].items():
        print(k, v['name'], round(v['worst_evolved_percent'], 4), round(v['median_evolved_percent'], 4),
              round(v['median_gpu_ms'], 2), v['table1_speedup_gpu'] and round(v['table1_speedup_gpu'], 3))
    for k, v in sel['secondary_faster_mode'].items():
        print('secondary', v['name'], round(v['worst_evolved_percent'], 4), round(v['median_gpu_ms'], 2),
              v['table1_speedup_gpu'] and round(v['table1_speedup_gpu'], 3))


if __name__ == '__main__':
    main()
