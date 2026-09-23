"""Independent NumPy audit of one burgers-repanel attempt (no JAX import anywhere in this file).

Adapted from experiments/burgers-eqcert/audit_eqcert.py @ 176b2a9a. From the SAVED outputs only it
  * recomputes the same-grid evolved error of every (arm, case) on the restricted grid and, for the audit case,
    the full grid, against the job's numbers;
  * checks the parity gate: every optimised arm that claims an audited pre-optimisation twin reproduces that
    twin's field to the configured bar with identical per-step iteration counts and stop reasons;
  * builds the per-arm table and the single-job speedup of every ROM arm against the full-order setting the
    paper's rule selects, on the full candidate grid AND on the b-panel-only subset;
  * reports the measured gain of the optimised solver path per mesh (pre-optimisation ms / optimised ms, both
    from this one job).
  * when the job DID certify rules (it does not here: `skip_certificates`), it also recomputes every arm's
    certification status and rho, as the burgers-eqcert audit did. Those gates are marked not-applicable when
    the job carries no populations.
Writes `<out>` (the small summary.json the report reads).

    python audit_repanel.py runs/<attempt>/archive --checkpoint <pkl> --out checks/<attempt>-summary.json
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


# ------------------------------------------------ NumPy re-implementation of rho ----

def silu(x):
    return x / (1. + np.exp(-x))


def bank_np(params, xy, chunk=65536):
    """sep_common.features in NumPy: out_scale * bc(x) * g_mlp([sin, cos](2 pi x B))."""
    out = []
    for s in range(0, len(xy), chunk):
        p = xy[s:s + chunk]
        ang = 2. * np.pi * (p @ params['B'])
        x = np.concatenate((np.sin(ang), np.cos(ang)), -1)
        for w, b in params['g'][:-1]:
            x = silu(x @ np.asarray(w) + np.asarray(b))
        w, b = params['g'][-1]
        x = x @ np.asarray(w) + np.asarray(b)
        bc = 16. * p[:, 0] * (1 - p[:, 0]) * p[:, 1] * (1 - p[:, 1])
        out.append(float(np.asarray(params['out_scale'])) * bc[:, None] * x)
    return np.concatenate(out)


def modes_np(L, M):
    k = np.arange(1, L)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    lam = 4 * L ** 2 * (np.sin(np.pi * kx / (2 * L)) ** 2 + np.sin(np.pi * ky / (2 * L)) ** 2)
    ind = np.argsort(lam.ravel(), kind='stable')[:M]
    return kx.ravel()[ind], ky.ravel()[ind]


def advection_np(U, L):
    """engines.spatial's upwind advection on the full (L-1)^2 interior, U (L-1, L-1)."""
    p = np.pad(U, 1)
    c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    return c * L * (np.where(c > 0, c - xm, xp - c) + np.where(c > 0, c - ym, yp - c))


def rho_np(G, h, L, kx, ky, ij, w):
    U = (G @ h).reshape(L - 1, L - 1)
    a = advection_np(U, L)
    x = np.arange(1, L) / L
    sx, sy = np.sin(np.pi * x[:, None] * kx), np.sin(np.pi * x[:, None] * ky)
    exact = (2. / L) * np.sum(sx * (a @ sy), axis=0)
    rows = (2. / L) * np.sin(np.pi * (ij[:, 0] / L)[:, None] * kx) * np.sin(np.pi * (ij[:, 1] / L)[:, None] * ky)
    samp = (w * a[ij[:, 0] - 1, ij[:, 1] - 1]) @ rows
    return float(np.linalg.norm(samp - exact) / np.linalg.norm(exact))


def status_from(r_held, meta_held, r_dep, meta_dep, j, ndraw):
    per = []
    for d in range(ndraw):
        sh = (meta_held[:, 0] == d) & (meta_held[:, 2] >= j)
        sd = (meta_dep[:, 0] == d) & (meta_dep[:, 2] >= j)
        per.append(bool(r_held[sh].max() <= BAR and r_dep[sd].max() <= BAR))
    n = sum(per)
    return ('confirmed' if n == ndraw else 'fails' if n == 0 else f'marginal ({n}/{ndraw})'), per


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--spot-random', type=int, default=6)
    a = p.parse_args()
    arc = Path(a.archive)
    o = arc / 'output'
    r = json.loads((o / 'result.json').read_text())
    cfg = r['config']
    L = r['intervals']
    tight = cfg['same_grid_reference']
    gates = {}

    def gate(name, ok, **kw):
        gates[name] = dict(passed=bool(ok), **kw)

    log = ''.join(x.read_text(errors='replace') for x in sorted((arc / 'logs').glob('*.out')))
    gate('log_says_backend_gpu', 'jax_backend=gpu' in log)
    gate('complete', r.get('complete', False))
    gate('x64_and_highest', r['x64'] and r['matmul_precision'] == 'highest')
    gate('cohort_matches_b_panel', r['gates']['evaluation_cohort_matches_b_panel']['passed'] in (True, None))
    gate('phi_free_operator_parity', r['gates']['phi_free_operator_parity']['passed'])
    for k in ('repetition_output_identical', 'five_retained_repetitions_everywhere'):
        if k in r['gates']:
            gate(k, r['gates'][k]['passed'])

    # ---- restricted / full-grid error recomputation ------------------------------------------
    cases = sorted({x['case'] for x in r['quick']})
    base = {c: np.load(o / f'restricted_{tight}_case{c}.npz')['fields'] for c in cases}
    worst_gap, rows = 0., []
    for x in r['quick']:
        f = np.load(o / f"restricted_{x['name']}_case{x['case']}.npz")['fields']
        b = base[x['case']]
        sg = np.linalg.norm((f - b).reshape(len(f), -1), axis=1) / np.linalg.norm(b[0])
        rows.append(dict(name=x['name'], case=x['case'], restricted_evolved=float(sg[1:].max()),
                         job_full_evolved=x['same_grid_evolved']))
        if x['same_grid_evolved'] > 1e-6:
            worst_gap = max(worst_gap, abs(sg[1:].max() / x['same_grid_evolved'] - 1))
    exact_restriction = max(1, L // cfg.get('restrict_to', 256)) == 1
    gate('restricted_recomputation_tracks_full_grid', worst_gap < (1e-9 if exact_restriction else 0.05),
         worst_relative_gap=worst_gap, restriction_is_identity=exact_restriction)
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
                             abs_diff=abs(max(sg[1:]) - job['same_grid_evolved']), sha256=sha(fp)))
    gate('full_grid_errors_recomputed', bool(full) and all(x['abs_diff'] <= 1e-12 for x in full),
         worst_abs_diff=max([x['abs_diff'] for x in full] or [None]), arms=len(full))

    # ---- parity: every optimised arm against its audited pre-optimisation twin -----------------
    cov = [x for x in r['parity'] if x.get('covered')]
    gate('parity_against_preoptimisation_arm', bool(cov) and all(x['passed'] for x in cov),
         pairs=[dict(fast=x['fast'], base=x['base'], worst_relative=x['worst_relative'],
                     integers_identical=x['integers_identical'], passed=x['passed']) for x in cov],
         bar=r['config']['parity_bar'],
         note='a covered pair is an optimised arm that changes only the kernel or the factorisation, so it must '
              'reproduce the audited arm bit-for-bit up to the bar AND take identical iterations; clip / '
              'damping carry-over / quadratic predictor / exact-first-step arms change the iterates by design '
              'and are NOT parity arms -- they are gated on their directly measured error instead')

    # ---- certification status, recomputed from the saved per-state rho ------------------------
    CERTIFIED_HERE = bool(r['populations'])        # false when the job ran with skip_certificates
    draws = r['population_source']['draws']
    nd = r['population_source'].get('certification_draws', len(draws))
    ci = r['population_source'].get('confirmation_draw_index')
    pops = {}
    for q in r['populations']:
        z = np.load(o / f'population_q{q}.npz')
        pops[int(q)] = dict(coefficients=z['coefficients'], meta=z['meta'])
    rules = {(x['q'], x['M'], x['rule']): x for x in r['rules']}
    status, mism = {}, []
    for name, st in r['arm_status'].items():
        if st['status'] == 'exact residual':
            status[name] = dict(st, audited_status='exact residual')
            continue
        ru = rules[(st['q'], st['M'], st['rule'])]
        rh = np.asarray(ru['heldout_rho_per_state'])
        mh = pops[st['q']]['meta']
        dz = np.load(o / f'deployed_{name}.npz')
        s_, per = status_from(rh, mh, dz['rho'], dz['meta'], st['exact_steps'], nd)
        conf_pass = None
        if ci is not None:
            j = st['exact_steps']
            sh, sd = (mh[:, 0] == ci) & (mh[:, 2] >= j), (dz['meta'][:, 0] == ci) & (dz['meta'][:, 2] >= j)
            conf_pass = bool(rh[sh].max() <= BAR and dz['rho'][sd].max() <= BAR)
            if conf_pass != st.get('confirmation_pass'):
                mism.append(name + ' (confirmation)')
        # diagnostics beside the status (never change it)
        diag = {}
        cm, cd = mh[:, 0] < nd, dz['meta'][:, 0] < nd                   # certification draws only
        for j in (0, 1, 2):
            diag[f'heldout_rho_max_k>={j}'] = float(rh[cm & (mh[:, 2] >= j)].max())
            diag[f'deployed_rho_max_k>={j}'] = float(dz['rho'][cd & (dz['meta'][:, 2] >= j)].max())
            if ci is not None:
                diag[f'confirmation_heldout_rho_max_k>={j}'] = float(rh[(mh[:, 0] == ci) & (mh[:, 2] >= j)].max())
                diag[f'confirmation_deployed_rho_max_k>={j}'] = float(dz['rho'][(dz['meta'][:, 0] == ci) & (dz['meta'][:, 2] >= j)].max())
            diag[f'draws_passed_if_k>={j}'] = int(sum(status_from(rh, mh, dz['rho'], dz['meta'], j, nd)[1]))
        status[name] = dict(st, audited_status=s_, audited_per_draw=per, audited_confirmation_pass=conf_pass, **diag)
        if s_ != st['status']:
            mism.append(name)
    gate('arm_status_recomputed_matches_job', not mism, mismatches=mism, applicable=CERTIFIED_HERE)
    ctrl = [n for n, s_ in status.items() if s_.get('control')]
    gate('control_fails_certificate',
         (not CERTIFIED_HERE) or (bool(ctrl) and all(status[n]['audited_status'] != 'confirmed' for n in ctrl)),
         controls={n: status[n]['audited_status'] for n in ctrl}, applicable=CERTIFIED_HERE,
         note=('not applicable: this lane re-times FROZEN rules certified by burgers-eqcert and runs no '
               'certificate of its own; the rules\' status is carried in config.rule_status'
               if not CERTIFIED_HERE else ''))

    # ---- switching parity: the first j steps of an x_j arm ARE the exact arm's first j steps -------
    sw = []
    setup = {x['arm']: x for x in r['arm_setup']}
    for name, st in setup.items():
        j = st.get('exact_steps') or 0
        if not j:
            continue
        gtag = name.split('_fast')[0].split('_')[-1]                  # e.g. g0p001
        twin = f"q{st['q']}_M{st['M']}_exact_{gtag}_fast_chol_clip_lamcarry_pred2"
        if st['q'] != 256 or not (o / f'restricted_{twin}_case0.npz').exists():
            continue
        for c in cases:
            wa = np.load(o / f'restricted_{name}_case{c}.npz')['internal_latents'][:j + 1]
            wb = np.load(o / f'restricted_{twin}_case{c}.npz')['internal_latents'][:j + 1]
            sw.append(dict(arm=name, twin=twin, case=c, rel=float(np.linalg.norm(wa - wb) / np.linalg.norm(wb))))
    has_twin = any((x.get('exact_steps') or 0) for x in setup.values())
    gate('exact_phase_matches_exact_arm', (not has_twin) or (bool(sw) and max(x_['rel'] for x_ in sw) <= 1e-10),
         applicable=has_twin,
         worst=max([x_['rel'] for x_ in sw] or [None]), pairs=len(sw),
         note='internal latents w_0..w_j of every x_j arm vs the exact-residual arm at the same tolerance')

    # ---- rho itself, NumPy re-implementation, spot-check -----------------------------------------
    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = ck['params']
    x = np.arange(1, L) / L
    G = bank_np(params, np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2))
    rng = np.random.default_rng(20260921)
    spot = []
    q256 = [n for n, s_ in status.items() if CERTIFIED_HERE and s_['q'] == 256 and s_['status'] != 'exact residual']
    for name in q256:
        st = status[name]
        ru = rules[(st['q'], st['M'], st['rule'])]
        rz = np.load(o / f"rule_L{L}_q{st['q']}_M{st['M']}_{st['rule']}.npz")
        ij, w = rz['ij'], rz['weights']
        kx, ky = modes_np(L, st['M'])
        dz = np.load(o / f'deployed_{name}.npz')
        pick = {int(np.argmax(np.where(dz['meta'][:, 0] == d, dz['rho'], -1))) for d in range(nd)}
        pick |= set(rng.choice(len(dz['rho']), a.spot_random, replace=False).tolist())
        for i in sorted(pick):
            got = rho_np(G, dz['coefficients'][i], L, kx, ky, ij, w)
            spot.append(dict(arm=name, population='deployed', state=i, meta=dz['meta'][i].tolist(), job=float(dz['rho'][i]),
                             numpy=got, rel_diff=abs(got - float(dz['rho'][i])) / max(float(dz['rho'][i]), 1e-300)))
        rh = np.asarray(ru['heldout_rho_per_state'])
        mh = pops[st['q']]['meta']
        for d in range(nd):
            i = int(np.argmax(np.where(mh[:, 0] == d, rh, -1)))
            got = rho_np(G, pops[st['q']]['coefficients'][i], L, kx, ky, ij, w)
            spot.append(dict(arm=name, population='heldout', state=i, meta=mh[i].tolist(), job=float(rh[i]), numpy=got,
                             rel_diff=abs(got - rh[i]) / max(rh[i], 1e-300)))
    wr = max([x_['rel_diff'] for x_ in spot] or [None])
    gate('rho_recomputed_in_numpy', (not CERTIFIED_HERE) or (bool(spot) and wr <= 1e-7),
         applicable=CERTIFIED_HERE, worst_relative_diff=wr, states=len(spot),
         note='bank, advection, sine tests and rule re-implemented in NumPy from the checkpoint; spot-check set = every '
              'per-draw argmax of every q=256 rule arm (both populations) + random deployed states')
    del G

    # ---- per-arm table -----------------------------------------------------------------------------
    inv = r['invocations']
    table = {}
    for n in dict.fromkeys(x_['name'] for x_ in inv):
        xs = [x_ for x_ in inv if x_['name'] == n]
        pc = {}
        for x_ in xs:
            pc.setdefault(x_['case'], x_)
        t = dict(name=n, family=xs[0]['family'], reps=len(xs) // max(len(pc), 1), cases=len(pc),
                 median_gpu_ms=1e3 * med([x_['gpu_seconds'] for x_ in xs]),
                 median_host_ms=1e3 * med([x_['host_seconds'] for x_ in xs]),
                 worst_evolved_percent=100 * max(x_['same_grid_evolved'] for x_ in pc.values()),
                 worst_all_times_percent=100 * max(x_['same_grid_all'] for x_ in pc.values()),
                 median_evolved_percent=100 * med([x_['same_grid_evolved'] for x_ in pc.values()]),
                 worst_current_relative_evolved_percent=100 * max(x_['current_relative_evolved'] for x_ in pc.values()))
        if t['family'] == 'rom':
            t.update(q=xs[0]['q'], M=xs[0]['M'], m=xs[0]['m'], rule=xs[0]['rule'], gtol=xs[0]['gtol'],
                     total_iterations_median=med([x_['total_iterations'] for x_ in pc.values()]),
                     stalled_exits=int(sum(x_['stalled_exits'] for x_ in pc.values())),
                     steps=int(sum(len(x_['stop_reasons']) for x_ in pc.values())),
                     damping_retries=int(sum(x_.get('damping_retries_total', 0) for x_ in pc.values())))
            s_ = status.get(n, {})
            t.update(status=s_.get('audited_status'), exact_steps=s_.get('exact_steps'), control=bool(s_.get('control')),
                     heldout_rho_max=s_.get('heldout_rho_max'), deployed_rho_max=s_.get('deployed_rho_max'),
                     confirmation_pass=s_.get('audited_confirmation_pass'),
                     confirmation_rho_max=(max(s_[f"confirmation_heldout_rho_max_k>={s_['exact_steps']}"],
                                               s_[f"confirmation_deployed_rho_max_k>={s_['exact_steps']}"])
                                           if s_.get('exact_steps') is not None and f"confirmation_heldout_rho_max_k>={s_['exact_steps']}" in s_ else None))
        else:
            t.update(mesh=xs[0]['mesh'], dt=xs[0]['dt'], ntol=xs[0]['ntol'], ltol=xs[0]['ltol'], impl=xs[0].get('impl'),
                     stalled_steps=int(sum(x_['stalled_steps'] for x_ in pc.values())),
                     nonlinear_converged=all(x_['nonlinear_converged'] for x_ in pc.values()))
        table[n] = t
    phys = {}
    for x_ in r.get('physical', []):
        phys.setdefault(x_['name'], []).append(x_['reference_evolved'])
    for n, v in phys.items():
        if n in table:
            table[n]['worst_reference_evolved_percent'] = 100 * max(v)
    foms = {n: t for n, t in table.items() if t['family'] == 'fom' and t.get('mesh') == L}
    subsets = cfg.get('fom_subsets') or dict(full=list(foms))

    def fom_for(t, names=None):
        """The paper's rule: the FASTEST tested setting whose worst evolved error is at least as small as
        the ROM arm's. `names` restricts the candidate grid (b-panel subset vs the full union)."""
        pool = [f for f in foms if names is None or f in names]
        ok = [f for f in pool if foms[f]['nonlinear_converged']
              and foms[f]['worst_evolved_percent'] <= t['worst_evolved_percent']]
        return min(ok, key=lambda f: foms[f]['median_gpu_ms']) if ok else None

    for n, t in table.items():
        if t['family'] != 'rom':
            continue
        f = fom_for(t)
        t.update(fom_by_paper_rule=f, speedup_gpu=(foms[f]['median_gpu_ms'] / t['median_gpu_ms']) if f else None,
                 speedup_host=(foms[f]['median_host_ms'] / t['median_host_ms']) if f else None)
        t['by_grid'] = {}
        for gname, names in subsets.items():
            g = fom_for(t, names)
            t['by_grid'][gname] = dict(fom=g, fom_gpu_ms=foms[g]['median_gpu_ms'] if g else None,
                                       fom_host_ms=foms[g]['median_host_ms'] if g else None,
                                       fom_worst_evolved_percent=foms[g]['worst_evolved_percent'] if g else None,
                                       speedup_gpu=(foms[g]['median_gpu_ms'] / t['median_gpu_ms']) if g else None,
                                       speedup_host=(foms[g]['median_host_ms'] / t['median_host_ms']) if g else None)

    def row(t, basis):
        f = t['fom_by_paper_rule']
        return dict(arm=t['name'], basis=basis, m=t.get('m'), q=t.get('q'), M=t.get('M'), gtol=t.get('gtol'),
                    rule=t.get('rule'), exact_steps=t.get('exact_steps'),
                    worst_evolved_percent=t['worst_evolved_percent'],
                    median_evolved_percent=t['median_evolved_percent'],
                    worst_all_times_percent=t['worst_all_times_percent'],
                    worst_current_relative_evolved_percent=t['worst_current_relative_evolved_percent'],
                    rom_gpu_ms=t['median_gpu_ms'], rom_host_ms=t['median_host_ms'],
                    stalled_exits=t['stalled_exits'], steps=t['steps'],
                    total_iterations_median=t['total_iterations_median'],
                    fom=f, fom_gpu_ms=foms[f]['median_gpu_ms'] if f else None,
                    fom_host_ms=foms[f]['median_host_ms'] if f else None,
                    fom_worst_evolved_percent=foms[f]['worst_evolved_percent'] if f else None,
                    speedup_gpu=t['speedup_gpu'], speedup_host=t['speedup_host'], by_grid=t.get('by_grid'))

    # ---- the repanel verdict: named roles, one job, one row ratio -------------------------------
    roles = cfg.get('roles', {})
    verdict = dict(mesh=L, error_convention='same-grid, evolved (max over the five evolved output times of '
                                            '||u_arm - u_ref|| / ||u_0||, reference = %s at this mesh)' % tight,
                   cohort=r.get('cohort_name'), rows={}, missing_roles=[])
    for role, name in roles.items():
        t = table.get(name)
        if t is None or t['family'] != 'rom':
            verdict['missing_roles'].append(dict(role=role, arm=name))
            continue
        verdict['rows'][role] = row(t, role)
    # the measured gain of the optimised solver path, both arms from THIS job
    def gain(pre, post, label):
        A_, B_ = table.get(pre), table.get(post)
        if not A_ or not B_:
            return None
        return dict(label=label, preoptimisation_arm=pre, optimised_arm=post,
                    preoptimisation_gpu_ms=A_['median_gpu_ms'], optimised_gpu_ms=B_['median_gpu_ms'],
                    gpu_ratio=A_['median_gpu_ms'] / B_['median_gpu_ms'],
                    preoptimisation_host_ms=A_['median_host_ms'], optimised_host_ms=B_['median_host_ms'],
                    host_ratio=A_['median_host_ms'] / B_['median_host_ms'],
                    preoptimisation_worst_evolved_percent=A_['worst_evolved_percent'],
                    optimised_worst_evolved_percent=B_['worst_evolved_percent'])
    verdict['optimisation_gain'] = [g for g in (
        gain(roles.get('preoptimisation_accurate'), roles.get('optimised_accurate'),
             'accurate row: the arm the paper prints -> the arm it should print (different quadrature rule)'),
        gain(roles.get('preoptimisation_accurate'), roles.get('optimised_on_the_papers_own_rule'),
             'accurate row, SAME quadrature rule and tolerance: solver path only'),
        gain(roles.get('preoptimisation_fast'), roles.get('optimised_fast'),
             'fast row: the arm the paper prints -> the arm it should print')) if g]
    verdict['rule_status'] = cfg.get('rule_status')
    verdict['certificates_run_in_this_job'] = CERTIFIED_HERE
    verdict['parity'] = gates['parity_against_preoptimisation_arm']
    failed = [k for k, v in gates.items() if not v['passed']]
    verdict['accepted'] = not failed
    verdict['acceptance_rule'] = 'every audit gate passed (else the verdict is reported but not accepted)'
    summary = dict(attempt=cfg['attempt'], job_id=r['job_id'], commit=r['commit'], gpu=r['gpu'], nvidia_smi=r['nvidia_smi'],
                   intervals=L, cohort=r.get('cohort_name'), cohort_cases=len(r['physical_cases']),
                   elapsed_seconds=r.get('elapsed_seconds'), population_source=r['population_source'],
                   populations=r['populations'], verdict=verdict, arm_status=status, rho_spot_check=spot, gates=gates,
                   failed_gates=failed, table=table, switching_parity=sw,
                   dense_truth=[{k: v for k, v in x_.items() if k in ('name', 'case', 'same_grid_evolved', 'deployed_vs_dense',
                                                                      'total_iterations', 'stalled_exits')} for x_ in r['dense_truth']],
                   rules=[{k: v for k, v in x_.items() if k != 'heldout_rho_per_state'} for x_ in r['rules']],
                   parity=r['parity'], profile=r['profile'], dropped=r['dropped'], full_grid_recheck=full,
                   sources=dict(result_json_sha256=sha(o / 'result.json'), audit_script_sha256=sha(__file__),
                                checkpoint_sha256=sha(a.checkpoint), job_commit=r['commit'],
                                provenance_sha256=sha(arc / 'PROVENANCE.json')))
    Path(a.out).write_text(json.dumps(summary, indent=1) + '\n')
    print('failed gates:', summary['failed_gates'])
    for n, t in sorted(table.items(), key=lambda kv: (kv[1]['family'], kv[1]['median_gpu_ms'])):
        print(f"{n:58s} ev {t['worst_evolved_percent']:8.4f}%  gpu {t['median_gpu_ms']:9.2f} ms"
              + (f"  FOM {t['fom_by_paper_rule']}  S {t['speedup_gpu']}" if t['family'] == 'rom'
                 else f"  conv {t['nonlinear_converged']}"))
    print(json.dumps(verdict, indent=1))


if __name__ == '__main__':
    main()
