"""Fixed-checkpoint solver-effort and quadrature tuning for the inherited head.

One frozen checkpoint, latent dimension, bank rank, weak-mode count, mesh, time
step and cold initializer.  Only three controls vary: the Gauss-Newton evolution
iteration cap, the evolution stopping tolerance, and which offline-fitted
empirical-quadrature (EQ) rule supplies the sampled advection term.  The
full-grid FOM-exact upwind advection projected onto the same weak modes is the
quadrature-free control.

Nothing here retrains, changes capacity or relaxes an acceptance gate.  Arms that
stop on their iteration cap are recorded as early-stopped and never relabelled
stationary.  Every deployable arm returns the supplied initial field exactly; the
decoder's own compression of that field is retained as a separate diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import data as d
from refine import configure

CHECKPOINT_RELATIVE = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
CONFIG_PATH = d.HERE / 'tuning-config.json'


# ---------------------------------------------------------------- weak residual


def make_weak_full(e, sc):
    """FOM-exact upwind advection on every interior node, projected on the weak modes.

    Identical algebra to the sampled ``engines.weak`` apart from the advection
    term: ``Phi.T @ spatial(G h(z))`` replaces ``Pq.T @ adv(sampled)``.  The exact
    linear terms, the weak-mode set and the Helmholtz-style scaling are unchanged.
    """
    def weak_full(z, prev, nu, data, params, L, dt):
        G, A, lam = data[0], data[1], data[2]
        phi = data[8]
        h = sc.head(params, z)
        adv = e.spatial(G @ h, L)[0]
        ah = A @ h
        return (ah - prev + dt * (phi.T @ adv + nu * lam * ah)) / (1 + dt * nu * lam)
    return weak_full


# ------------------------------------------------------------------- EQ fitting


def bounded_nnls_capped(design, b, max_support, deadline, scipy_optimize):
    """``engines.nnls_capped`` with a walltime deadline and preserved partial support."""
    support = []
    w = np.empty(0)
    residual = b.copy()
    truncated = False
    rounds = 0
    for _ in range(5 * max_support):
        if time.monotonic() > deadline:
            truncated = True
            break
        rounds += 1
        grad = design.T @ residual
        if support:
            grad[support] = -np.inf
        idx = int(np.argmax(grad))
        if grad[idx] <= 1e-10 * max(np.linalg.norm(b), 1e-300) or len(support) >= max_support:
            break
        support.append(idx)
        w, _ = scipy_optimize.nnls(design[:, support], b, maxiter=10 * len(support))
        active = w > 1e-14
        support = list(np.asarray(support)[active])
        w = w[active]
        residual = b - design[:, support] @ w
    return np.asarray(support, dtype=int), w, truncated, rounds


def fit_eq_rule(jax, jnp, e, params, Z, L, M, m, G, phi, deadline, seed, candidate_cap, fit_states):
    """Decoder-output NNLS quadrature fit, replicating ``engines.build_rom`` exactly.

    The candidate draw, the fitting-state draw, the row scaling and the greedy
    positive active-set criterion are the incumbent ones; only the walltime bound
    and the recorded support/weight/residual evidence are new.  Weights are fit on
    decoder-output snapshots, never on residual snapshots.
    """
    import scipy.optimize as scipy_optimize
    begin = time.monotonic()
    rng = np.random.default_rng(seed)
    cp = np.sort(rng.choice((L - 1) ** 2, min(candidate_cap, (L - 1) ** 2), replace=False))
    fit = np.sort(rng.choice(len(Z), min(fit_states, len(Z)), replace=False))
    h = jax.jit(lambda z: e.sc.head(params, z))
    adv = jax.jit(lambda G, z: e.spatial(G @ h(z), L)[0])
    Phi = np.asarray(phi)
    rows, targets = [], []
    for i in fit:
        n = adv(G, jnp.asarray(Z[i]))
        targets.append(np.asarray(jnp.asarray(Phi).T @ n))
        rows.append(Phi[cp].T * np.asarray(n)[cp])
    design = np.concatenate(rows)
    b = np.concatenate(targets)
    scale = np.linalg.norm(design, axis=1) + 1e-300
    design = design / scale[:, None]
    b = b / scale
    supp, w, truncated, rounds = bounded_nnls_capped(design, b, m, deadline, scipy_optimize)
    padded = False
    if len(supp) < m and time.monotonic() < deadline:
        rest = np.setdiff1d(np.arange(len(cp)), supp)
        supp = np.concatenate((supp, rest[np.argsort(-np.abs(design[:, rest]).mean(0))[:m - len(supp)]]))
        padded = True
    if len(supp) and time.monotonic() < deadline:
        w, _ = scipy_optimize.nnls(design[:, supp], b, maxiter=10 * len(supp))
        refit = True
    else:
        refit = False
    pos = cp[supp]
    relative_fit = float(np.linalg.norm(design[:, supp] @ w - b) / np.linalg.norm(b))
    info = dict(requested_m=int(m), actual_m=int(len(supp)), greedy_rounds=int(rounds),
                deadline_truncated=bool(truncated), padded_to_requested=bool(padded),
                final_nonnegative_refit=bool(refit), eq_relative_fit=relative_fit,
                eq_seed=int(seed), eq_candidate_cap=int(candidate_cap),
                eq_actual_candidates=int(len(cp)), eq_fit_rows=fit.tolist(),
                eq_indices=pos.tolist(), eq_weights=np.asarray(w, dtype=float).tolist(),
                fit_seconds=time.monotonic() - begin,
                fitting_data='decoder-output advection snapshots on the recorded training codes')
    return pos, np.asarray(w, dtype=float), info


def quadrature_arrays(e, jnp, dec, phi, pos, weights, L, R):
    """Five-point upwind stencil features and weighted mode rows for one EQ rule."""
    ij = np.stack(np.unravel_index(pos, (L - 1, L - 1)), 1) + 1
    offsets = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    G5 = dec.feat_at(((ij[:, None, :] + offsets[None, :, :]) / L).reshape(-1, 2)).reshape(len(pos), 5, R)
    Pq = jnp.asarray(np.asarray(phi)[pos] * weights[:, None])
    return G5, Pq


# ------------------------------------------------------------------ measurement


def field_hash(fields):
    return hashlib.sha256(np.ascontiguousarray(fields, dtype=np.float64).tobytes()).hexdigest()


def error_metrics(fields, reference):
    metrics = d.fixed_initial_errors(np.asarray(fields), np.asarray(reference))
    current = np.linalg.norm((np.asarray(fields) - reference)[:, 1:-1, 1:-1].reshape(len(d.TIMES), -1), axis=1)
    denominator = np.maximum(np.linalg.norm(reference[:, 1:-1, 1:-1].reshape(len(d.TIMES), -1), axis=1), 1e-300)
    metrics['current_relative_per_time'] = (current / denominator).tolist()
    metrics['current_relative_maximum'] = float(np.max(current / denominator))
    return metrics


def stopping_record(host):
    """Actual stopping evidence.  reason 4 = normalized gradient, 1 = residual,
    2 = tiny step / stall, 3 = rejected or nonfinite, 0 = iteration cap reached."""
    reasons = np.asarray(host[3]).astype(int)
    gradients = np.asarray(host[8])
    iterations = np.asarray(host[1]).astype(int)
    return dict(step_iterations_total=int(iterations.sum()), step_iterations_max=int(iterations.max()),
                step_reason_counts={str(k): int((reasons == k).sum()) for k in range(5)},
                steps_at_iteration_cap=int((reasons == 0).sum()),
                worst_step_normalized_gradient=float(gradients.max()),
                initial_iterations=int(host[5]), initial_reason=int(host[6]),
                initial_normalized_gradient=float(host[9]),
                gradient_stationary=bool(gradients.max() <= 1e-6 and float(host[9]) <= 1e-6),
                early_stopped=bool((reasons == 0).any()))


# ----------------------------------------------------------------------- driver


def build_context(args, cfg):
    jax, e = d.gpu_modules()
    import jax.numpy as jnp
    import accuracy_paths as ap
    from diagnose import burn

    root = d.ROOT
    checkpoint = root / CHECKPOINT_RELATIVE
    assert d.sha(checkpoint) == cfg['checkpoint_sha256'], 'frozen checkpoint hash mismatch'
    params, Z, _ = e.sc.load_pkl(checkpoint)
    K, R = Z.shape[1], params['h_lin'].shape[1]
    L, M, dt = cfg['fixed']['intervals'], cfg['fixed']['weak_modes'], cfg['fixed']['dt']
    assert K == cfg['fixed']['latent_dimension'] and R == cfg['fixed']['bank_rank']
    dec = e.sc.SeparableDecoder(params, K, R)
    G = dec.feat_at(e.coords(L), chunk=8192)
    phi, lam, mode_ids = e.modes(L, M)
    A = jnp.asarray(phi).T @ G
    candidate_z = jnp.asarray(Z[::max(1, len(Z) // 8192)])
    cold, cold_info = e.build_gauss_cold(params, candidate_z)
    jax.block_until_ready(A)
    context = dict(jax=jax, jnp=jnp, e=e, ap=ap, burn=burn, params=params, Z=Z, K=K, R=R, L=L, M=M, dt=dt,
                   dec=dec, G=G, phi=phi, lam=lam, mode_ids=mode_ids, A=A, candidate_z=candidate_z,
                   cold=cold, cold_info=cold_info, checkpoint=checkpoint,
                   weak_full=make_weak_full(e, e.sc))
    return context


def make_operators(ctx, G5, Pq):
    jnp = ctx['jnp']
    return (ctx['G'], ctx['A'], jnp.asarray(ctx['lam']), G5, Pq, None, None, ctx['candidate_z'],
            jnp.asarray(ctx['phi']))


def make_arm_query(ctx, quadrature, setting, trust):
    """One compiled deployable query.  The supplied initial field is returned exactly."""
    jax = ctx['jax']
    inner = ctx['ap'].make_rom(ctx['params'], ctx['L'], ctx['dt'], trust,
                               ic_budget=ctx['ic_budget'], step_budget=setting['step_budget'],
                               gtol=ctx['initial_gtol'],
                               evolution_gtol=setting['evolution_gtol'],
                               evolution_residual_scale=setting['evolution_residual_scale'],
                               weak_fn=ctx['weak_full'] if quadrature == 'full' else None)

    @jax.jit
    def deployable(u0, nu, data, cold):
        out = inner(u0, nu, data, cold)
        return (out[0].at[0].set(u0),) + tuple(out[1:])
    return deployable


def make_fom_query(ctx, preset):
    jax = ctx['jax']
    native = ctx['e'].make_fom(preset['mesh'], preset['dt'], target=ctx['L'])[0]

    @jax.jit
    def query(u0, nu):
        fields, it, rn = native(u0, nu, preset['ntol'], preset['ltol'])
        return fields.at[0].set(u0), it, rn
    return query


def run_timed_pass(ctx, report, path, out, arms, foms, cases, reps, deadline, pass_name, cfg):
    """Interleaved, burned-in, same-GPU timing.  Cost and accuracy come from one invocation."""
    jax, jnp, np_ = ctx['jax'], ctx['jnp'], np
    for case in cases:
        record, reference, supplied, nu = case['record'], case['reference'], case['supplied'], case['nu']
        u0 = jnp.asarray(supplied)
        nu_device = jnp.asarray(nu)
        for name, arm in arms.items():
            jax.block_until_ready(arm['query'](u0, nu_device, arm['operators'], ctx['cold']))
        for name, fom in foms.items():
            jax.block_until_ready(fom['query'](u0, nu_device))
        order = sorted(arms) + sorted(foms)
        rng = np_.random.default_rng(cfg['timing_seed'] + case['index'])
        for rep in range(reps):
            for method in rng.permutation(order):
                method = str(method)
                if time.monotonic() > deadline:
                    report['stop_reason'] = f'walltime deadline inside {pass_name}'
                    d.write_json(path, report)
                    return False
                ctx['burn'](jax, jnp, cfg['burn_seconds'])
                complete_start = time.perf_counter()
                device_u0 = jax.device_put(supplied)
                device_nu = jax.device_put(np_.asarray(nu, dtype=np_.float64))
                jax.block_until_ready((device_u0, device_nu))
                input_seconds = time.perf_counter() - complete_start
                start = time.perf_counter()
                if method in arms:
                    timed = jax.block_until_ready(arms[method]['query'](device_u0, device_nu,
                                                                        arms[method]['operators'], ctx['cold']))
                else:
                    timed = jax.block_until_ready(foms[method]['query'](device_u0, device_nu))
                gpu_seconds = time.perf_counter() - start
                host_start = time.perf_counter()
                host = jax.tree_util.tree_map(np_.asarray, timed)
                output_seconds = time.perf_counter() - host_start
                complete_seconds = time.perf_counter() - complete_start
                fields = host[0].copy()
                assert fields.dtype == np_.float64 and np_.isfinite(fields).all()
                assert np_.array_equal(fields[0], supplied), 'deployable output must return the supplied field'
                row = dict(pass_name=pass_name, case_id=record['case_id'], case_index=case['index'],
                           method=method, rep=rep, gpu_seconds=gpu_seconds,
                           input_transfer_seconds=input_seconds, output_transfer_seconds=output_seconds,
                           host_to_host_seconds=complete_seconds, error=error_metrics(fields, reference),
                           fields_sha256=field_hash(fields))
                if method in arms:
                    row.update(arm=arms[method]['spec'], **stopping_record(host))
                else:
                    row.update(preset=foms[method]['preset'], max_relative_residual=float(host[2].max()),
                               newton_iterations_total=int(np_.asarray(host[1]).sum()),
                               converged=bool(np_.isfinite(host[2]).all() and host[2].max() <= foms[method]['preset']['ntol']))
                if rep == 0:
                    arrays = dict(fields=fields, iterations=host[1], residuals=host[2])
                    if method in arms:
                        arrays.update(step_reasons=host[3], output_latents=host[4], initial_iterations=host[5],
                                      initial_reason=host[6], internal_latents=host[7],
                                      step_gradients=host[8], initial_gradient=host[9])
                    name = f"{record['case_id']}_{method}_rep{rep}.npz"
                    np_.savez_compressed(out / name, **arrays)
                    row.update(path=name, sha256=d.sha(out / name))
                else:
                    row['fields_retention'] = 'rep0 arrays retained; later repetitions keep timings, errors, stopping records and the field hash'
                report['invocations'].append(row)
                d.write_json(path, report)
        print('TUNING CASE COMPLETE', pass_name, record['case_id'], flush=True)
    return True


def summarize(report, pass_name, method):
    rows = [r for r in report['invocations'] if r['pass_name'] == pass_name and r['method'] == method]
    if not rows:
        return None
    gpu = np.asarray([r['gpu_seconds'] for r in rows])
    host = np.asarray([r['host_to_host_seconds'] for r in rows])
    worst = max(r['error']['maximum'] for r in rows)
    per_case = {}
    for r in rows:
        per_case.setdefault(r['case_id'], []).append(r['gpu_seconds'])
    q1, q3 = np.percentile(gpu, [25, 75])
    return dict(method=method, invocations=len(rows), median_gpu_seconds=float(np.median(gpu)),
                median_complete_host_seconds=float(np.median(host)),
                median_of_case_median_gpu_seconds=float(np.median([np.median(v) for v in per_case.values()])),
                worst_fixed_initial_error=float(worst),
                worst_current_relative_error=float(max(r['error']['current_relative_maximum'] for r in rows)),
                upper_tukey_latency_outliers=int((gpu > q3 + 1.5 * (q3 - q1)).sum()),
                gpu_seconds_all_repetitions=gpu.tolist(),
                early_stopped_invocations=int(sum(1 for r in rows if r.get('early_stopped'))),
                gradient_stationary_invocations=int(sum(1 for r in rows if r.get('gradient_stationary'))),
                unconverged_fom_invocations=int(sum(1 for r in rows if r.get('converged') is False)))


def calibration_cases(ctx, reference_index, cfg, indices):
    """Independently calibrated development cases with their recorded fine reference."""
    e, L = ctx['e'], ctx['L']
    selected, evidence = d.read_calibration(reference_index, L)
    report = json.loads(Path(reference_index).read_text())
    anchor_cfg = cfg['reference_anchor']
    cases = []
    for i in indices:
        record = d.case_record('calibration', i)
        anchor = next(r for r in report['solves'] if r['case_id'] == record['case_id']
                      and r['intervals'] == anchor_cfg['intervals'] and r['dt'] == anchor_cfg['dt'])
        artifact = Path(reference_index).parent / anchor['path']
        assert d.sha(artifact) == anchor['sha256']
        truth = d.restrict(np.load(artifact)['fields'], L)
        physical = e.params_draw(record['seed'], 1)[0]
        supplied = e.initial(L, physical)
        assert np.array_equal(supplied, truth[0])
        cases.append(dict(index=i, record=record, reference=truth, supplied=supplied,
                          nu=float(physical[4]), anchor_sha256=anchor['sha256']))
    return cases, dict(selected=selected, evidence=evidence)


def validation_cases(ctx, dataset_index, cfg, count):
    """Held-out common-data cases; the supplied field and viscosity are the only inputs."""
    L = ctx['L']
    index = json.loads(Path(dataset_index).read_text())
    assert index['complete'] and index['split'] == 'validation' and index['mesh'] == L
    assert d.sha(dataset_index) == cfg['validation_index_sha256']
    cases = []
    for i in range(count):
        row = index['records'][i]
        assert row['case_id'] == d.case_record('validation', i)['case_id']
        artifact = Path(dataset_index).parent / row['path']
        assert d.sha(artifact) == row['sha256']
        with np.load(artifact) as arrays:
            truth = np.ascontiguousarray(arrays['target'][:, 0], dtype=np.float64)
            supplied = np.ascontiguousarray(arrays['input'][0], dtype=np.float64)
            nu = float(arrays['parameters'][0])
        assert np.array_equal(supplied, truth[0])
        cases.append(dict(index=i, record=d.case_record('validation', i), reference=truth,
                          supplied=supplied, nu=nu, anchor_sha256=row['sha256']))
    return cases, dict(dataset_index_sha256=d.sha(dataset_index), reference_setting=index['reference_setting'])


def select_rule(report, cfg):
    """Pre-registered: lowest worst-case physical error at the converged setting;
    rules within 1% of that error are separated by lower median GPU seconds."""
    scored = []
    for q in cfg['quadratures']:
        summary = summarize(report, 'quadrature', f'{q}_converged')
        if summary:
            scored.append((summary['worst_fixed_initial_error'], summary['median_gpu_seconds'], q))
    if not scored:
        return None, None
    best = min(s[0] for s in scored)
    near = [s for s in scored if s[0] <= best * 1.01]
    choice = min(near, key=lambda s: s[1])
    return choice[2], dict(criterion=cfg['rule_selection'], scored=[dict(quadrature=s[2],
                           worst_fixed_initial_error=s[0], median_gpu_seconds=s[1]) for s in scored],
                           selected=choice[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--shortlist', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--phase', choices=['calibration', 'validation'], required=True)
    parser.add_argument('--cases', type=int)
    parser.add_argument('--reps', type=int)
    args = parser.parse_args()
    configure()
    cfg = json.loads(CONFIG_PATH.read_text())
    started = time.monotonic()
    deadline = started + args.seconds
    ctx = build_context(args, cfg)
    jax, jnp, e, ap = ctx['jax'], ctx['jnp'], ctx['e'], ctx['ap']
    ctx['ic_budget'] = cfg['fixed']['initial_fit']['ic_budget']
    ctx['initial_gtol'] = cfg['fixed']['initial_fit']['gtol']
    Z = ctx['Z']
    trust = .01 * float(np.max(np.linalg.norm(Z - Z.mean(0), axis=1)))
    assert abs(trust - cfg['fixed']['trust_radius']) <= 1e-12 * cfg['fixed']['trust_radius']
    out = d.make_output(args.out)
    reps = args.reps if args.reps else cfg['repetitions']
    report = dict(schema_version=1, pde='burgers', kind=cfg['kind'], phase=args.phase, complete=False,
                  provenance=d.provenance(jax), config=cfg, config_sha256=d.sha(CONFIG_PATH),
                  own_source_sha256=d.sha(Path(__file__)), accuracy_paths_sha256=d.sha(Path(ap.__file__)),
                  checkpoint_sha256=d.sha(ctx['checkpoint']), trust_radius=trust,
                  cold_setup={k: v for k, v in ctx['cold_info'].items() if k not in ('xy', 'weights')},
                  bank_bytes=int(np.asarray(ctx['G']).nbytes), repetitions=reps,
                  mode_ids=np.asarray(ctx['mode_ids']).tolist(),
                  eq_rules=[], sentinel=[], initial_fit_diagnostic=[], rule_selection=None,
                  invocations=[], summaries={}, native_compression=[])
    path = out / 'index.json'
    d.write_json(path, report)
    assert np.array_equal(np.asarray(ctx['mode_ids']), np.asarray(cfg['fixed']['mode_ids'])), 'weak-mode set changed'

    # --- offline quadrature rules -------------------------------------------------
    rules = {}
    archived = cfg['archived_rule']
    pos = np.asarray(archived['eq_indices'])
    weights = np.asarray(archived['eq_weights'])
    G5, Pq = quadrature_arrays(e, jnp, ctx['dec'], ctx['phi'], pos, weights, ctx['L'], ctx['R'])
    rules['m256'] = dict(operators=make_operators(ctx, G5, Pq),
                         info=dict(requested_m=256, actual_m=len(pos), source='exact archived accepted rule',
                                   eq_relative_fit=archived['eq_relative_fit'], eq_seed=archived['eq_seed']))
    report['eq_rules'].append(rules['m256']['info'])
    d.write_json(path, report)

    if args.phase == 'calibration':
        # Independent reproduction of the accepted rule with the bounded fitter.
        budget = min(cfg['eq_fit_seconds_per_rule'], max(60., deadline - time.monotonic() - 600))
        check_pos, check_w, check_info = fit_eq_rule(jax, jnp, e, ctx['params'], Z, ctx['L'], ctx['M'], 256,
                                                     ctx['G'], ctx['phi'], time.monotonic() + budget,
                                                     archived['eq_seed'], archived['eq_candidate_cap'],
                                                     archived['eq_fit_states'])
        reproduction = dict(indices_identical=bool(np.array_equal(check_pos, pos)),
                            maximum_absolute_weight_difference=float(np.max(np.abs(check_w - weights)))
                            if check_w.shape == weights.shape else None,
                            refit_relative_fit=check_info['eq_relative_fit'],
                            archived_relative_fit=archived['eq_relative_fit'],
                            deadline_truncated=check_info['deadline_truncated'],
                            fit_seconds=check_info['fit_seconds'],
                            interpretation='bounded refit of the accepted m=256 rule; the archived support/weights are used for the m256 arm regardless')
        report['archived_rule_reproduction'] = reproduction
        d.write_json(path, report)
        print('EQ REPRODUCTION', json.dumps(reproduction), flush=True)

        for m in cfg['eq_refit_levels']:
            if time.monotonic() + 600 > deadline:
                report['stop_reason'] = 'walltime deadline before EQ refit'
                d.write_json(path, report)
                break
            budget = min(cfg['eq_fit_seconds_per_rule'], deadline - time.monotonic() - 600)
            print('REFIT EQ', m, 'budget', budget, flush=True)
            rpos, rw, info = fit_eq_rule(jax, jnp, e, ctx['params'], Z, ctx['L'], ctx['M'], m, ctx['G'],
                                         ctx['phi'], time.monotonic() + budget, archived['eq_seed'],
                                         archived['eq_candidate_cap'], archived['eq_fit_states'])
            report['eq_rules'].append(info)
            d.write_json(path, report)
            if info['actual_m'] < 4 * ctx['M']:
                print('EQ RULE TOO SMALL, SKIPPED', m, info['actual_m'], flush=True)
                continue
            G5, Pq = quadrature_arrays(e, jnp, ctx['dec'], ctx['phi'], rpos, rw, ctx['L'], ctx['R'])
            rules[f'm{m}'] = dict(operators=make_operators(ctx, G5, Pq), info=info)
    else:
        for entry in cfg['validation_rules']:
            if entry['name'] == 'm256':
                continue
            rpos = np.asarray(entry['eq_indices'])
            rw = np.asarray(entry['eq_weights'])
            G5, Pq = quadrature_arrays(e, jnp, ctx['dec'], ctx['phi'], rpos, rw, ctx['L'], ctx['R'])
            rules[entry['name']] = dict(operators=make_operators(ctx, G5, Pq), info=entry['info'])
            report['eq_rules'].append(entry['info'])
    rules['full'] = dict(operators=rules['m256']['operators'],
                         info=dict(requested_m=None, actual_m=int((ctx['L'] - 1) ** 2),
                                   source='quadrature-free control: FOM-exact upwind advection on every interior node projected onto the same 64 weak modes'))
    if args.phase == 'calibration':
        report['eq_rules'].append(rules['full']['info'])
    d.write_json(path, report)

    foms = {p['name']: dict(query=make_fom_query(ctx, p), preset=p) for p in cfg['fom_controls']}

    if args.phase == 'calibration':
        cases, evidence = calibration_cases(ctx, args.reference, cfg, list(range(args.cases or 8)))
        report['reference_evidence'] = evidence
    else:
        cases, evidence = validation_cases(ctx, args.dataset, cfg, args.cases or 32)
        report['reference_evidence'] = evidence
    d.write_json(path, report)

    settings = cfg['settings']

    def arm_spec(quadrature, setting_name):
        s = settings[setting_name]
        return dict(quadrature=quadrature, setting=setting_name, step_budget=s['step_budget'],
                    evolution_gtol=s['evolution_gtol'], evolution_residual_scale=s['evolution_residual_scale'],
                    ic_budget=ctx['ic_budget'], initial_gtol=ctx['initial_gtol'],
                    actual_m=rules[quadrature]['info']['actual_m'])

    def build_arms(pairs):
        arms = {}
        for name, quadrature, setting_name in pairs:
            if quadrature not in rules:
                continue
            arms[name] = dict(query=make_arm_query(ctx, quadrature, settings[setting_name], trust),
                              operators=rules[quadrature]['operators'], spec=arm_spec(quadrature, setting_name))
        return arms

    if args.phase == 'calibration':
        # ---- stage 1: converged sentinel, untimed output-stability control -------
        sentinel_arms = build_arms([(f'{q}_{s}', q, s) for q in cfg['stage1']['quadratures']
                                    for s in cfg['stage1']['settings'] if q in rules])
        for case in [c for c in cases if c['index'] in cfg['stage1']['cases']]:
            u0 = jnp.asarray(case['supplied'])
            nu_device = jnp.asarray(case['nu'])
            fields = {}
            for name, arm in sentinel_arms.items():
                host = jax.tree_util.tree_map(np.asarray, arm['query'](u0, nu_device, arm['operators'], ctx['cold']))
                fields[name] = host[0]
                report['sentinel'].append(dict(case_id=case['record']['case_id'], method=name, spec=arm['spec'],
                                               error=error_metrics(host[0], case['reference']), **stopping_record(host)))
                d.write_json(path, report)
            for q in cfg['stage1']['quadratures']:
                if f'{q}_ultra' in fields and f'{q}_converged' in fields:
                    a, b = fields[f'{q}_ultra'], fields[f'{q}_converged']
                    report['sentinel'].append(dict(case_id=case['record']['case_id'],
                        stability_check=f'{q}: ultra versus converged',
                        relative_field_difference=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                        declared_stability_tolerance=cfg['stage1']['stability_tolerance']))
            d.write_json(path, report)
            # sampled versus full supplied-field initial fit, diagnostic only
            Gfull = ctx['G'] / ctx['L']
            target = jnp.asarray(case['supplied'][1:-1, 1:-1].reshape(-1)) / ctx['L']
            Qf, Rf = jnp.linalg.qr(Gfull, mode='reduced')
            yf = Qf.T @ target
            lm = ap.make_stationary_lm(lambda z, y, R: R @ e.sc.head(ctx['params'], z) - y, ctx['K'],
                                       ctx['ic_budget'], gtol=ctx['initial_gtol'])
            Hrot = e.sc.head(ctx['params'], ctx['candidate_z']) @ Rf.T
            idx = jnp.argmin(jnp.sum(Hrot * Hrot, 1) - 2 * Hrot @ yf)
            zf, rnf, itf, reasonf, gnf = jax.tree_util.tree_map(np.asarray,
                jax.jit(lambda z0: lm(z0, (yf, Rf), 0.))(ctx['candidate_z'][idx]))
            full_field = np.asarray(ctx['G'] @ e.sc.head(ctx['params'], jnp.asarray(zf))).reshape(ctx['L'] - 1, ctx['L'] - 1)
            xy, w, Q, R, HrotG, Hnorm = ctx['cold']
            ui = e.sample_field(jnp.asarray(case['supplied']), xy, ctx['L']) * w
            ys = Q.T @ ui
            idx = jnp.argmin(Hnorm - 2 * HrotG @ ys)
            zs, rns, its, reasons_, gns = jax.tree_util.tree_map(np.asarray,
                jax.jit(lambda z0: lm(z0, (ys, R), 0.))(ctx['candidate_z'][idx]))
            sampled_field = np.asarray(ctx['G'] @ e.sc.head(ctx['params'], jnp.asarray(zs))).reshape(ctx['L'] - 1, ctx['L'] - 1)
            interior = case['supplied'][1:-1, 1:-1]
            denominator = np.linalg.norm(interior)
            report['initial_fit_diagnostic'].append(dict(case_id=case['record']['case_id'],
                sampled_relative_field_error=float(np.linalg.norm(sampled_field - interior) / denominator),
                full_grid_relative_field_error=float(np.linalg.norm(full_field - interior) / denominator),
                sampled_iterations=int(its), full_grid_iterations=int(itf),
                sampled_reason=int(reasons_), full_grid_reason=int(reasonf),
                sampled_normalized_gradient=float(gns), full_grid_normalized_gradient=float(gnf),
                interpretation='diagnostic comparison only; the deployed cold initializer is unchanged in every timed arm'))
            d.write_json(path, report)
            del Gfull, Qf, Rf
        print('SENTINEL COMPLETE', flush=True)

        # ---- stage 2: quadrature control ----------------------------------------
        arms = build_arms([(f'{q}_{s}', q, s) for q in cfg['quadratures'] for s in ['native', 'converged']])
        ok = run_timed_pass(ctx, report, path, out, arms, foms, cases, reps, deadline, 'quadrature', cfg)
        for name in list(arms) + list(foms):
            report['summaries'][f'quadrature/{name}'] = summarize(report, 'quadrature', name)
        selected, selection = select_rule(report, cfg)
        report['rule_selection'] = selection
        d.write_json(path, report)
        if not ok or selected is None:
            report['elapsed_seconds'] = time.monotonic() - started
            d.write_json(path, report)
            return

        # ---- stages 3 and 4: iteration cap and stopping tolerance ----------------
        pairs = [(f'{selected}_cap{b}', selected, f'cap{b}') for b in cfg['stage3']['step_budgets']]
        pairs += [(f'{selected}_gtol{g:g}', selected, f'gtol{g:g}') for g in cfg['stage4']['evolution_gtols']]
        arms = build_arms(pairs)
        drift = {k: v for k, v in foms.items() if k in cfg['drift_controls']}
        run_timed_pass(ctx, report, path, out, arms, drift, cases, reps, deadline, 'effort', cfg)
        for name in list(arms) + list(drift):
            report['summaries'][f'effort/{name}'] = summarize(report, 'effort', name)
    else:
        shortlist = json.loads(args.shortlist.read_text())
        report['shortlist'] = shortlist
        arms = build_arms([(a['name'], a['quadrature'], a['setting']) for a in shortlist['arms']])
        run_timed_pass(ctx, report, path, out, arms, foms, cases, reps, deadline, 'validation', cfg)
        for name in list(arms) + list(foms):
            report['summaries'][f'validation/{name}'] = summarize(report, 'validation', name)

    # ---- native decoder compression of the supplied field, reported separately ----
    for case in cases[:8]:
        u0 = jnp.asarray(case['supplied'])
        name, arm = sorted(arms.items())[0]
        host = jax.tree_util.tree_map(np.asarray, arm['query'](u0, jnp.asarray(case['nu']), arm['operators'], ctx['cold']))
        native_t0 = np.asarray(e.output_field(ctx['G'] @ e.sc.head(ctx['params'], jnp.asarray(host[7][0])), ctx['L'], ctx['L']))
        interior = case['supplied'][1:-1, 1:-1]
        report['native_compression'].append(dict(case_id=case['record']['case_id'], method=name,
            relative_initial_compression_error=float(np.linalg.norm(native_t0[1:-1, 1:-1] - interior) / np.linalg.norm(interior)),
            note='the decoder cannot reproduce the supplied field exactly; every deployable arm returns the supplied field and still pays for the internal initial fit'))
    report['complete'] = True
    report['elapsed_seconds'] = time.monotonic() - started
    d.write_json(path, report)
    print('TUNING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
