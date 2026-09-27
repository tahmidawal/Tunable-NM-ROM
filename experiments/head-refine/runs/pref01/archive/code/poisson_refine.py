"""Per-query head refinement on Poisson 2D at one mesh, one frozen bank and head.

The Poisson weak residual is exactly B h(z) - f_m, with no time stepping and no
quadrature approximation. The query is therefore a SINGLE STATIC SOLVE: there is no
per-step structure for a per-step variant to exploit, so V2 collapses onto V1 and
only V1 is run. V1 here means refining the head against the supplied source-projected
weak data, which on this PDE is simultaneously the initial fit and the solve.

Staged flat: every module sits beside this file on the cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import pilot as PL
import arms as A
import sep_common as sc
import refine_core as RC


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def radius(Z):
    Z = np.asarray(Z)
    return float(np.max(np.linalg.norm(Z - Z.mean(0), axis=1)))


def theta_layout(params):
    leaves = jax.tree_util.tree_leaves(RC.theta_parts(params))
    shapes = [list(np.asarray(x).shape) for x in leaves]
    assert len(shapes) % 2 == 1 and shapes[-1][0] < shapes[-1][1], shapes
    return dict(order='jax.tree_util.tree_leaves of {h: [(w,b), ...], h_lin}, C-order ravel, '
                      'concatenated; the last leaf is the linear skip h_lin and the leaves '
                      'before it are the MLP weight/bias pairs in order',
                shapes=shapes, sizes=[int(np.prod(s)) for s in shapes],
                total=int(sum(int(np.prod(s)) for s in shapes)))


def query_once(kernel, source, ops, predictions, codes, B, bank, theta0, mu, alpha):
    start = time.perf_counter()
    src = jax.device_put(source)
    src.block_until_ready()
    input_end = time.perf_counter()
    out = kernel(src, ops['S'], ops['I'], ops['J'], ops['W'], B, bank, predictions, codes,
                 theta0, mu, alpha)
    jax.block_until_ready(out)
    device_end = time.perf_counter()
    theta = out.pop('theta')
    v = jax.device_get(out)
    end = time.perf_counter()
    row = dict(total_seconds=end - start, input_seconds=input_end - start,
               fused_device_seconds=device_end - input_end, output_seconds=end - device_end,
               residual=float(v['residual']), iterations=int(v['iterations']),
               reason=int(v['reason']), stationarity=float(v['stationarity']),
               first_residual=float(v['first_residual']),
               first_iterations=int(v['first_iterations']), first_reason=int(v['first_reason']),
               first_stationarity=float(v['first_stationarity']),
               relative_residual=float(v['residual']) / max(float(v['source_norm']), 1e-300),
               selected_code_index=int(v['index']), latent=np.asarray(v['latent']).tolist(),
               drift=float(v['drift']),
               refine_residuals=np.asarray(v['refine_residuals']).tolist(),
               refine_iterations=np.asarray(v['refine_iterations']).tolist(),
               refine_reasons=np.asarray(v['refine_reasons']).tolist(),
               refine_stationarity=np.asarray(v['refine_stationarity']).tolist(),
               refine_data_term=np.asarray(v['refine_data_term']).tolist(),
               refine_anchor_term=np.asarray(v['refine_anchor_term']).tolist(),
               refine_data_grad=np.asarray(v['refine_data_grad']).tolist(),
               refine_anchor_grad=np.asarray(v['refine_anchor_grad']).tolist())
    return np.asarray(v['field']), row, theta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--basis', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    gate = json.loads(Path(a.gate).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    params, codes, ckcfg = sc.load_pkl(a.checkpoint)
    basis = np.load(a.basis)
    np.testing.assert_array_equal(np.asarray(codes), basis['training_latents'])
    K = int(np.asarray(codes).shape[1])
    R = int(np.asarray(params['h_lin']).shape[1])
    n = int(cfg['intervals'])
    theta0, unravel = RC.flatten_theta(params)
    head_of = RC.make_head_of(unravel)
    head0 = lambda z: sc.head(params, z)
    Pn = int(theta0.size)

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
                  gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                  jax_version=jax.__version__, checkpoint_sha256=sha_file(a.checkpoint),
                  basis_sha256=sha_file(a.basis), gate_reference=dict(
                      file=sha_file(a.gate), **{k: gate[k] for k in
                                                ('source', 'source_sha256', 'job_id', 'gpu', 'arm')}),
                  K=K, R=R, intervals=n, theta_count=Pn,
                  theta_norm=float(jnp.linalg.norm(theta0)), theta_layout=theta_layout(params),
                  spatial_bank_frozen=True, network_weights_frozen_offline=True,
                  head_weights_refined_online=True, final_cohort_unopened=True,
                  variant_note=('the Poisson query is a single static solve, so the per-step variant '
                                'V2 has no per-step structure to exploit and collapses onto V1; only '
                                'V1 is defined and run here'),
                  timing_contract=('host source array in to dense nodal field out, including '
                                   'projection, initialization, every refinement step, the nonlinear '
                                   'solve and the decode; identical for every arm'),
                  adam=dict(**RC.ADAM, moments_reset_each_query=True,
                            implementation='inline, no optax in the timed path'),
                  references=[], arm_setup=[], calibration={}, reconstruction=[],
                  invocations=[], declared_subjects=[], gate_in_job={}, complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    draws = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                            C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    report['cohort'] = dict(parameters=draws.tolist(), sha256=sha_array(draws),
                            groups=['existing development'] * cfg['eval_count']
                                   + ['fresh development'] * cfg['fresh_count'])
    assert np.allclose(np.asarray(gate['cohort']['parameters']), draws), 'cohort differs from pabl01'
    train_draws = C.source_params(cfg['train_seed'], cfg['train_count'])
    assert not any(np.allclose(t, s) for t in train_draws for s in draws), 'train/eval overlap'
    report['train_parameters_sha256'] = sha_array(train_draws)
    report['calibration_case'] = dict(
        source='first draw of the training family; not an evaluation case',
        parameters=train_draws[0].tolist(), sha256=sha_array(train_draws[0]))
    save()

    nlo, nhi = cfg['reference_intervals'][0], cfg['reference_intervals'][-1]
    chain = {}
    for case, param in enumerate(draws):
        t0 = time.perf_counter()
        fine = PL.reference(param, nhi)
        coarse = PL.reference(param, nlo)
        chain[case] = np.array(fine[::nhi // n, ::nhi // n], copy=True)
        name = f'ref_case{case}.npz'
        np.savez_compressed(out / name, field=chain[case])
        report['references'].append(dict(
            case=case, fine_intervals=nhi, coarse_intervals=nlo, artifact=name, restricted_to=n,
            reference_delta=C.relative(coarse, fine[::nhi // nlo, ::nhi // nlo]),
            fine_sha256=sha_array(fine), restricted_sha256=sha_array(chain[case]),
            seconds=time.perf_counter() - t0))
        del fine, coarse
    save()
    print('REFERENCES done', round(time.perf_counter() - begin, 1), flush=True)

    lam = jnp.asarray(C.eigenvalues(n))
    ops = C.assemble(params, np.asarray(codes), n, cfg['requested_modes'], cfg['lm_budget'])
    bank, B, S, I, J = ops['bank'], ops['B'], ops['S'], ops['I'], ops['J']
    M = int(B.shape[0])
    linear = 'gj' if K <= cfg['gauss_jordan_max'] else 'lu'
    trust = radius(np.asarray(codes))
    predictions = jax.jit(jax.vmap(lambda z: B @ head0(z)))(jnp.asarray(codes))
    codes_j = jnp.asarray(codes)
    report['arm_setup'].append(dict(intervals=n, arm='shared_assembly', M=M, trust_radius=trust,
                                    linear_solve=linear, **{k: ops['info'][k] for k in
                                    ('retained_modes', 'stored_features', 'retained_bank_rank',
                                     'trust_delta', 'operator_sha256', 'bank_sha256')}))
    save()

    rng = np.random.default_rng(cfg['redecode_seed'])
    nodes = np.sort(rng.choice((n - 1) ** 2, min(cfg['redecode_nodes'], (n - 1) ** 2), replace=False))
    np.savez_compressed(out / 'redecode_bank.npz', nodes=nodes,
                        bank=np.asarray(bank)[nodes], theta0=np.asarray(theta0))
    report['redecode'] = dict(artifact='redecode_bank.npz', nodes=int(len(nodes)),
                              seed=cfg['redecode_seed'], node_sha256=sha_array(nodes),
                              note=('a fixed interior-node sample of the frozen bank, so a NumPy-only '
                                    'audit can recompute bank @ h_theta(z) at those nodes from the '
                                    'saved refined weights and compare against the saved field'))
    save()

    cache = {}

    def query_for(nn):
        if nn not in cache:
            t0 = time.perf_counter()
            cache[nn] = RC.make_poisson_query(head_of, nn, n, trust, cfg['lm_budget'],
                                              cfg['stationarity_tolerance'], linear)
            report.setdefault('compiled_queries', []).append(
                dict(n=nn, build_seconds=time.perf_counter() - t0))
        return cache[nn]

    # ----------------------------------------------------------- calibration
    cal_source = C.full_source(n, train_draws[0])
    t0 = time.perf_counter()
    cal_fn = query_for(cfg['calibration_n'])
    trace = []
    for alpha in cfg['alpha_grid']:
        _, row, _ = query_once(cal_fn, cal_source, ops, predictions, codes_j, B, bank, theta0,
                               jnp.asarray(cfg['mu'][cfg['calibration_mu']]), jnp.asarray(alpha))
        trace.append(dict(alpha=float(alpha), final_data_term=row['refine_data_term'][-1],
                          final_anchor_term=row['refine_anchor_term'][-1], drift=row['drift'],
                          data_term_per_step=row['refine_data_term'],
                          initial_data_term=(row['first_residual']
                                             / max(row['residual'] / max(row['relative_residual'],
                                                                         1e-300), 1e-300)) ** 2,
                          refine_reasons=row['refine_reasons']))
        print('CALIBRATE alpha', alpha, 'data', trace[-1]['final_data_term'],
              'drift', trace[-1]['drift'], flush=True)
    best = min(trace, key=lambda r: r['final_data_term'])
    tie = [r for r in trace if r['final_data_term'] <= best['final_data_term']
           * (1 + cfg['calibration_tie_relative'])]
    alpha = float(min(r['alpha'] for r in tie))
    report['calibration'] = dict(
        rule=('lowest V1 data term after n = calibration_n alternating steps at the loose anchor, '
              'on one training-family source; ties within calibration_tie_relative broken by the '
              'smaller step size; then frozen for both anchor weights, every n and every '
              'evaluation source'),
        n=cfg['calibration_n'], mu=cfg['mu'][cfg['calibration_mu']], trace=trace,
        selected_alpha=alpha, seconds=time.perf_counter() - t0)
    print('CALIBRATED alpha', alpha, flush=True)
    save()

    t0 = time.perf_counter()
    diag_fn = query_for(cfg['mu_diagnostic_n'])
    diag = []
    for mu in cfg['mu_diagnostic_grid']:
        _, row, _ = query_once(diag_fn, cal_source, ops, predictions, codes_j, B, bank, theta0,
                               jnp.asarray(mu), jnp.asarray(alpha))
        diag.append(dict(mu=float(mu), drift=row['drift'],
                         final_data_term=row['refine_data_term'][-1],
                         final_anchor_term=row['refine_anchor_term'][-1],
                         anchor_to_data_gradient=float(mu * row['refine_anchor_grad'][-1]
                                                       / max(row['refine_data_grad'][-1], 1e-300))))
        print('MU-DIAG', mu, diag[-1]['drift'], diag[-1]['anchor_to_data_gradient'], flush=True)
    report['anchor_diagnostic'] = dict(n=cfg['mu_diagnostic_n'], alpha=alpha, rows=diag,
                                       seconds=time.perf_counter() - t0,
                                       note='calibration source only; selects nothing')
    save()

    # ------------------------------------------------------------------ arms
    specs = [dict(name='n0', variant='baseline', n=0, mu=None, mu_key=None)]
    for nn in [x for x in cfg['n_ladder'] if x > 0]:
        for mk in ('loose', 'tight'):
            specs.append(dict(name=f'v1_n{nn}_mu{mk}', variant='v1', n=nn,
                              mu=cfg['mu'][mk], mu_key=mk))
    built = []
    for s in specs:
        built.append(dict(**s, kernel=query_for(s['n']),
                          mu_value=jnp.asarray(0. if s['mu'] is None else s['mu']),
                          alpha_value=jnp.asarray(alpha)))
        report['declared_subjects'].append(dict(intervals=n, name=s['name'], method='rom',
                                                variant=s['variant'], n=s['n'], mu=s['mu'],
                                                mu_key=s['mu_key'], alpha=alpha, k=K, M=M,
                                                linear_solve=linear))
    report['declared_subjects'].append(dict(intervals=n, name='dst_direct', method='fom'))
    save()

    sources = [C.full_source(n, q) for q in draws]

    # --------------------------------- untimed diagnostics with the REFINED head
    recon = RC.make_reconstruction(head_of, K, cfg['recon_budget'],
                                   cfg['stationarity_tolerance'], linear)
    heads_at = jax.jit(lambda th, Z: jax.vmap(lambda z: head_of(th, z))(Z))
    Qb, Rb = A.whiten(bank)
    jax.block_until_ready(Rb)
    targets, projected, norms, bankproj = {}, {}, {}, {}
    for case in chain:
        target = jnp.asarray(chain[case][1:-1, 1:-1].ravel())
        targets[case] = target
        projected[case] = bank.T @ target
        norms[case] = float(np.linalg.norm(chain[case]))
        bankproj[case] = float(jnp.linalg.norm(Qb @ (Qb.T @ target) - target)) / norms[case]
    report['bank_projection'] = dict(
        note=('the best any coefficients at all could do in the frozen bank; unchanged by '
              'refinement, so it is a floor for every arm'),
        per_case={str(c): bankproj[c] for c in bankproj},
        worst=float(max(bankproj.values())))
    save()

    thetas_saved = {}
    for b in built:
        t0 = time.perf_counter()
        rows, finals = [], {}
        for case in chain:
            _, row, theta = query_once(b['kernel'], sources[case], ops, predictions, codes_j, B,
                                       bank, theta0, b['mu_value'], b['alpha_value'])
            Hc = heads_at(theta, codes_j)
            score = jnp.sum((Hc @ Rb.T) ** 2, 1) - 2 * (Hc @ projected[case])
            starts = codes_j[jnp.argsort(score)[:cfg['recon_starts']]]
            z, rn, it, reason = jax.device_get(recon(starts, targets[case], bank, theta))
            rows.append(dict(case=case, best_found=float(rn) / norms[case],
                             bank_projection=bankproj[case], drift=row['drift'],
                             reconstruction_iterations=int(it)))
            finals[f'case{case}'] = np.asarray(theta)
        entry = dict(arm=b['name'], variant=b['variant'], n=b['n'], mu=b['mu'], cases=rows,
                     worst_bank_projection=float(max(r['bank_projection'] for r in rows)),
                     worst_best_found=float(max(r['best_found'] for r in rows)),
                     worst_drift=float(max(r['drift'] for r in rows)),
                     seconds=time.perf_counter() - t0)
        if b['n'] > 0:
            fn = f"theta_{b['name']}.npz"
            np.savez_compressed(out / fn, **finals)
            entry['weights_artifact'] = fn
            thetas_saved[b['name']] = fn
        else:
            entry['weights_artifact'] = None
            entry['weights_note'] = 'theta = theta_0 exactly; theta_0 is stored in redecode_bank.npz'
        report['reconstruction'].append(entry)
        print('RECON', b['name'], round(entry['worst_best_found'] * 100, 5),
              'drift', round(entry['worst_drift'], 8), round(time.perf_counter() - begin, 1),
              flush=True)
        save()

    # ------------------------------------------------------------ timed queries
    subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
    subjects.append(dict(kind='fom', name='dst_direct'))
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, case):
        if sub['kind'] == 'rom':
            b = built[sub['index']]
            f, row, _ = query_once(b['kernel'], sources[case], ops, predictions, codes_j, B, bank,
                                   theta0, b['mu_value'], b['alpha_value'])
            return f, row
        return C.fom_query(sources[case], lam)

    t = time.perf_counter()
    for sub in subjects:
        invoke(sub, 0)
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
    print('WARMUP', round(time.perf_counter() - t, 1), flush=True)
    save()

    artifacts = {}
    for rep in range(cfg['repetitions']):
        for case in range(len(draws)):
            for i in order_rng.permutation(len(subjects)):
                sub = subjects[int(i)]
                C.burn(cfg['burn_seconds'])
                field, row = invoke(sub, case)
                assert np.isfinite(field).all()
                h = sha_array(field)
                key = (sub['name'], case, h)
                if key not in artifacts:
                    fn = f"n{n}_{sub['name']}_case{case}_rep{rep}.npz"
                    np.savez_compressed(out / fn, field=field)
                    artifacts[key] = fn
                b = built[sub['index']] if sub['kind'] == 'rom' else None
                allr = ([row['first_reason']] + row['refine_reasons']) if b else []
                report['invocations'].append(dict(
                    intervals=n, case=case, group=report['cohort']['groups'][case], rep=rep,
                    kind=sub['kind'], name=sub['name'], field_sha256=h, artifact=artifacts[key],
                    physical_error=C.relative(field, chain[case]),
                    source_sha256=C.sha(sources[case]), finite=True,
                    k=(K if b else None), M=(M if b else None),
                    variant=(b['variant'] if b else 'fom'), n=(b['n'] if b else None),
                    mu=(b['mu'] if b else None), mu_key=(b['mu_key'] if b else None),
                    alpha=(alpha if b else None), linear_solve=(linear if b else None),
                    latent_solves=len(allr),
                    budget_exits=int(sum(1 for r in allr if r == 0)),
                    rejected_exits=int(sum(1 for r in allr if r == 3)),
                    completed=(bool(all(r in (1, 2, 4) for r in allr)) if b else None),
                    stationary=(bool(max([row['stationarity'], row['first_stationarity']]
                                         + (row['refine_stationarity'] or []))
                                     <= cfg['stationarity_tolerance'] * (1 + 1e-7)) if b else None),
                    **row))
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    want = {g['case']: g for g in gate['cases']}
    got = {x['case']: x for x in report['invocations'] if x['name'] == 'n0'}
    shared = sorted(set(want) & set(got))
    deltas = [abs(got[c]['physical_error'] - want[c]['physical_error'])
              / max(want[c]['physical_error'], 1e-300) for c in shared]
    report['gate_in_job'] = dict(
        arm='n0', reference_arm=gate['arm'], reference_job=gate['job_id'],
        reference_gpu=gate['gpu'], this_gpu=report['gpu'], compared=len(shared),
        worst_relative_delta=(max(deltas) if deltas else None), tolerance=1e-6,
        bitwise_identical_fields=int(sum(1 for c in shared
                                         if want[c]['field_sha256'] == got[c]['field_sha256'])),
        passed=bool(shared and max(deltas) <= 1e-6),
        note=('n0 versus the head-ablation Poisson job arm a_neural at the same mesh and sources; '
              'the declared cross-job tolerance is 1e-6, the same bar the correction ladder used, '
              'because the two jobs may run on different A100 models where XLA can select '
              'different kernels'))
    print('GATE n0 vs pabl01', report['gate_in_job']['worst_relative_delta'], flush=True)

    report['weights_artifacts'] = thetas_saved
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('HEAD REFINE POISSON COMPLETE', flush=True)


if __name__ == '__main__':
    main()
