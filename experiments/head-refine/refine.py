"""Per-query head refinement on Burgers 2D at one mesh, one frozen checkpoint.

The frozen bank, the latent dimension, the weak objective, the test-mode family,
the time discretization, the initializer policy, the Levenberg-Marquardt stopping
rule and the output contract are the head-ablation arm (a) contract. The only online
knobs are the variant (initial-only or per-step), the number n of Adam steps taken
on the head's own weights inside the timed query, and the anchor weight mu.

n = 0 is arm (a) with theta carried as a traced runtime operand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip
import arms as A
import refine_core as RC


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def theta_layout(params):
    """The exact flat layout `ravel_pytree` produces, recorded so a NumPy-only audit
    can rebuild the head without importing JAX."""
    leaves = jax.tree_util.tree_leaves(RC.theta_parts(params))
    shapes = [list(np.asarray(x).shape) for x in leaves]
    assert len(shapes) % 2 == 1 and shapes[-1][0] < shapes[-1][1], shapes
    return dict(order='jax.tree_util.tree_leaves of {h: [(w,b), ...], h_lin}, C-order ravel, '
                      'concatenated; the last leaf is the linear skip h_lin and the leaves '
                      'before it are the MLP weight/bias pairs in order',
                shapes=shapes, sizes=[int(np.prod(s)) for s in shapes],
                total=int(sum(int(np.prod(s)) for s in shapes)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
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

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L = int(cfg['intervals'])
    dt = cfg['dt']
    strict = cfg['strict']
    theta0, unravel = RC.flatten_theta(params)
    head_of = RC.make_head_of(unravel)
    P = int(theta0.size)

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
                  gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                  jax_version=jax.__version__, checkpoint_sha256=sha_file(a.checkpoint),
                  gate_reference=dict(file=sha_file(a.gate), **{k: gate[k] for k in
                                      ('source', 'source_sha256', 'job_id', 'gpu', 'arm')}),
                  K=K, R=R, intervals=L, theta_count=P,
                  theta_norm=float(jnp.linalg.norm(theta0)), theta_layout=theta_layout(params),
                  spatial_bank_frozen=True, network_weights_frozen_offline=True,
                  head_weights_refined_online=True, final_cohort_unopened=True,
                  output_times=[0, .05, .1, .15, .2, .25],
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output '
                                   'fields, INCLUDING every refinement step; same-invocation host '
                                   'transfers also measured; identical for every arm'),
                  adam=dict(**RC.ADAM, moments_reset_each_query=True,
                            implementation='inline, no optax in the timed path'),
                  reference=[], arm_setup=[], calibration={}, reconstruction=[],
                  invocations=[], declared_subjects=[], verification=ip.verify(),
                  gate_in_job={}, complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    report['cohort_note'] = ('exactly the six opened development cases of the head-ablation job, '
                             'which is what arm (a) used; no new case is opened')
    assert np.allclose(np.asarray(gate['physical_cases']), physical), 'cohort differs from abl01'
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'train/eval overlap'
    report['train_physical_sha256'] = sha_array(train_physical)
    report['calibration_case'] = dict(
        source='first draw of the training family; not an evaluation case',
        physical=train_physical[0].tolist(), sha256=sha_array(train_physical[0]))
    save()

    # --------------------------------------------------------------- reference
    rf, rt = cfg['reference_mesh'], cfg['reference_dt']
    refs = {}
    q, _ = e.make_fom(rf, rt)
    for case, phys in enumerate(physical):
        t = time.perf_counter()
        f, it, rn = host(q(jnp.asarray(e.initial(rf, phys)), float(phys[4]), 1e-11, 1e-9))
        assert np.isfinite(f).all() and np.max(rn) < 2e-11
        f = np.array(f[:, ::rf // L, ::rf // L], copy=True)
        refs[case] = f
        name = f'ref_L{rf}_dt{rt}_case{case}.npz'
        np.savez_compressed(out / name, fields=f, iterations=it, residuals=rn)
        report['reference'].append(dict(intervals=rf, dt=rt, case=case, artifact=name,
                                        max_relative_residual=float(np.max(rn)), downsampled_to=L,
                                        field_sha256=sha_array(f), seconds=time.perf_counter() - t))
        print('REFERENCE', case, round(time.perf_counter() - begin, 1), flush=True)
        save()
    del q
    jax.clear_caches()

    # ------------------------------------------------------- shared operators
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qb, Rb = A.whiten(G)
    jax.block_until_ready(Rb)
    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = jnp.asarray(Zold[::stride])
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    M = int(cfg['test_multiplier'] * K)
    linear = 'gj' if K <= cfg['gauss_jordan_max'] else 'lu'
    head0 = A.neural_head(params)

    operators = {}
    for quadrature in ('eq', 'dense'):
        t0 = time.perf_counter()
        m = int(cfg['quadrature_multiplier'] * M) if quadrature == 'eq' else None
        # arm (a)'s exact rule: the full code table and the arm (a) seed.
        data, info = A.build_operators(bank, L, M, quadrature, Zcoef=Zold, m=m,
                                       eq_seed=cfg['eq_seed'], candidate_cap=cfg['candidate_cap'],
                                       fit_states=cfg['fit_states'], head=head0)
        cold, cinfo = A.build_cold(bank, head0, Zsub, cfg['cold_axis_points'])
        operators[quadrature] = (data, cold)
        entry = dict(info)
        entry.update(trust_radius=trust, linear_solve=linear, cold=cinfo,
                     decoder_codes=int(len(Zsub)), code_stride=int(stride),
                     total_setup_seconds=time.perf_counter() - t0)
        report['arm_setup'].append(entry)
        print('OPERATORS', quadrature, round(time.perf_counter() - begin, 1), flush=True)
        save()

    rng = np.random.default_rng(cfg['redecode_seed'])
    nodes = np.sort(rng.choice((L - 1) ** 2, min(cfg['redecode_nodes'], (L - 1) ** 2), replace=False))
    np.savez_compressed(out / 'redecode_bank.npz', nodes=nodes,
                        bank=np.asarray(G)[nodes], theta0=np.asarray(theta0))
    report['redecode'] = dict(artifact='redecode_bank.npz', nodes=int(len(nodes)),
                              seed=cfg['redecode_seed'], node_sha256=sha_array(nodes),
                              note=('a fixed interior-node sample of the frozen bank, so a NumPy-only '
                                    'audit can recompute G h_theta(z) at those nodes from the saved '
                                    'refined weights and compare against the saved output field'))
    save()

    # ------------------------------------------------------------ query cache
    cache = {}

    def query_for(variant, n, quadrature):
        key = (variant, n, quadrature)
        if key not in cache:
            t0 = time.perf_counter()
            cache[key] = RC.make_query(head_of, K, L, dt, trust, quadrature, variant, n,
                                       linear=linear, **strict)
            report.setdefault('compiled_queries', []).append(
                dict(variant=variant, n=n, quadrature=quadrature,
                     build_seconds=time.perf_counter() - t0))
        return cache[key]

    # ----------------------------------------------------------- calibration
    cal_phys = train_physical[0]
    cal_u0 = jnp.asarray(e.initial(L, cal_phys))
    cal_nu = jnp.asarray(float(cal_phys[4]))
    data_eq, cold_eq = operators['eq']
    t0 = time.perf_counter()
    cal_fn = query_for('v1', cfg['calibration_n'], 'eq')
    trace = []
    for alpha in cfg['alpha_grid']:
        v = host(cal_fn(cal_u0, cal_nu, data_eq, cold_eq, theta0,
                        jnp.asarray(cfg['mu'][cfg['calibration_mu']]), jnp.asarray(alpha)))
        trace.append(dict(alpha=float(alpha), final_data_term=float(v['refine_data_term'][-1]),
                          final_anchor_term=float(v['refine_anchor_term'][-1]),
                          drift=float(v['drift'][-1]),
                          data_term_per_step=v['refine_data_term'].tolist(),
                          initial_data_term=float(v['ic_residual'] / v['ic_input_norm']) ** 2,
                          refine_reasons=v['refine_reasons'].tolist(),
                          budget_exits=int(np.sum(v['reasons'] == 0))))
        print('CALIBRATE alpha', alpha, 'data', trace[-1]['final_data_term'],
              'drift', trace[-1]['drift'], flush=True)
    best = min(trace, key=lambda r: r['final_data_term'])
    tie = [r for r in trace if r['final_data_term'] <= best['final_data_term']
           * (1 + cfg['calibration_tie_relative'])]
    alpha = float(min(r['alpha'] for r in tie))
    report['calibration'] = dict(
        rule=('lowest V1 data term after n = calibration_n alternating steps at the loose anchor, '
              'on one training-family case; ties within calibration_tie_relative broken by the '
              'smaller step size; then frozen for both variants, both anchor weights, every n and '
              'every evaluation case'),
        n=cfg['calibration_n'], mu=cfg['mu'][cfg['calibration_mu']], trace=trace,
        selected_alpha=alpha, seconds=time.perf_counter() - t0)
    print('CALIBRATED alpha', alpha, flush=True)
    save()

    # anchor diagnostic: where does mu start to bind? Diagnostic only; it selects nothing.
    t0 = time.perf_counter()
    diag_fn = query_for('v1', cfg['mu_diagnostic_n'], 'eq')
    diag = []
    for mu in cfg['mu_diagnostic_grid']:
        v = host(diag_fn(cal_u0, cal_nu, data_eq, cold_eq, theta0,
                         jnp.asarray(mu), jnp.asarray(alpha)))
        diag.append(dict(mu=float(mu), drift=float(v['drift'][-1]),
                         final_data_term=float(v['refine_data_term'][-1]),
                         final_anchor_term=float(v['refine_anchor_term'][-1]),
                         anchor_to_data_gradient=float(mu * v['refine_anchor_grad'][-1]
                                                       / max(float(v['refine_data_grad'][-1]), 1e-300))))
        print('MU-DIAG', mu, diag[-1]['drift'], diag[-1]['anchor_to_data_gradient'], flush=True)
    report['anchor_diagnostic'] = dict(n=cfg['mu_diagnostic_n'], alpha=alpha, rows=diag,
                                       seconds=time.perf_counter() - t0,
                                       note='calibration case only; selects nothing')
    save()

    # ------------------------------------------------------------------ arms
    specs = [dict(name='n0', variant='baseline', n=0, mu=None, mu_key=None, quadrature='eq')]
    for variant in ('v1', 'v2'):
        for n in [x for x in cfg['n_ladder'] if x > 0]:
            for mk in ('loose', 'tight'):
                specs.append(dict(name=f'{variant}_n{n}_mu{mk}', variant=variant, n=n,
                                  mu=cfg['mu'][mk], mu_key=mk, quadrature='eq'))
    for variant, n, mk in cfg['dense_controls']:
        nm = 'n0_dense' if variant == 'baseline' else f'{variant}_n{n}_mu{mk}_dense'
        specs.append(dict(name=nm, variant=variant, n=int(n), mu=(cfg['mu'][mk] if mk else None),
                          mu_key=mk, quadrature='dense'))

    built = []
    for s in specs:
        data, cold = operators[s['quadrature']]
        fn = query_for(s['variant'], s['n'], s['quadrature'])
        built.append(dict(**s, data=data, cold=cold, query=fn,
                          mu_value=jnp.asarray(0. if s['mu'] is None else s['mu']),
                          alpha_value=jnp.asarray(alpha)))
        report['declared_subjects'].append(dict(name=s['name'], method='rom', variant=s['variant'],
                                                n=s['n'], mu=s['mu'], mu_key=s['mu_key'],
                                                alpha=alpha, quadrature=s['quadrature'],
                                                solved_dimension=K, M=M,
                                                m=(int(cfg['quadrature_multiplier'] * M)
                                                   if s['quadrature'] == 'eq' else None),
                                                linear_solve=linear, dt=dt))
    for fs in cfg['fom_settings']:
        report['declared_subjects'].append(dict(method='fom', dt=dt, **fs))
    save()

    def run(b, case):
        return b['query'](jnp.asarray(e.initial(L, physical[case])),
                          jnp.asarray(float(physical[case, 4])), b['data'], b['cold'],
                          theta0, b['mu_value'], b['alpha_value'])

    # ------------------------------------- untimed diagnostics with the REFINED head
    recon = RC.make_reconstruction(head_of, K, cfg['recon_budget'], strict['gtol'], linear)
    heads_at = jax.jit(lambda th, Z: jax.vmap(lambda z: head_of(th, z))(Z))
    targets, projected, norms = {}, {}, {}
    bankproj = {}
    for case in refs:
        ref = refs[case]
        n0 = float(np.linalg.norm(ref[0]))
        norms[case] = n0
        vals = []
        for ti in range(ref.shape[0]):
            target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
            targets[(case, ti)] = target
            projected[(case, ti)] = G.T @ target
            vals.append(float(jnp.linalg.norm(Qb @ (Qb.T @ target) - target)) / n0)
        bankproj[case] = vals
    report['bank_projection'] = dict(
        note=('the best any coefficients at all could do in the frozen bank; unchanged by '
              'refinement, so it is a floor for every arm'),
        per_case={str(c): bankproj[c] for c in bankproj},
        worst=float(max(max(v) for v in bankproj.values())))
    save()

    thetas_saved = {}
    for b in built:
        t0 = time.perf_counter()
        rows = []
        finals = {}
        for case in refs:
            v = host(run(b, case))
            n0 = norms[case]
            th = jnp.asarray(v['theta'])
            man, iters = [], []
            for ti in range(refs[case].shape[0]):
                theta_t = th[ti] if th.shape[0] > 1 else th[0]
                Hc = heads_at(theta_t, Zsub)
                score = jnp.sum((Hc @ Rb.T) ** 2, 1) - 2 * (Hc @ projected[(case, ti)])
                starts = Zsub[jnp.argsort(score)[:cfg['recon_starts']]]
                z, rn, it, reason = host(recon(starts, targets[(case, ti)], G, theta_t))
                man.append(float(rn) / n0)
                iters.append(int(it))
            rows.append(dict(case=case, best_found_per_time=man, best_found_max=float(np.max(man)),
                             bank_projection_per_time=bankproj[case],
                             bank_projection_max=float(np.max(bankproj[case])),
                             drift=np.asarray(v['drift']).tolist(),
                             reconstruction_iterations=iters))
            finals[f'case{case}'] = np.asarray(v['theta'])[-1]
        entry = dict(arm=b['name'], variant=b['variant'], n=b['n'], mu=b['mu'],
                     quadrature=b['quadrature'], cases=rows,
                     worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
                     worst_best_found=float(max(r['best_found_max'] for r in rows)),
                     worst_drift=float(max(max(r['drift']) for r in rows)),
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
              'drift', round(entry['worst_drift'], 6), round(time.perf_counter() - begin, 1),
              flush=True)
        save()

    # ------------------------------------------------------------ timed queries
    foms = {'fft': ip.make_fom(L, dt, 'fft')}
    subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
    subjects += [dict(kind='fom', name=fs['name'], setting=fs) for fs in cfg['fom_settings']]
    inputs = [e.initial(L, phys) for phys in physical]
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, u, case):
        nu = float(physical[case, 4])
        if sub['kind'] == 'rom':
            b = built[sub['index']]
            return b['query'](u, jnp.asarray(nu), b['data'], b['cold'], theta0,
                              b['mu_value'], b['alpha_value'])
        fs = sub['setting']
        fn, pre = foms[fs['preconditioner']]
        return fn(u, jnp.asarray(nu), fs['ntol'], fs['ltol'], *pre)

    t = time.perf_counter()
    for sub in subjects:
        jax.block_until_ready(invoke(sub, jnp.asarray(inputs[0]), 0))
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
    print('WARMUP', round(time.perf_counter() - t, 1), flush=True)
    save()

    artifacts = {}
    for rep in range(cfg['reps']):
        for case in range(len(physical)):
            for i in order_rng.permutation(len(subjects)):
                sub = subjects[int(i)]
                e.burn(cfg['burn_seconds'])
                ht = time.perf_counter()
                u = jax.device_put(np.array(inputs[case], copy=True))
                jax.block_until_ready(u)
                gt = time.perf_counter()
                value = invoke(sub, u, case)
                jax.block_until_ready(value)
                gs = time.perf_counter() - gt
                f = np.asarray(value['fields'] if sub['kind'] == 'rom' else value[0])
                hs = time.perf_counter() - ht
                assert np.isfinite(f).all()
                h = sha_array(f)
                key = (sub['name'], case, h)
                row = dict(intervals=L, case=case, cohort=report['cohort_roles'][case], rep=rep,
                           kind=sub['kind'], name=sub['name'], gpu_seconds=gs, host_seconds=hs,
                           output_bytes=int(f.nbytes), field_sha256=h,
                           error=e.errors(f, refs[case], L), finite=True)
                if sub['kind'] == 'rom':
                    b = built[sub['index']]
                    v = host({k: value[k] for k in value if k != 'theta'})
                    reasons = v['reasons'].tolist()
                    rres = v['resolve_reasons'].tolist()
                    fres = v['refine_reasons'].tolist()
                    allr = reasons + rres + fres + [int(v['ic_reason'])]
                    if key not in artifacts:
                        fn = f"L{L}_{sub['name']}_case{case}_rep{rep}.npz"
                        np.savez_compressed(out / fn, fields=f, internal_latents=v['internal'],
                                            latents=v['latents'], drift=v['drift'])
                        artifacts[key] = fn
                    row.update(artifact=artifacts[key], variant=b['variant'], n=b['n'], mu=b['mu'],
                               mu_key=b['mu_key'], alpha=alpha, quadrature=b['quadrature'],
                               solved_dimension=K, M=M,
                               m=(int(cfg['quadrature_multiplier'] * M)
                                  if b['quadrature'] == 'eq' else None),
                               linear_solve=linear, dt=dt,
                               iterations=v['iterations'].tolist(), residuals=v['residuals'].tolist(),
                               stop_reasons=reasons, step_stationarity=v['stationarity'].tolist(),
                               resolve_iterations=v['resolve_iterations'].tolist(),
                               resolve_reasons=rres,
                               resolve_stationarity=v['resolve_stationarity'].tolist(),
                               refine_iterations=v['refine_iterations'].tolist(),
                               refine_reasons=fres,
                               refine_stationarity=v['refine_stationarity'].tolist(),
                               refine_data_term=v['refine_data_term'].tolist(),
                               refine_anchor_term=v['refine_anchor_term'].tolist(),
                               refine_data_grad=v['refine_data_grad'].tolist(),
                               refine_anchor_grad=v['refine_anchor_grad'].tolist(),
                               drift=v['drift'].tolist(), ic_iterations=int(v['ic_iterations']),
                               ic_reason=int(v['ic_reason']),
                               ic_stationarity=float(v['ic_stationarity']),
                               ic_residual=float(v['ic_residual']),
                               ic_input_norm=float(v['ic_input_norm']),
                               ic_relative_residual=float(v['ic_residual'])
                               / max(float(v['ic_input_norm']), 1e-300),
                               budget_exits=int(sum(1 for r in allr if r == 0)),
                               rejected_exits=int(sum(1 for r in allr if r == 3)),
                               latent_solves=len(allr),
                               # Two separate, both reported, statuses: the campaign's
                               # normalized-gradient stationarity, and completion under the
                               # shared stopping rule with no budget and no rejection exit,
                               # over EVERY latent solve including the refinement re-solves.
                               stationary=bool(max([float(np.max(v['stationarity'])),
                                                    float(v['ic_stationarity'])]
                                                   + ([float(np.max(v['resolve_stationarity']))]
                                                      if len(rres) else [])
                                                   + ([float(np.max(v['refine_stationarity']))]
                                                      if len(fres) else []))
                                               <= strict['gtol'] * (1 + 1e-7)),
                               completed=bool(all(r in (1, 2, 4) for r in allr)))
                else:
                    v = host(value)
                    if key not in artifacts:
                        fn = f"L{L}_{sub['name']}_case{case}_rep{rep}.npz"
                        np.savez_compressed(out / fn, fields=f)
                        artifacts[key] = fn
                    row.update(artifact=artifacts[key], iterations=v[1].tolist(),
                               residuals=v[2].tolist(),
                               nonlinear_converged=bool(np.max(v[2])
                                                        <= sub['setting']['ntol'] * (1 + 1e-9)))
                report['invocations'].append(row)
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    # --------------------------------------------- gate (ii): n0 versus abl01
    want = {g['case']: g for g in gate['cases']}
    got = {x['case']: x for x in report['invocations'] if x['name'] == 'n0'}
    shared = sorted(set(want) & set(got))
    deltas = [abs(got[c]['error']['fixed_initial_max'] - want[c]['fixed_initial_max'])
              / max(want[c]['fixed_initial_max'], 1e-300) for c in shared]
    report['gate_in_job'] = dict(
        arm='n0', reference_arm=gate['arm'], reference_job=gate['job_id'],
        reference_gpu=gate['gpu'], this_gpu=report['gpu'], compared=len(shared),
        worst_relative_delta=(max(deltas) if deltas else None), tolerance=1e-9,
        bitwise_identical_fields=int(sum(1 for c in shared
                                         if want[c]['field_sha256'] == got[c]['field_sha256'])),
        passed=bool(shared and max(deltas) <= 1e-9),
        note=('n0 versus the head-ablation job arm a_neural_eq on the same six cases and mesh; '
              'bitwise code equivalence is established separately by the local smoke'))
    print('GATE n0 vs abl01', report['gate_in_job']['worst_relative_delta'], flush=True)
    save()

    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['checkpoint_sha256'] == report['checkpoint_sha256_after']
    report['weights_artifacts'] = thetas_saved
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('HEAD REFINE BURGERS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
