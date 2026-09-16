"""Cheap correction ladder on Burgers 2D: variable projection, test count, EQ per rung.

Same frozen checkpoint, same nested direction rule and seed, same reachable set, same
weak objective, same initializer policy, same stopping tolerance and budgets, same mesh,
same time step and the same six opened development cases as the audited ladder
(`experiments/head-ablation`, job 3713867). Three things change, each isolated:

  1 the solver        joint LM on K+q unknowns -> variable projection on z alone
  2 the test count    M = 4(K+q) -> also 2(K+q) and a fixed M
  3 the quadrature    one bounded-NNLS empirical rule per rung, on enriched codes

Everything is written incrementally to `result.json` so a truncated job is still
collectable.
"""
from __future__ import annotations

import argparse
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
import ladder as LD
import varpro as VP
import directions as DIR

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump


def test_count(rule, K, q, fixed):
    if rule == 'm4':
        return 4 * (K + q)
    if rule == 'm2':
        return 2 * (K + q)
    if rule == 'm256':
        return int(fixed)
    raise ValueError(rule)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
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

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
                  checkpoint_sha256=sha_file(a.checkpoint), K=K, R=R, intervals=L,
                  spatial_bank_frozen=True, network_weights_frozen=True, final_cohort_unopened=True,
                  output_times=[0, .05, .1, .15, .2, .25],
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output fields; '
                                   'same-invocation host transfers also measured; identical for every arm'),
                  reference=[], snapshots={}, directions={}, directions_flat={}, arm_setup=[],
                  reconstruction=[], invocations=[], declared_subjects=[], gates={},
                  verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    report['cohort_note'] = ('exactly the six opened development cases of the audited ladder and of '
                             'head-ablation arm (a); no new case is opened')
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'training/eval overlap'
    report['train_physical_sha256'] = sha_array(train_physical)
    save()

    # ------------------------------------------------------------- references --
    rf, rt = cfg['reference_mesh'], cfg['reference_dt']
    refs = {}
    q_fom, _ = e.make_fom(rf, rt)
    for case, phys in enumerate(physical):
        t = time.perf_counter()
        f, it, rn = host(q_fom(jnp.asarray(e.initial(rf, phys)), float(phys[4]), 1e-11, 1e-9))
        assert np.isfinite(f).all() and np.max(rn) < 2e-11
        f = np.array(f[:, ::rf // L, ::rf // L], copy=True)
        refs[case] = f
        name = f'ref_L{rf}_dt{rt}_case{case}.npz'
        np.savez_compressed(out / name, fields=f, iterations=it, residuals=rn)
        got = sha_array(f)
        want = cfg.get('expected_reference_sha256', {}).get(str(case))
        report['reference'].append(dict(intervals=rf, dt=rt, case=case, artifact=name,
                                        max_relative_residual=float(np.max(rn)), downsampled_to=L,
                                        field_sha256=got, qlad01_field_sha256=want,
                                        bitwise_matches_qlad01=(None if want is None else got == want),
                                        seconds=time.perf_counter() - t))
        print('REFERENCE', case, round(time.perf_counter() - begin, 1), flush=True)
        save()
    del q_fom
    jax.clear_caches()

    # ------------------------------------------------------------- snapshots ---
    U, sinfo = LD.generate_snapshots(L, dt, train_physical, cfg['train_state_stride'],
                                     cfg['snapshot_ntol'], cfg['snapshot_ltol'])
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qb, Rb = A.whiten(G)
    jax.block_until_ready(Rb)
    Ut = jnp.asarray(U.T)
    coef_truth = jnp.linalg.solve(Rb, Qb.T @ Ut).T
    sinfo['bank_projection_relative_rms'] = float(
        jnp.linalg.norm(Ut - Qb @ (Qb.T @ Ut)) / jnp.linalg.norm(Ut))
    del Ut, U
    jax.clear_caches()
    report['snapshots'][str(L)] = sinfo
    save()
    print('SNAPSHOTS', round(sinfo['seconds'], 1), flush=True)

    # ------------------------------------------------------------- directions --
    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    dcfg = {**cfg, 'strict': strict}
    Cfull, Ct, Zstar, rho, dinfo = DIR.audited(params, Rb, coef_truth, Zsub, K, dcfg)
    want = cfg.get('expected_directions_sha256')
    dinfo['expected_sha256'] = want
    dinfo['bitwise_matches_qlad01'] = (None if want is None else dinfo['directions_sha256'] == want)
    report['directions'] = dinfo
    report['gates']['directions_hash'] = dict(expected=want, got=dinfo['directions_sha256'],
                                              passed=dinfo['bitwise_matches_qlad01'])
    save()
    print('DIRECTIONS audited', round(dinfo['seconds'], 1), 'rank', dinfo['available_rank'],
          'hash_match', dinfo['bitwise_matches_qlad01'], flush=True)
    if cfg.get('run_flat_directions', True):
        Cf, _, _, _, finfo = DIR.flat(params, Rb, coef_truth, Zsub, K, dcfg)
        finfo['max_abs_difference_from_audited'] = float(np.max(np.abs(np.asarray(Cf - Cfull))))
        finfo['bitwise_identical_to_audited'] = bool(
            finfo['directions_sha256'] == dinfo['directions_sha256'])
        finfo['compile_seconds_saved'] = dinfo['seconds'] - finfo['seconds']
        report['directions_flat'] = finfo
        del Cf
        jax.clear_caches()
        save()
        print('DIRECTIONS flat', round(finfo['seconds'], 1), 'identical',
              finfo['bitwise_identical_to_audited'], flush=True)

    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))

    # ---------------------------------------------------------------- subjects -
    specs = []
    seen = set()

    def add(q, rule, quadrature, variant, fitter=None, tag=''):
        M = test_count(rule, K, q, cfg['fixed_test_count'])
        if M <= K + q:
            return
        name = f'q{q}_{rule}_{quadrature}_{variant}' + tag
        if name in seen:
            return
        seen.add(name)
        specs.append(dict(name=name, q=q, M=M, rule=rule, quadrature=quadrature,
                          variant=variant, fitter=(fitter or 'retained')))

    for blk in cfg['variant_arms']:
        for v in blk['variants']:
            add(blk['q'], blk['rule'], blk['quadrature'], v)
    for arm in cfg['eq_gate_arms']:
        add(arm['q'], arm['rule'], 'eq', arm['variant'], arm['fitter'],
            '' if arm['fitter'] == 'retained' else '_bnd')
    lad = cfg['ladder_arms']
    for rule, spec in lad['rules'].items():
        for q in spec['q']:
            add(q, rule, 'dense', lad['variant'])
        for q in spec.get('eq_q', []):
            add(q, rule, 'eq', lad['variant'], 'bounded',
                '' if rule != 'm4' else '_bnd')
    # q = 0 control at the ladder's largest test count, so the growth of M is separable
    Mmax = test_count('m4', K, max(lad['rules']['m4']['q']), cfg['fixed_test_count'])
    specs.append(dict(name='q0_Mmax_dense_varpro', q=0, M=Mmax, rule='Mmax',
                      quadrature='dense', variant='varpro', fitter='retained'))

    built, datas, colds, recon_done = [], {}, {}, {}
    for s in specs:
        q, M = s['q'], s['M']
        assert M > K + q, (s['name'], M, K + q)
        C = Cfull[:, :q]
        head = VP.corrected_head(params, C, K)
        if q not in colds:
            Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
            colds[q] = A.build_cold(bank, head, Zaug, cfg['cold_axis_points'])
        cold, cinfo = colds[q]
        key = (q, M, s['quadrature'], s['fitter'])
        t0 = time.perf_counter()
        if key not in datas:
            if s['quadrature'] == 'eq':
                m = int(min(cfg['quadrature_multiplier'] * M, cfg['quadrature_cap']))
                if q == 0:
                    # the retained rule reuses arm (a)'s exact full code table and seed
                    W = np.concatenate((Zold, np.zeros((len(Zold), 0))), axis=1)
                else:
                    W = VP.enriched_codes(Zstar, rho, Rb, Ct, q)
            else:
                m, W = None, None
            datas[key] = VP.build_operators(
                bank, L, M, s['quadrature'], head=head, Wcodes=W, m=m, fitter=s['fitter'],
                eq_seed=cfg['eq_seed'], candidate_cap=cfg['candidate_cap'],
                fit_states=cfg['fit_states'], max_fit_rows=cfg['max_fit_rows'],
                eq_seconds=cfg['eq_seconds'])
        data, info = datas[key]
        info = dict(info)
        dim = K + q
        linear = 'gj' if dim <= cfg['gauss_jordan_max'] else 'lu'
        query = VP.make_query(params, C, K, q, L, dt, trust, s['quadrature'], s['variant'],
                              linear=linear, inner_iters=cfg['inner_iters'],
                              inner_damping=cfg['inner_damping'], alt_rounds=cfg['alt_rounds'],
                              **strict)
        info.update(arm=s['name'], q=q, rule=s['rule'], variant=s['variant'], fitter=s['fitter'],
                    solved_dimension=dim, outer_dimension=(K if (s['variant'] != 'joint' and q) else dim),
                    dt=dt, trust_radius=trust, linear_solve=linear, cold=cinfo,
                    total_setup_seconds=time.perf_counter() - t0, correction_directions_used=q)
        report['arm_setup'].append(info)
        report['declared_subjects'].append(dict(
            name=s['name'], method='rom', q=q, M=M, m=info.get('m'), rule=s['rule'],
            quadrature=s['quadrature'], variant=s['variant'], fitter=s['fitter'],
            solved_dimension=dim, linear_solve=linear, dt=dt))
        built.append(dict(**s, data=data, cold=cold, query=query, head=head, C=C,
                          linear_solve=linear, m=info.get('m')))
        print('ARM', s['name'], 'M', M, 'm', info.get('m'),
              round(info['total_setup_seconds'], 1), flush=True)
        save()

    foms = {'fft': ip.make_fom(L, dt, 'fft')}
    for fs in cfg['fom_settings']:
        report['declared_subjects'].append(dict(method='fom', dt=dt, **fs))

    # ------------------------------------------------- untimed diagnostics -----
    for b in built:
        if b['q'] in recon_done:
            continue
        head, q = b['head'], b['q']
        Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
        Hc = jax.jit(jax.vmap(head))(jnp.asarray(Zaug))
        Hn = jnp.sum((Hc @ Rb.T) ** 2, 1)
        recon = A.make_reconstruction(head, K + q, L, cfg['recon_budget'],
                                      linear=('gj' if K + q <= cfg['gauss_jordan_max'] else 'lu'))
        rows = []
        for case in refs:
            ref = refs[case]
            n0 = float(np.linalg.norm(ref[0]))
            bank_err, man_err, iters = [], [], []
            for ti in range(ref.shape[0]):
                target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
                bank_err.append(float(jnp.linalg.norm(Qb @ (Qb.T @ target) - target)) / n0)
                score = Hn - 2 * (Hc @ (G.T @ target))
                starts = jnp.asarray(Zaug)[jnp.argsort(score)[:cfg['recon_starts']]]
                z, rn, it, reason = host(recon(starts, target, G))
                man_err.append(float(rn) / n0)
                iters.append(int(it))
            rows.append(dict(case=case, bank_projection_per_time=bank_err,
                             bank_projection_max=float(np.max(bank_err)),
                             best_found_per_time=man_err, best_found_max=float(np.max(man_err)),
                             reconstruction_iterations=iters))
        entry = dict(q=q, solved_dimension=K + q, cases=rows,
                     worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
                     worst_best_found=float(max(r['best_found_max'] for r in rows)))
        recon_done[q] = entry
        report['reconstruction'].append(entry)
        print('RECON q', q, round(entry['worst_best_found'] * 100, 5), flush=True)
        save()

    # ---------------------------------------------------------- timed queries --
    subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
    subjects += [dict(kind='fom', name=fs['name'], setting=fs) for fs in cfg['fom_settings']]
    inputs = [e.initial(L, phys) for phys in physical]
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, u, case):
        nu = float(physical[case, 4])
        if sub['kind'] == 'rom':
            b = built[sub['index']]
            return b['query'](u, nu, b['data'], b['cold'])
        fs = sub['setting']
        fn, pre = foms[fs['preconditioner']]
        return fn(u, nu, fs['ntol'], fs['ltol'], *pre)

    t = time.perf_counter()
    for sub in subjects:
        jax.block_until_ready(invoke(sub, jnp.asarray(inputs[0]), 0))
        print('WARM', sub['name'], round(time.perf_counter() - t, 1), flush=True)
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
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
                f = np.asarray(value[0])
                hs = time.perf_counter() - ht
                v = host(value)
                assert np.isfinite(f).all()
                h = sha_array(f)
                key = (sub['name'], case, h)
                if key not in artifacts:
                    fn = f"L{L}_{sub['name']}_case{case}_rep{rep}.npz"
                    extra = dict(internal_latents=v[7]) if sub['kind'] == 'rom' else {}
                    np.savez_compressed(out / fn, fields=f, **extra)
                    artifacts[key] = fn
                row = dict(intervals=L, case=case, cohort=report['cohort_roles'][case], rep=rep,
                           kind=sub['kind'], name=sub['name'], gpu_seconds=gs, host_seconds=hs,
                           output_bytes=int(f.nbytes), field_sha256=h, artifact=artifacts[key],
                           error=e.errors(f, refs[case], L), iterations=v[1].tolist(),
                           residuals=v[2].tolist(), finite=True)
                if sub['kind'] == 'rom':
                    b = built[sub['index']]
                    reasons = v[3].tolist()
                    gj = np.asarray(v[12], dtype=float)
                    worst = max(float(np.max(gj)), float(v[13]))
                    row.update(q=b['q'], solved_dimension=K + b['q'], M=b['M'], m=b['m'],
                               rule=b['rule'], variant=b['variant'], fitter=b['fitter'],
                               quadrature=b['quadrature'], dt=dt, linear_solve=b['linear_solve'],
                               stop_reasons=reasons, ic_iterations=int(v[5]), ic_reason=int(v[6]),
                               step_stationarity=v[8].tolist(), ic_stationarity=float(v[9]),
                               step_joint_stationarity=gj.tolist(), ic_joint_stationarity=float(v[13]),
                               step_inner_stationarity=np.asarray(v[14], dtype=float).tolist(),
                               ic_residual=float(v[10]), ic_input_norm=float(v[11]),
                               ic_relative_residual=float(v[10]) / max(float(v[11]), 1e-300),
                               budget_exits=int(sum(1 for r in reasons if r == 0)),
                               rejected_exits=int(sum(1 for r in reasons if r == 3)),
                               worst_joint_stationarity=worst,
                               stationary=bool(worst <= strict['gtol'] * (1 + 1e-7)),
                               completed=bool(all(r in (1, 2, 4) for r in reasons)
                                              and int(v[6]) in (1, 2, 4)),
                               converged=bool(worst <= strict['gtol'] * (1 + 1e-7)
                                              and all(r in (1, 2, 4) for r in reasons)
                                              and int(v[6]) in (1, 2, 4)
                                              and sum(1 for r in reasons if r == 0) == 0))
                else:
                    row.update(nonlinear_converged=bool(np.max(v[2]) <= sub['setting']['ntol'] * (1 + 1e-9)))
                report['invocations'].append(row)
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    # --------------------------------------------------------------- in-job gates
    def fields_by(name):
        return {x['case']: x['field_sha256'] for x in report['invocations'] if x['name'] == name}

    base = fields_by('q0_m4_dense_joint')
    same = {}
    for nm in ('q0_m4_dense_varpro', 'q0_m4_dense_block', 'q0_m4_dense_alt'):
        got = fields_by(nm)
        same[nm] = dict(compared=len(set(base) & set(got)),
                        identical=sum(1 for c in set(base) & set(got) if base[c] == got[c]))
    report['gates']['q0_variants_bitwise'] = dict(
        reference='q0_m4_dense_joint', arms=same,
        passed=all(v['compared'] > 0 and v['identical'] == v['compared'] for v in same.values()))
    save()

    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['checkpoint_sha256'] == report['checkpoint_sha256_after']
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('CHEAP LADDER COMPLETE', flush=True)


if __name__ == '__main__':
    main()
