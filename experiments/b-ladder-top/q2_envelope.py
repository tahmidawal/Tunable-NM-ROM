"""Q2: the combined envelope for one frozen Burgers checkpoint, in one job.

One allocation, one GPU, one randomised order, three timed repetitions with burn-in:

  * the correction ladder q in {0, 16, 32, 64, 128} (+ 256 when Q1 converged it), each
    with its own empirical quadrature rule, crossed with the evolution tolerance
    {1e-3, 1e-6}, plus a dense control at q = 0;
  * same-job full-order controls: `fft_tight`, `nt1e-4_dt005`, `nt1e-2_dt01`;
  * POD-LSPG at k' in {16, 32, 64, 128}, head-ablation arm (e) verbatim;
  * the model-facing cohort for the trained FNO, written here so a second process in
    this same allocation can time it on the same GPU against the same references.

Every reported ROM/FOM error is measured twice by the audit: worst over ALL output
times (pinned at t = 0 by the decoder's compression of the supplied field) and worst
over EVOLVED times only, with the t = 0 compression as its own column.
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
import ladder as LD
import directions as DIR
import topfix as TF
from ablation import pod_basis, radius

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump
TIMES = np.array([0., .05, .1, .15, .2, .25], dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fno-train-index', default=None)
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
                  output_times=TIMES.tolist(),
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output fields; '
                                   'same-invocation host transfers also measured; identical for every '
                                   'JAX subject; the FNO phase replicates the neural-operator lane '
                                   'timing.py protocol in a second process in this same allocation'),
                  reference=[], snapshots={}, directions={}, arm_setup=[], reconstruction=[],
                  invocations=[], declared_subjects=[], gates={}, fno=dict(prepared=False),
                  verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
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
                                        field_sha256=got, cclad01_field_sha256=want,
                                        bitwise_matches_cclad01=(None if want is None else got == want),
                                        seconds=time.perf_counter() - t))
        print('REFERENCE', case, round(time.perf_counter() - begin, 1), flush=True)
        save()
    del q_fom
    jax.clear_caches()

    # ------------------------------------------------- the model-facing cohort --
    cohort = out / 'fno-cohort'
    cohort.mkdir(exist_ok=True)
    records = []
    for case, phys in enumerate(physical):
        target = refs[case][:, None].astype(np.float64)
        supplied = np.ascontiguousarray(target[0])
        path = cohort / f'burgers-dev-{case:05d}.npz'
        np.savez(path, input=supplied, target=target,
                 parameters=np.array([float(phys[4])], dtype=np.float64), times=TIMES)
        records.append(dict(case_id=f'burgers-dev-{case:05d}', split='development', case_index=case,
                            seed=int(cfg['eval_seed'] if case < cfg['eval_cases'] else cfg['eval_fresh_seed']),
                            path=path.name, mesh=L,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            generation_descriptors=dict(zip(('cx', 'cy', 'width', 'amplitude', 'nu'),
                                                            map(float, phys)))))
    (cohort / 'index.json').write_text(json.dumps(dict(
        schema_version=1, pde='burgers', split='development', count=len(records), mesh=L,
        complete=True, descriptors_are_model_inputs=False,
        derivation=('the same six opened development cases and the same 4096-interval reference '
                    'restricted to the 256-interval grid that every ROM subject in this job is '
                    'graded against'), records=records), indent=2) + '\n')
    report['fno'] = dict(prepared=True, cohort_index=str(cohort / 'index.json'), cases=len(records))
    save()

    # disjointness of the six cases from the FNO training set
    if a.fno_train_index and Path(a.fno_train_index).exists():
        tr = json.loads(Path(a.fno_train_index).read_text())
        desc = np.array([[r['generation_descriptors'][k] for k in
                          ('cx', 'cy', 'width', 'amplitude', 'nu')] for r in tr['records']])
        dist = np.min(np.linalg.norm(desc[None] - physical[:, None], axis=2), axis=1)
        report['gates']['fno_cohort_disjoint_from_training'] = dict(
            passed=bool(np.min(dist) > 1e-8), training_cases=int(len(desc)),
            nearest_descriptor_distance=dist.tolist(),
            note=('exact-parameter disjointness of the six development cases from the FNO training '
                  'draw; the two draws come from different generators, so this is a check, not a '
                  'construction'))
        save()

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
    ranks = sorted(set(cfg['pod_ranks']))
    Vmodes, eigen, energy = pod_basis(Ut, max(ranks))
    coords_full = np.asarray(Ut.T @ Vmodes)
    sinfo.update(pod_total_energy=energy, pod_eigenvalues=np.asarray(eigen).tolist(),
                 pod_tail_fraction={str(k): float(max(energy - float(np.sum(eigen[:k])), 0.)
                                                  / max(energy, 1e-300)) for k in ranks})
    del Ut, U
    jax.clear_caches()
    report['snapshots'][str(L)] = sinfo
    save()
    print('SNAPSHOTS', round(sinfo['seconds'], 1), flush=True)

    # ------------------------------------------------------------- directions --
    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    Cfull, Ct, Zstar, rho, dinfo = DIR.audited(params, Rb, coef_truth, Zsub, K,
                                               {**cfg, 'strict': strict})
    want = cfg.get('expected_directions_sha256')
    dinfo['expected_sha256'] = want
    dinfo['bitwise_matches_cclad01'] = (None if want is None else dinfo['directions_sha256'] == want)
    report['directions'] = dinfo
    report['gates']['directions_hash'] = dict(expected=want, got=dinfo['directions_sha256'],
                                              passed=dinfo['bitwise_matches_cclad01'])
    save()
    print('DIRECTIONS', round(dinfo['seconds'], 1), 'rank', dinfo['available_rank'], flush=True)

    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    report['trust_radius'] = trust

    # ---------------------------------------------------------------- subjects -
    built, datas, colds, recon_done = {}, {}, {}, {}
    for spec in cfg['rom_arms']:
        q, M = spec['q'], spec['M']
        assert M > K + q, spec
        gtol = spec['gtol']
        fix = spec.get('fix', 'base')
        name = f"q{q}_M{M}_{spec['quadrature']}_g{gtol:g}".replace('-', 'm').replace('.', 'p')
        C = Cfull[:, :q]
        head = TF.corrected_head(params, C, K)
        if q not in colds:
            Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
            colds[q] = A.build_cold(bank, head, Zaug, cfg['cold_axis_points'])
        cold, cinfo = colds[q]
        key = (q, M, spec['quadrature'])
        t0 = time.perf_counter()
        if key not in datas:
            if spec['quadrature'] == 'eq':
                m = int(min(cfg['quadrature_multiplier'] * M, cfg['quadrature_cap']))
                W = (np.concatenate((Zold, np.zeros((len(Zold), 0))), axis=1) if q == 0
                     else TF.enriched_codes(Zstar, rho, Rb, Ct, q))
            else:
                m, W = None, None
            datas[key] = TF.build_operators(
                bank, L, M, spec['quadrature'], head=head, Wcodes=W, m=m, fitter='bounded',
                eq_seed=cfg['eq_seed'], candidate_cap=cfg['candidate_cap'],
                fit_states=cfg['fit_states'], max_fit_rows=cfg['max_fit_rows'],
                eq_seconds=cfg['eq_seconds'], blocks_wanted=cfg['eq_blocks'])
            if spec['quadrature'] == 'eq':
                print('EQFIT', name, 'm', datas[key][1].get('m'), 'valid',
                      datas[key][1].get('eq_rule_valid'),
                      round(datas[key][1]['eq_fit']['seconds'], 1), flush=True)
        data, info = datas[key]
        info = dict(info)
        dim = K + q
        linear = 'gj' if dim <= cfg['gauss_jordan_max'] else 'lu'
        query = TF.make_query(params, C, K, q, L, dt, trust, spec['quadrature'], fix, Rb=Rb,
                              ic_budget=strict['ic_budget'], step_budget=strict['step_budget'],
                              gtol=gtol, ic_gtol=cfg['ic_gtol'], linear=linear,
                              inner_damping=cfg['inner_damping'], tau_y=cfg['tau_y'])
        info.update(arm=name, family='rom', q=q, rule=f'M{M}', fix=fix, solved_dimension=dim,
                    dt=dt, gtol=gtol, ic_gtol=cfg['ic_gtol'], trust_radius=trust,
                    linear_solve=linear, cold=cinfo, correction_directions_used=q,
                    total_setup_seconds=time.perf_counter() - t0)
        report['arm_setup'].append(info)
        report['declared_subjects'].append(dict(
            name=name, method='rom', family='rom', q=q, M=M, m=info.get('m'),
            quadrature=spec['quadrature'], fix=fix, gtol=gtol, solved_dimension=dim,
            linear_solve=linear, dt=dt))
        built[name] = dict(name=name, q=q, M=M, m=info.get('m'), quadrature=spec['quadrature'],
                           fix=fix, gtol=gtol, family='rom', data=data, cold=cold, query=query,
                           head=head, linear_solve=linear, k=dim)
        print('ARM', name, 'M', M, 'm', info.get('m'), round(info['total_setup_seconds'], 1), flush=True)
        save()

    # POD-LSPG: head-ablation arm (e) verbatim
    for k in cfg['pod_ranks']:
        co = coords_full[:, :k]
        gb = A.GridBank(Vmodes[:, :k], L)
        head = A.identity_head()
        M = int(cfg['pod_test_multiplier'] * k)
        name = f'pod{k}_dense'
        t0 = time.perf_counter()
        data, info = A.build_operators(gb, L, M, 'dense')
        cold, cinfo = A.build_cold(gb, head, co, cfg['cold_axis_points'])
        tr = radius(co)
        linear = 'gj' if k <= cfg['gauss_jordan_max'] else 'lu'
        query = A.make_query(head, k, L, dt, tr, 'dense', linear=linear,
                             ic_budget=strict['ic_budget'], step_budget=strict['step_budget'],
                             gtol=cfg['pod_gtol'])
        info = dict(info)
        info.update(arm=name, family='pod', k=k, solved_dimension=k, dt=dt, gtol=cfg['pod_gtol'],
                    trust_radius=tr, linear_solve=linear, cold=cinfo,
                    fit='classical POD of the same truth snapshots, identity head',
                    total_setup_seconds=time.perf_counter() - t0)
        report['arm_setup'].append(info)
        report['declared_subjects'].append(dict(name=name, method='rom', family='pod', k=k, M=M,
                                                quadrature='dense', gtol=cfg['pod_gtol'],
                                                solved_dimension=k, linear_solve=linear, dt=dt))
        built[name] = dict(name=name, q=None, M=M, m=None, quadrature='dense', fix=None,
                           gtol=cfg['pod_gtol'], family='pod', data=data, cold=cold, query=query,
                           head=head, linear_solve=linear, k=k)
        print('ARM', name, round(info['total_setup_seconds'], 1), flush=True)
        save()

    foms = {}
    for fs in cfg['fom_settings']:
        key = (fs['preconditioner'], fs['dt'])
        if key not in foms:
            foms[key] = ip.make_fom(L, fs['dt'], fs['preconditioner'])
        report['declared_subjects'].append(dict(method='fom', **fs))
    save()

    # ------------------------------------------------- untimed diagnostics -----
    for name, b in built.items():
        if b['family'] != 'rom' or b['q'] in recon_done:
            continue
        q = b['q']
        head = b['head']
        Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
        Hc = jax.jit(jax.vmap(head))(jnp.asarray(Zaug))
        Hn = jnp.sum((Hc @ Rb.T) ** 2, 1)
        recon = A.make_reconstruction(head, K + q, L, cfg['recon_budget'],
                                      linear=('gj' if K + q <= cfg['gauss_jordan_max'] else 'lu'))
        rows = []
        for case in refs:
            ref = refs[case]
            n0 = float(np.linalg.norm(ref[0]))
            bank_err, man_err = [], []
            for ti in range(ref.shape[0]):
                target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
                bank_err.append(float(jnp.linalg.norm(Qb @ (Qb.T @ target) - target)) / n0)
                score = Hn - 2 * (Hc @ (G.T @ target))
                starts = jnp.asarray(Zaug)[jnp.argsort(score)[:cfg['recon_starts']]]
                z, rn, it, reason = host(recon(starts, target, G))
                man_err.append(float(rn) / n0)
            rows.append(dict(case=case, bank_projection_per_time=bank_err,
                             bank_projection_max=float(np.max(bank_err)),
                             best_found_per_time=man_err, best_found_max=float(np.max(man_err))))
        entry = dict(family='rom', q=q, solved_dimension=K + q, cases=rows,
                     worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
                     worst_best_found=float(max(r['best_found_max'] for r in rows)))
        recon_done[q] = entry
        report['reconstruction'].append(entry)
        print('RECON q', q, round(entry['worst_best_found'] * 100, 5), flush=True)
        save()
    for k in cfg['pod_ranks']:
        B = jnp.asarray(Vmodes[:, :k])
        span, _ = jnp.linalg.qr(B, mode='reduced')
        rows = []
        for case in refs:
            ref = refs[case]
            n0 = float(np.linalg.norm(ref[0]))
            err = []
            for ti in range(ref.shape[0]):
                target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
                err.append(float(jnp.linalg.norm(span @ (span.T @ target) - target)) / n0)
            rows.append(dict(case=case, best_found_per_time=err, best_found_max=float(np.max(err))))
        report['reconstruction'].append(dict(
            family='pod', k=k, solved_dimension=k, cases=rows,
            worst_best_found=float(max(r['best_found_max'] for r in rows))))
        save()

    # ---------------------------------------------------------- timed queries --
    subjects = [dict(kind='rom', name=n) for n in built]
    subjects += [dict(kind='fom', name=fs['name'], setting=fs) for fs in cfg['fom_settings']]
    inputs = [e.initial(L, phys) for phys in physical]
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, u, case):
        nu = float(physical[case, 4])
        if sub['kind'] == 'rom':
            b = built[sub['name']]
            return b['query'](u, nu, b['data'], b['cold'])
        fs = sub['setting']
        fn, pre = foms[(fs['preconditioner'], fs['dt'])]
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
                    b = built[sub['name']]
                    reasons = v[3].tolist()
                    if b['family'] == 'rom':
                        gj = np.asarray(v[12], dtype=float)
                        worst = max(float(np.max(gj)), float(v[13]))
                        row['step_joint_stationarity'] = gj.tolist()
                        row['ic_joint_stationarity'] = float(v[13])
                    else:
                        worst = max(float(np.max(v[8])), float(v[9]))
                    row.update(family=b['family'], q=b['q'], k=b['k'], solved_dimension=b['k'],
                               M=b['M'], m=b['m'], quadrature=b['quadrature'], fix=b['fix'],
                               gtol=b['gtol'], dt=dt, linear_solve=b['linear_solve'],
                               stop_reasons=reasons, ic_iterations=int(v[5]), ic_reason=int(v[6]),
                               step_stationarity=np.asarray(v[8], dtype=float).tolist(),
                               ic_stationarity=float(v[9]), ic_residual=float(v[10]),
                               ic_input_norm=float(v[11]),
                               budget_exits=int(sum(1 for r in reasons if r == 0)),
                               rejected_exits=int(sum(1 for r in reasons if r == 3)),
                               worst_joint_stationarity=worst,
                               stationary_at_arm_tolerance=bool(worst <= b['gtol'] * (1 + 1e-7)),
                               stationary_1e6=bool(worst <= 1e-6 * (1 + 1e-7)),
                               completed=bool(all(r in (1, 2, 4) for r in reasons)
                                              and int(v[6]) in (1, 2, 4)),
                               converged=bool(worst <= 1e-6 * (1 + 1e-7)
                                              and all(r in (1, 2, 4) for r in reasons)
                                              and int(v[6]) in (1, 2, 4)
                                              and sum(1 for r in reasons if r == 0) == 0))
                else:
                    row.update(family='fom', dt=sub['setting']['dt'],
                               nonlinear_converged=bool(np.max(v[2])
                                                        <= sub['setting']['ntol'] * (1 + 1e-9)))
                report['invocations'].append(row)
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['checkpoint_sha256'] == report['checkpoint_sha256_after']
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('Q2 ENVELOPE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
