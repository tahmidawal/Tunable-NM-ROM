"""Evaluate every frozen checkpoint through the UNCHANGED head-ablation arm (a)
machinery, with the incumbent re-run in the same job as the control.

Nothing here is a re-implementation: the banks, the reduced operators, the
empirical-quadrature rule, the cold initializer, the weak residual, the
Levenberg-Marquardt solver and the complete query all come from
`experiments/head-ablation/arms.py` exactly as `ablation.py` uses them for arm
(a). The only thing that differs between arms is which checkpoint supplies the
bank, the head and the training codes.

Three layers are reported per checkpoint: the bank projection floor, the
best-found reconstruction on that checkpoint's own manifold, and the solved
error of the real online query -- the last split into all output times and the
evolved times, because the incumbent's worst-over-all-times number is pinned by
its own t = 0 compression of the supplied field.
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


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False, default=float) + '\n')


def radius(Z):
    Z = np.asarray(Z)
    return .01 * float(np.max(np.linalg.norm(Z - Z.mean(0), axis=1)))


def same_grid(field, base, n0):
    f, g = np.asarray(field), np.asarray(base)
    d = np.linalg.norm((f - g).reshape(len(f), -1), axis=1) / n0
    return dict(per_time=d.tolist(), max=float(np.max(d)),
                evolved_max=float(np.max(d[1:])), t0=float(d[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoints', required=True, help='directory of ckpt_*.pkl')
    p.add_argument('--incumbent', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    L, dt = cfg['intervals'], cfg['dt']
    strict = cfg['strict']
    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
                  gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                  jax_version=jax.__version__, intervals=L, dt=dt,
                  output_times=[0, .05, .1, .15, .2, .25],
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output '
                                   'fields; same-invocation host transfers also measured; '
                                   'identical for every arm'),
                  spatial_bank_frozen=True, network_weights_frozen=True,
                  final_cohort_unopened=True, checkpoints=[], reference=[], arm_setup=[],
                  reconstruction=[], invocations=[], declared_subjects=[],
                  verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    # ------------------------------------------------------------ the arms ---
    entries = [dict(name='incumbent', path=Path(a.incumbent))]
    for pth in sorted(Path(a.checkpoints).glob('ckpt_*.pkl')):
        entries.append(dict(name=pth.stem[len('ckpt_'):], path=pth))
    if cfg.get('arm_filter'):
        entries = [x for x in entries if x['name'] == 'incumbent' or x['name'] in cfg['arm_filter']]
    for x in entries:
        ck = pickle.load(open(x['path'], 'rb'))
        x['params'] = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
        x['Z'] = np.asarray(ck['Z_tr'])
        x['K'] = int(x['Z'].shape[1])
        x['R'] = int(np.asarray(ck['params']['h_lin']).shape[1])
        x['sha256'] = sha_file(x['path'])
        x['ckpt_cfg'] = {k: v for k, v in ck.get('cfg', {}).items() if k != 'hfit_pick'}
        report['checkpoints'].append(dict(arm=x['name'], file=x['path'].name, K=x['K'], R=x['R'],
                                          sha256=x['sha256'], codes=int(len(x['Z'])),
                                          cfg=x['ckpt_cfg']))
    save()
    print('CHECKPOINTS', [(x['name'], x['K'], x['R']) for x in entries], flush=True)

    # ------------------------------------------------------------ cohort -----
    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    # The six development cases must be the ones abl01 solved, or the incumbent is not a control.
    # Compared as values, bitwise, against abl01's own recorded cases -- not by re-deriving a seed,
    # because a NumPy upgrade can move `np.exp` by one unit in the last place (DESIGN.md A5).
    if cfg.get('abl01_result') and Path(cfg['abl01_result']).exists():
        ab = np.asarray(json.loads(Path(cfg['abl01_result']).read_text())['physical_cases'])
        same = ab.shape == physical.shape and bool((ab == physical).all())
        rel = (float(np.max(np.abs(ab - physical) / np.maximum(np.abs(ab), 1e-300)))
               if ab.shape == physical.shape else None)
        report['cohort_matches_abl01'] = dict(bitwise=same, max_relative=rel,
                                              n=int(len(physical)))
        print('GATE cohort_matches_abl01', report['cohort_matches_abl01'], flush=True)
        assert same, ('the evaluation cohort is not abl01\'s', rel)
    save()

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
                                        max_relative_residual=float(np.max(rn)),
                                        downsampled_to=L, field_sha256=sha_array(f),
                                        seconds=time.perf_counter() - t))
        print('REFERENCE', case, round(time.perf_counter() - begin, 1), flush=True)
        save()
    del q
    jax.clear_caches()
    truth = {c: refs[c] for c in refs}

    # ---------------------------------------------- same-grid reference ------
    foms = {'fft': ip.make_fom(L, dt, 'fft')}
    tight = next(x for x in cfg['fom_settings'] if x['name'] == cfg['same_grid_reference'])
    fn, pre = foms[tight['preconditioner']]
    base = {}
    for case, phys in enumerate(physical):
        v = host(fn(jnp.asarray(e.initial(L, phys)), float(phys[4]), tight['ntol'],
                    tight['ltol'], *pre))
        assert np.isfinite(v[0]).all() and np.max(v[2]) <= tight['ntol'] * (1 + 1e-9)
        base[case] = np.asarray(v[0])
        np.savez_compressed(out / f'samegrid_case{case}.npz', fields=base[case])
    report['same_grid_reference'] = dict(
        name=tight['name'], note='untimed converged full-order solve on the evaluation mesh; '
                                 'every same-grid number below is a discrepancy against it',
        field_sha256={str(c): sha_array(base[c]) for c in base})
    save()
    print('SAME-GRID REFERENCE', round(time.perf_counter() - begin, 1), flush=True)

    # ------------------------------------------------------------ build -----
    order_rng = np.random.default_rng(cfg['order_seed'])
    built = []
    for x in entries:
        K, R = x['K'], x['R']
        bank = A.CoordBank(x['params'], K, R)
        head = A.neural_head(x['params'])
        stride = max(1, len(x['Z']) // cfg['decoder_code_subsample'])
        Zsub = np.asarray(x['Z'][::stride])
        trust = radius(x['Z'])
        for quadrature in (cfg['quadratures'].get(x['name']) or cfg['default_quadratures']):
            M = cfg['test_multiplier'] * K
            m = cfg['quadrature_multiplier'] * M if quadrature == 'eq' else None
            t0 = time.perf_counter()
            data, info = A.build_operators(bank, L, M, quadrature, Zcoef=x['Z'], m=m,
                                           eq_seed=cfg['eq_seed'], candidate_cap=cfg['candidate_cap'],
                                           fit_states=cfg['fit_states'], head=head)
            cold, cinfo = A.build_cold(bank, head, Zsub, cfg['cold_axis_points'])
            name = f"{x['name']}_{quadrature}"
            info.update(arm=name, checkpoint=x['name'], K=K, R=R, trust_radius=trust,
                        linear_solve='gj', cold=cinfo, code_stride=int(stride),
                        candidates=int(len(Zsub)), checkpoint_sha256=x['sha256'],
                        total_setup_seconds=time.perf_counter() - t0)
            report['arm_setup'].append({k: v for k, v in info.items()
                                        if k not in ('eq_indices', 'eq_weights', 'eq_fit_rows')})
            query = A.make_query(head, K, L, dt, trust, quadrature, linear='gj', **strict)
            built.append(dict(name=name, checkpoint=x['name'], entry=x, bank=bank, head=head,
                              data=data, cold=cold, query=query, K=K, R=R, M=M, m=m,
                              quadrature=quadrature, trust=trust, Zsub=Zsub))
            report['declared_subjects'].append(dict(name=name, method='rom', K=K, R=R, M=M, m=m,
                                                    quadrature=quadrature, dt=dt))
            print('ARM', name, round(info['total_setup_seconds'], 1), flush=True)
            save()
    for fs in cfg['fom_settings']:
        report['declared_subjects'].append(dict(method='fom', dt=dt, **fs))

    # ------------------------------------------------- untimed diagnostics ---
    whitened = {}
    for b in built:
        # one manifold per checkpoint: the eq and dense arms of one checkpoint share
        # the same head and bank, so the representation layers are computed once
        if any(r['checkpoint'] == b['checkpoint'] for r in report['reconstruction']):
            continue
        B = b['data']['G']
        if id(B) not in whitened:
            whitened[id(B)] = A.whiten(B)
        Bq, Br = whitened[id(B)]
        Hc = jax.jit(jax.vmap(b['head']))(jnp.asarray(b['Zsub']))
        Hrot = Hc @ Br.T
        Hn = jnp.sum(Hrot * Hrot, 1)
        recon = A.make_reconstruction(b['head'], b['K'], L, cfg['recon_budget'], linear='gj')
        rows = []
        for case in refs:
            ref = truth[case]
            n0 = float(np.linalg.norm(ref[0]))
            bank_err, man_err, iters = [], [], []
            for ti in range(ref.shape[0]):
                target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
                proj = Bq @ (Bq.T @ target)
                bank_err.append(float(jnp.linalg.norm(proj - target)) / n0)
                bt = B.T @ target
                starts = jnp.asarray(b['Zsub'])[jnp.argsort(Hn - 2 * (Hc @ bt))[:cfg['recon_starts']]]
                z, rn, it, reason = host(recon(starts, target, B))
                man_err.append(float(rn) / n0)
                iters.append(int(it))
            rows.append(dict(case=case, bank_projection_per_time=bank_err,
                             bank_projection_max=float(np.max(bank_err)),
                             best_found_per_time=man_err, best_found_max=float(np.max(man_err)),
                             best_found_t0=float(man_err[0]),
                             best_found_evolved_max=float(np.max(man_err[1:])),
                             reconstruction_iterations=iters))
        report['reconstruction'].append(dict(
            arm=b['name'], checkpoint=b['checkpoint'], K=b['K'], R=b['R'], cases=rows,
            worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
            worst_best_found=float(max(r['best_found_max'] for r in rows)),
            worst_best_found_t0=float(max(r['best_found_t0'] for r in rows)),
            worst_best_found_evolved=float(max(r['best_found_evolved_max'] for r in rows))))
        print('RECON', b['name'], round(max(r['best_found_max'] for r in rows) * 100, 5),
              round(time.perf_counter() - begin, 1), flush=True)
        save()

    # ------------------------------------------------------- timed queries ---
    subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
    subjects += [dict(kind='fom', name=fs['name'], setting=fs) for fs in cfg['fom_settings']]
    inputs = [e.initial(L, phys) for phys in physical]

    def invoke(sub, u, case):
        nu = float(physical[case, 4])
        if sub['kind'] == 'rom':
            b = built[sub['index']]
            return b['query'](u, nu, b['data'], b['cold'])
        fs = sub['setting']
        f_, pre_ = foms[fs['preconditioner']]
        return f_(u, nu, fs['ntol'], fs['ltol'], *pre_)

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
                f = np.asarray(value[0])
                hs = time.perf_counter() - ht
                v = host(value)
                assert np.isfinite(f).all()
                h = sha_array(f)
                key = (sub['name'], case, h)
                if key not in artifacts:
                    fnm = f"{sub['name']}_case{case}_rep{rep}.npz"
                    extra = dict(internal_latents=v[7]) if sub['kind'] == 'rom' else {}
                    np.savez_compressed(out / fnm, fields=f, **extra)
                    artifacts[key] = fnm
                n0 = float(np.linalg.norm(truth[case][0]))
                row = dict(case=case, cohort=report['cohort_roles'][case], rep=rep,
                           kind=sub['kind'], name=sub['name'], gpu_seconds=gs, host_seconds=hs,
                           output_bytes=int(f.nbytes), field_sha256=h, artifact=artifacts[key],
                           error=e.errors(f, truth[case], L),
                           same_grid=same_grid(f, base[case], n0),
                           iterations=v[1].tolist(), residuals=v[2].tolist(), finite=True)
                if sub['kind'] == 'rom':
                    b = built[sub['index']]
                    reasons = v[3].tolist()
                    row.update(checkpoint=b['checkpoint'], K=b['K'], R=b['R'], M=b['M'], m=b['m'],
                               quadrature=b['quadrature'], linear_solve='gj', stop_reasons=reasons,
                               ic_iterations=int(v[5]), ic_reason=int(v[6]),
                               step_stationarity=v[8].tolist(), ic_stationarity=float(v[9]),
                               ic_residual=float(v[10]), ic_input_norm=float(v[11]),
                               ic_relative_residual=float(v[10]) / max(float(v[11]), 1e-300),
                               budget_exits=int(sum(1 for r in reasons if r == 0)),
                               rejected_exits=int(sum(1 for r in reasons if r == 3)),
                               ic_budget_exit=bool(int(v[6]) in (0, 3)),
                               stationary=bool(max(float(np.max(v[8])), float(v[9]))
                                               <= strict['gtol'] * (1 + 1e-7)),
                               completed=bool(all(r in (1, 2, 4) for r in reasons)
                                              and int(v[6]) in (1, 2, 4)))
                else:
                    row.update(nonlinear_converged=bool(np.max(v[2])
                                                        <= sub['setting']['ntol'] * (1 + 1e-9)))
                report['invocations'].append(row)
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    # ------------------------------------------- in-job fidelity gate --------
    gate = None
    if cfg.get('abl01_result') and Path(cfg['abl01_result']).exists():
        prev = json.loads(Path(cfg['abl01_result']).read_text())
        want = {r['case']: r['error']['fixed_initial_max'] for r in prev['invocations']
                if r['intervals'] == L and r['name'] == cfg['abl01_arm'] and r['rep'] == 0}
        got = {r['case']: r['error']['fixed_initial_max'] for r in report['invocations']
               if r['name'] == cfg['incumbent_arm'] and r['rep'] == 0}
        common = sorted(set(want) & set(got))
        worst = max((abs(got[c] - want[c]) / max(want[c], 1e-300) for c in common), default=1.)
        gate = dict(compared=len(common), worst_relative_delta=float(worst), tolerance=1e-9,
                    passed=bool(common and worst <= 1e-9), reference=cfg['abl01_arm'])
        report['gates'] = dict(incumbent_reproduces_abl01=gate)
        print('GATE incumbent_reproduces_abl01', gate, flush=True)
    save()

    report['elapsed_seconds'] = time.perf_counter() - begin
    report['checkpoint_sha256_after'] = {x['name']: sha_file(x['path']) for x in entries}
    assert all(report['checkpoint_sha256_after'][x['name']] == x['sha256'] for x in entries)
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('EVALUATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
