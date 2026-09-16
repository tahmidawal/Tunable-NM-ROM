"""Correction ladder on Poisson 2D at one mesh, from one frozen checkpoint.

The Poisson weak residual is exactly r(z) = B h(z) - f_m, LINEAR in the coefficients,
so the retained `correction_core` path already eliminates the correction coefficients
analytically: it is variable projection with a one-step exact inner solve, and the
nonlinear iteration stays K-dimensional at every q. That path is extended here from the
retained q = 32 to q in {0, 8, 16, 32, 64, 128}, and the test count is varied, on the
same checkpoint, the same twelve opened development sources and the same mesh that
`pabl01` used, with the direct DST full-order solve interleaved.

There is no quadrature on Poisson, so the ladder's third change does not apply.

Directions. `basis.npz` carries only 32 nested directions. Columns 1..32 are used
verbatim, so q <= 32 reproduces the retained arm exactly. Columns 33..R extend them by
the same rule — right singular vectors of the head residual in the bank's QR physical
metric — applied to the component orthogonal to the retained 32, from a regenerated
training set with best-found head codes. At q = R the corrections span the whole bank,
the eliminated operator is numerically zero and the rung is the degenerate bank-floor
endpoint, reported as such.

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
import correction_core as CC
import pilot as P
import arms as A
import sep_common as sc
import directions as DIR


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def chead(params, Cq, K):
    return lambda w: sc.head(params, w[:K]) + Cq @ w[K:]


def reassemble(base, modes):
    """`core.assemble` for a different test count, reusing the built bank.

    Only the sine table, the retained mode set and the reduced operator depend on the
    test count; the bank is over a gigabyte at 1024 intervals and is built once.
    """
    t0 = time.perf_counter()
    n = base['intervals']
    lam = C.eigenvalues(n)
    I, J = np.nonzero(lam <= np.sort(lam.ravel())[modes - 1])
    maxmode = int(max(I.max(), J.max())) + 1
    p = np.arange(1, n)
    S = np.sqrt(2.0 / n) * np.sin(np.pi * np.outer(p, np.arange(1, maxmode + 1)) / n)
    Sj, Ij, Jj = jnp.asarray(S), jnp.asarray(I), jnp.asarray(J)
    W = jnp.asarray(lam[I, J] ** -1)
    bank = base['bank']
    cubes = bank.reshape((n - 1, n - 1, bank.shape[-1]))
    B = jnp.einsum('xa,xyr,yb->abr', Sj, cubes, Sj)[Ij, Jj]
    B.block_until_ready()
    info = {**base['info'], 'requested_modes': int(modes), 'retained_modes': int(len(I)),
            'operator_sha256': C.sha(np.asarray(B)),
            'reassembly_seconds': time.perf_counter() - t0}
    return {**base, 'B': B, 'S': Sj, 'I': Ij, 'J': Jj, 'W': W, 'info': info}


def test_count(rule, K, q, fixed):
    if rule == 'm4':
        return 4 * (K + q)
    if rule == 'm2':
        return 2 * (K + q)
    if rule == 'm256':
        return int(fixed)
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--basis', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
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

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
                  checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                  basis_sha256=hashlib.sha256(Path(a.basis).read_bytes()).hexdigest(), K=K, R=R,
                  intervals=n, spatial_bank_frozen=True, network_weights_frozen=True,
                  final_cohort_unopened=True,
                  timing_contract=('host source array in to dense nodal field out, including projection, '
                                   'initialization, nonlinear solve, exact correction recovery and decode; '
                                   'identical for every arm'),
                  references=[], snapshots={}, directions={}, arm_setup=[], reconstruction=[],
                  invocations=[], declared_subjects=[], gates={}, complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    draws = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                            C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    report['cohort'] = dict(parameters=draws.tolist(), sha256=sha_array(draws),
                            groups=['existing development'] * cfg['eval_count']
                                   + ['fresh development'] * cfg['fresh_count'])
    train_draws = C.source_params(cfg['train_seed'], cfg['train_count'])
    assert not any(np.allclose(t, s) for t in train_draws for s in draws), 'training/eval overlap'
    report['train_parameters_sha256'] = sha_array(train_draws)
    save()

    nhi = cfg['reference_intervals'][-1]
    refs = {}
    for case, param in enumerate(draws):
        t0 = time.perf_counter()
        fine = P.reference(param, nhi)
        refs[case] = np.array(fine[::nhi // n, ::nhi // n], copy=True)
        name = f'ref_n{n}_case{case}.npz'
        np.savez_compressed(out / name, field=refs[case])
        report['references'].append(dict(case=case, fine_intervals=nhi, restricted_to=n,
                                         artifact=name, fine_sha256=sha_array(fine),
                                         restricted_sha256=sha_array(refs[case]),
                                         seconds=time.perf_counter() - t0))
    save()
    print('REFERENCES', round(time.perf_counter() - begin, 1), flush=True)

    lam = jnp.asarray(C.eigenvalues(n))
    base = C.assemble(params, np.asarray(codes), n, cfg['fixed_test_count'], cfg['lm_budget'])
    report['arm_setup'].append(dict(intervals=n, arm='shared_assembly', **{
        k: base['info'][k] for k in ('retained_modes', 'stored_features', 'retained_bank_rank',
                                     'trust_delta', 'operator_sha256', 'bank_sha256')}))
    save()
    print('ASSEMBLY', round(time.perf_counter() - begin, 1), flush=True)

    # ------------------------------------------------------------ directions ---
    t0 = time.perf_counter()
    bank = base['bank']
    Qb, Rb = A.whiten(bank)
    jax.block_until_ready(Rb)
    U = np.stack([np.asarray(C.dst_solve(jnp.asarray(C.full_source(n, qv)), lam))[1:-1, 1:-1].ravel()
                  for qv in train_draws])
    Ut = jnp.asarray(U.T)
    coef_truth = jnp.linalg.solve(Rb, Qb.T @ Ut).T
    bank_proj = float(jnp.linalg.norm(Ut - Qb @ (Qb.T @ Ut)) / jnp.linalg.norm(Ut))
    del Ut
    report['snapshots'][str(n)] = dict(sources=len(train_draws), snapshot_sha256=sha_array(U),
                                       bank_projection_relative_rms=bank_proj,
                                       seconds=time.perf_counter() - t0)
    del U
    save()

    dcfg = {**cfg, 'strict': dict(gtol=cfg['stationarity_tolerance'])}
    _, _, Zstar, rho, dinfo = DIR.audited(params, Rb, coef_truth, np.asarray(codes), K, dcfg)
    D32 = jnp.asarray(basis['coefficient_directions'])
    kept = int(D32.shape[1])
    T32, _ = jnp.linalg.qr(Rb @ D32, mode='reduced')
    tilde = rho @ Rb.T
    perp = tilde - (tilde @ T32) @ T32.T
    _, sv, Vt = jnp.linalg.svd(perp, full_matrices=False)
    Text = Vt[:R - kept].T
    Cfull = jnp.concatenate((D32, jnp.linalg.solve(Rb, Text)), axis=1)
    prefix = float(jnp.max(jnp.abs(Cfull[:, :kept] - D32)))
    dinfo.update(retained_prefix_columns=kept, retained_prefix_max_abs_difference=prefix,
                 extension_rule=('right singular vectors of the head residual in the bank QR metric, '
                                 'orthogonal to the retained 32; nested'),
                 extension_singular_values=np.asarray(sv[:min(len(sv), 200)]).tolist(),
                 full_directions_sha256=sha_array(np.asarray(Cfull)),
                 basis_directions_sha256=sha_array(np.asarray(D32)),
                 seconds=time.perf_counter() - t0)
    report['directions'] = dinfo
    report['gates']['retained_prefix_exact'] = dict(max_abs_difference=prefix, passed=prefix == 0.0)
    save()
    print('DIRECTIONS', round(dinfo['seconds'], 1), 'prefix delta', prefix, flush=True)

    # -------------------------------------------------------------- subjects ---
    specs = []
    for rule in cfg['rules']:
        for q in cfg['q_ladder']:
            modes = test_count(rule, K, q, cfg['fixed_test_count'])
            specs.append(dict(name=f'q{q}_{rule}', q=q, rule=rule, requested_modes=modes))
    opcache = {cfg['fixed_test_count']: base}
    built = []
    for s in specs:
        modes = s['requested_modes']
        if modes not in opcache:
            opcache[modes] = reassemble(base, modes)
        ops = opcache[modes]
        M = int(ops['B'].shape[0])
        if M <= K + s['q']:
            print('SKIP', s['name'], 'M', M, 'dim', K + s['q'], flush=True)
            continue
        t0 = time.perf_counter()
        engine = CC.prepare_correction(ops, np.asarray(codes), np.asarray(Cfull), s['q'], cfg['retained'])
        info = dict(intervals=n, arm=s['name'], q=s['q'], rule=s['rule'], M=M,
                    requested_modes=modes, nonlinear_dimension=K,
                    nominal_dimension=K + s['q'], setup_seconds=time.perf_counter() - t0,
                    degenerate_full_bank=bool(s['q'] >= R), **engine['info'])
        report['arm_setup'].append(info)
        report['declared_subjects'].append(dict(name=s['name'], method='rom', q=s['q'], rule=s['rule'],
                                                M=M, requested_modes=modes))
        built.append(dict(**s, ops=ops, engine=engine, M=M))
        print('ARM', s['name'], 'M', M, round(info['setup_seconds'], 1), flush=True)
        save()
    report['declared_subjects'].append(dict(name='dst_direct', method='fom'))

    # ------------------------------------------------ untimed diagnostics ------
    done = set()
    for b in built:
        q = b['q']
        if q in done:
            continue
        done.add(q)
        Cq = Cfull[:, :q]
        head = chead(params, Cq, K)
        Zaug = np.concatenate((np.asarray(codes), np.zeros((len(codes), q))), axis=1)
        Hc = jax.jit(jax.vmap(head))(jnp.asarray(Zaug))
        Hn = jnp.sum((Hc @ Rb.T) ** 2, 1)
        recon = A.make_reconstruction(head, K + q, n, cfg['recon_budget'],
                                      linear=('gj' if K + q <= cfg['gauss_jordan_max'] else 'lu'))
        rows = []
        for case in refs:
            target = jnp.asarray(refs[case][1:-1, 1:-1].ravel())
            nref = float(np.linalg.norm(refs[case]))
            bank_err = float(jnp.linalg.norm(Qb @ (Qb.T @ target) - target)) / nref
            score = Hn - 2 * (Hc @ (bank.T @ target))
            starts = jnp.asarray(Zaug)[jnp.argsort(score)[:cfg['recon_starts']]]
            z, rn, it, reason = jax.device_get(recon(starts, target, bank))
            rows.append(dict(case=case, bank_projection=bank_err, best_found=float(rn) / nref,
                             reconstruction_iterations=int(it)))
        report['reconstruction'].append(dict(intervals=n, q=q, cases=rows,
            worst_bank_projection=float(max(r['bank_projection'] for r in rows)),
            worst_best_found=float(max(r['best_found'] for r in rows))))
        print('RECON q', q, flush=True)
        save()

    # ---------------------------------------------------------- timed queries --
    subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
    subjects.append(dict(kind='fom', name='dst_direct'))
    sources = [C.full_source(n, qv) for qv in draws]
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, case):
        if sub['kind'] == 'rom':
            b = built[sub['index']]
            return CC.correction_query(sources[case], b['ops'], b['engine'], cfg['retained'])
        return C.fom_query(sources[case], lam)

    t = time.perf_counter()
    for sub in subjects:
        invoke(sub, 0)
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
    save()
    print('WARMUP', round(time.perf_counter() - t, 1), flush=True)

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
                report['invocations'].append(dict(
                    intervals=n, case=case, group=report['cohort']['groups'][case], rep=rep,
                    kind=sub['kind'], name=sub['name'], field_sha256=h, artifact=artifacts[key],
                    physical_error=C.relative(field, refs[case]),
                    source_sha256=C.sha(sources[case]), finite=True,
                    q=(b['q'] if b else None), rule=(b['rule'] if b else None),
                    M=(b['M'] if b else None), **row))
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('POISSON LADDER COMPLETE', flush=True)


if __name__ == '__main__':
    main()
