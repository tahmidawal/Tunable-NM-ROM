"""Frozen-checkpoint Poisson 2D mesh ladder in one job on one GPU.

The retained ``r128_q32`` model is evaluated unchanged across the ladder: same
weights, same latent dimension, same bank rank, same retained weak-mode count,
same correction count, same solver budget and same stopping rule. Only the mesh
changes. New-grid truth evaluates only.

Three costs are measured separately, each from its own completed device
computation:

* cached reduced solve - the 16-variable nonlinear solve on the preassembled
  reduced operator alone, which carries no mesh dimension,
* complete device query - supplied device source to the full dense device field,
  excluding the stationarity and rank diagnostics the native row also charges,
* offline per-mesh setup - bank evaluation on the new mesh and quadrature-free
  preassembly of the reduced weak operator and its correction projection.

The same-job full-order comparators are the direct DST solver (primary, exact for
this discrete operator) and unpreconditioned CG at stated tolerances (control).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'experiments/multiresolution-poisson'))

import core as pc                                                    # noqa: E402
from correction_core import prepare_correction, correction_query      # noqa: E402
from kernel_solver import make_lm_kernel                              # noqa: E402
from iterative_core import make_cg, verify_cg                         # noqa: E402
from pilot import reference                                           # noqa: E402
import sep_common as sc                                               # noqa: E402
import ladder_common as lc                                            # noqa: E402


def make_correction_parts(ops, engine, cfg, count):
    """The retained fused correction query, expressed as separable stages.

    Every expression below is copied from ``correction_core.prepare_correction``'s
    kernel. The stages are split so that the reduced solve can be timed without
    the mesh-dependent source contraction or dense decode, and so the complete
    device query can be timed without the stationarity and rank diagnostics that
    the native row charges into the same interval. ``complete`` reproduces the
    native field, which the driver asserts.
    """
    B, Bp, Q, R, C = ops['B'], engine['Bp'], engine['Q'], engine['R'], engine['C']
    predictions, cached_codes = engine['cache']['predictions'], engine['cache']['codes']
    params, bank, n = ops['params'], ops['bank'], ops['intervals']
    S, I, J, W = ops['S'], ops['I'], ops['J'], ops['W']
    reduced = {**ops, 'B': Bp, 'info': {**ops['info'], 'operator_sha256': pc.sha(Bp)}}
    solve = make_lm_kernel(reduced, cfg['online_preset']['budget'], True, False,
                           cfg['linear_backward_error_limit'],
                           cfg['online_preset']['stationarity_stop'])
    tau = jnp.asarray(cfg['online_preset']['tau'])

    def contract(source):
        f = ops['project'](source, S, I, J, W)
        fp = f - Q @ (Q.T @ f) if count else f
        squared = jnp.sum((predictions - fp[None, :]) ** 2, axis=1)
        _, indices = jax.lax.top_k(-squared, 1)
        return f, fp, cached_codes[indices[0]], indices[0]

    def recover(f, z, z0):
        h = sc.head(params, z)
        h0 = sc.head(params, z0)
        if count:
            rhs = Q.T @ (f - B @ h)
            y = jax.scipy.linalg.solve_triangular(R, rhs, lower=False)
            initial_y = jax.scipy.linalg.solve_triangular(R, Q.T @ (f - B @ h0), lower=False)
        else:
            y = jnp.zeros((0,))
            initial_y = y
        return h + C @ y, y, initial_y

    def lift(coefficients):
        return jnp.pad((bank @ coefficients).reshape(n - 1, n - 1), 1)

    def complete(source):
        f, fp, z0, index = contract(source)
        answer, stats, _ = solve(z0, fp, tau)
        coefficients, y, initial_y = recover(f, answer[0], z0)
        return lift(coefficients), answer, stats, index, z0, y, initial_y

    return dict(contract=jax.jit(contract),
                reduced_solve=jax.jit(lambda z0, fp: solve(z0, fp, tau)),
                recover=jax.jit(recover), lift=jax.jit(lift), complete=jax.jit(complete))


def measure(compute, host_input):
    """One invocation: host to device, completed device compute, device to host."""
    start = time.perf_counter()
    device_input = jax.device_put(host_input)
    jax.block_until_ready(device_input)
    input_end = time.perf_counter()
    value = jax.block_until_ready(compute(device_input))
    device_end = time.perf_counter()
    result = jax.device_get(value)
    end = time.perf_counter()
    return result, dict(input_transfer_seconds=input_end - start,
                        complete_device_seconds=device_end - input_end,
                        output_transfer_seconds=end - device_end,
                        host_to_host_seconds=end - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'fields').mkdir(exist_ok=True)
    (out / 'references').mkdir(exist_ok=True)
    begin = time.perf_counter()

    environment = lc.preflight()
    checkpoint = ROOT / cfg['checkpoint']
    basis_path = ROOT / cfg['basis']
    checkpoint_sha, basis_sha = lc.sha_file(checkpoint), lc.sha_file(basis_path)
    assert checkpoint_sha == cfg['checkpoint_sha256'], checkpoint_sha
    assert basis_sha == cfg['basis_sha256'], basis_sha
    params, codes, checkpoint_cfg = sc.load_pkl(checkpoint)
    basis = dict(np.load(basis_path))
    directions = basis['coefficient_directions']
    count = cfg['correction_count']
    assert (checkpoint_cfg['k'], checkpoint_cfg['r']) == (16, 128)
    np.testing.assert_array_equal(codes, basis['training_latents'])
    assert directions.shape == (128, 32) and count <= 32
    metric_error = float(np.linalg.norm(
        directions[:, :count].T @ basis['R'].T @ basis['R'] @ directions[:, :count] - np.eye(count)))
    assert metric_error < 1e-10, metric_error

    draws = np.concatenate((
        pc.source_params(cfg['existing_development_seed'], cfg['existing_development_count']),
        pc.source_params(cfg['fresh_development_seed'], cfg['fresh_development_count']),
        pc.source_params(cfg['new_development_seed'], cfg['new_development_count'])))
    groups = (['existing_development'] * (cfg['existing_development_count'] + cfg['fresh_development_count'])
              + ['new_development'] * cfg['new_development_count'])
    opened = np.concatenate((
        pc.source_params(cfg['existing_development_seed'], cfg['existing_development_count']),
        pc.source_params(cfg['fresh_development_seed'], cfg['fresh_development_count'])))
    # The cohort is regenerated from its seeds, never read from an archive. Its
    # width column is exp(uniform(log a, log b)), and that transcendental rounds
    # differently on this project's ARM host than on the x86 cluster, so the
    # recorded digest can miss by one unit in the last place without any sampling
    # difference. The digest is therefore checked first and a bounded deviation
    # from the archived cohort is the only accepted fallback.
    cohort_check = dict(recomputed_sha256=pc.sha(opened), expected_sha256=cfg['parameter_sha256'],
                        digest_matches=pc.sha(opened) == cfg['parameter_sha256'])
    if not cohort_check['digest_matches']:
        archived = np.asarray(json.loads((ROOT / cfg['archived_cohort']).read_text())['cohort']['parameters'])
        regenerated = np.concatenate((opened, pc.source_params(cfg['new_development_seed'],
                                                               cfg['new_development_count'])))
        assert archived.shape == regenerated.shape, (archived.shape, regenerated.shape)
        deviation = np.abs(archived - regenerated) / np.maximum(np.abs(archived), 1e-300)
        cohort_check.update(
            archived_cohort=cfg['archived_cohort'],
            max_relative_deviation_from_archive=float(deviation.max()),
            deviating_columns=[int(j) for j in range(archived.shape[1]) if deviation[:, j].max() > 0],
            note=('digest mismatch accepted only as last-place transcendental rounding; '
                  'the sampled cohort is numerically the archived one'))
        assert cohort_check['max_relative_deviation_from_archive'] <= 4 * np.finfo(float).eps, cohort_check
    ncase = len(draws)

    meshes = list(cfg['meshes'])
    obs = cfg['observation_intervals']
    finest = max(meshes)
    fine_reference, coarse_reference = cfg['reference_intervals']
    assert all(n % obs == 0 for n in meshes)
    assert all(coarse_reference % n == 0 and fine_reference % n == 0 for n in meshes)

    report = dict(
        pde='poisson', config=cfg, **environment,
        checkpoint=cfg['checkpoint'], checkpoint_sha256=checkpoint_sha,
        basis=cfg['basis'], basis_sha256=basis_sha, correction_count=count,
        latent_dimension=int(codes.shape[1]), bank_rank=int(params['h_lin'].shape[1]),
        nominal_latent_dimension=int(codes.shape[1]) + count,
        physical_metric_orthogonality_error=metric_error,
        frozen_weight_transfer=('one checkpoint, one latent dimension, one bank rank, one retained '
                                'weak-mode count, one correction count, one solver budget and one '
                                'stopping rule across every mesh; new-grid truth evaluates only'),
        network_weights_frozen=True, final_cohort_unopened=True,
        cohort=dict(parameters=draws.tolist(), groups=groups, parameter_sha256=pc.sha(draws),
                    opened_cohort_check=cohort_check,
                    scope='already-opened development sources only; final cases remain sealed'),
        cost_contract=dict(
            cached_reduced_solve=('the nonlinear solve on the preassembled reduced operator alone, '
                                  'with the projected source and starting code already resident'),
            complete_device_query=('supplied dense source already on the device to the full dense '
                                   'field still on the device; excludes the stationarity and rank '
                                   'diagnostics the native row charges'),
            offline_setup='bank evaluation on the new mesh and quadrature-free reduced preassembly',
            host_transfers='timed in the same invocation as the complete device query, reported separately',
            rule='each cost is its own completed device computation; no cost is obtained by subtraction'),
        restriction=dict(operator='nested-node injection (stride selection)',
                         common_observation_intervals=obs,
                         reason='every ladder rung and both reference levels share the coarsest grid as nested nodes'),
        verification=verify_cg(),
        references=[], restriction_checks={}, setup=[], invocations=[],
        cached_invocations=[], component_invocations=[], native_rows=[],
        declared_subjects=[], warmup=[], complete=False)
    save = lambda: lc.dump(out / 'result.json', report)
    save()

    # ------------------------------------------------------------- references
    reference_start = time.perf_counter()
    # The refined level is archived on the finest ladder rung so every requested-grid
    # error stays re-derivable; the coarser level, which only supports the level
    # difference, is archived on a mid rung and its finer differences kept as scalars.
    archive_coarse = min(256, finest)
    refs = {}
    for case, param in enumerate(draws):
        case_start = time.perf_counter()
        fine = reference(param, fine_reference)
        coarse = reference(param, coarse_reference)
        if case == 0:
            report['restriction_checks'] = lc.validate_restriction(fine[None], meshes + [obs])
        fine_top = lc.restrict(fine[None], finest)[0]
        coarse_top = lc.restrict(coarse[None], finest)[0]
        np.savez_compressed(out / 'references' / f'case{case}.npz',
                            fine=fine_top, coarse=lc.restrict(coarse[None], archive_coarse)[0])
        row = dict(case=case, group=groups[case], seconds=time.perf_counter() - case_start,
                   fine_intervals=fine_reference, coarse_intervals=coarse_reference,
                   archived_fine_intervals=finest, archived_coarse_intervals=archive_coarse,
                   fine_sha256=lc.sha_array(fine_top), deltas={})
        for n in meshes:
            lo = lc.restrict(coarse_top[None], n)[0]
            hi = lc.restrict(fine_top[None], n)[0]
            row['deltas'][str(n)] = pc.relative(lo, hi)
            refs[n, case] = hi
        report['references'].append(row)
        del fine, coarse, fine_top, coarse_top
        if case % 6 == 0:
            print(f'REFERENCE case={case} elapsed={time.perf_counter() - begin:.1f}s', flush=True)
            save()
    report['reference_seconds'] = time.perf_counter() - reference_start
    report['reference_note'] = ('two independently refined discrete levels computed on the host with '
                                'SciPy DST-I, separate from the JAX solve path; the level difference '
                                'is empirical development evidence, not a continuum error bound')
    save()

    # -------------------------------------------------------------- the ladder
    stored = set()

    def store(field, intervals):
        """Archive the common-grid restriction of every distinct field once."""
        field = np.asarray(field)
        digest = pc.sha(field)
        if digest not in stored:
            np.savez_compressed(out / 'fields' / (digest + '.npz'),
                                observation_field=lc.restrict(field[None], obs)[0])
            stored.add(digest)
        return digest

    order_rng = np.random.default_rng(cfg['order_seed'])
    for n in meshes:
        print(f'MESH n={n}', flush=True)
        # ---- offline per-mesh setup, charged separately ---------------------
        setup_start = time.perf_counter()
        ops = pc.assemble(params, codes, n, cfg['requested_modes'], cfg['lm_budget'])
        assemble_seconds = time.perf_counter() - setup_start
        engine = prepare_correction(ops, codes, directions, count, cfg)
        setup_seconds = time.perf_counter() - setup_start
        info = dict(ops['info'], method=cfg['primary_model'], **engine['info'],
                    cache=engine['cache']['info'],
                    assemble_seconds=assemble_seconds,
                    total_offline_setup_seconds=setup_seconds,
                    component_note=('bank_build_seconds and weak_assembly_seconds are the assembler\'s '
                                    'own internal timers; the authoritative per-mesh total is '
                                    'total_offline_setup_seconds'))
        assert engine['info']['linear_rank_valid'], 'correction basis lost rank before evaluation'
        report['setup'].append(info)
        np.savez_compressed(out / f'cache_n{n}.npz', B=np.asarray(ops['B']),
                            projected_B=np.asarray(engine['Bp']), C=np.asarray(engine['C']),
                            Q=np.asarray(engine['Q']), R=np.asarray(engine['R']),
                            predictions=np.asarray(engine['cache']['predictions']))
        save()

        parts = make_correction_parts(ops, engine, cfg, count)
        cg = make_cg(n, cfg['cg_maxiter'])
        lam = jnp.asarray(pc.eigenvalues(n))

        subjects = [dict(name=cfg['primary_model'], method='rom', solver_intervals=n)]
        subjects.append(dict(name='dst', method='fom_direct', solver_intervals=n))
        for tolerance in cfg['cg_tolerances']:
            subjects.append(dict(name=f'cg_{tolerance:.0e}', method='fom_iterative',
                                 solver_intervals=n, cg_tolerance=tolerance))
        report['declared_subjects'] += [dict(intervals=n, **s) for s in subjects]

        def compute_for(subject):
            if subject['method'] == 'rom':
                return lambda source: parts['complete'](source)
            if subject['method'] == 'fom_direct':
                return lambda source: pc.dst_solve(source, lam)
            tolerance = jnp.asarray(subject['cg_tolerance'])
            return lambda source: cg(source, tolerance)

        computes = {s['name']: compute_for(s) for s in subjects}
        sources = [pc.full_source(n, param) for param in draws]

        # ---- compile and warm every shape before any timed block -------------
        warm_start = time.perf_counter()
        warm_source = jax.device_put(sources[0])
        for _ in range(cfg['warmup']):
            for subject in subjects:
                jax.block_until_ready(computes[subject['name']](warm_source))
        warm_f, warm_fp, warm_z0, _ = jax.block_until_ready(parts['contract'](warm_source))
        jax.block_until_ready(parts['reduced_solve'](warm_z0, warm_fp))
        jax.block_until_ready(parts['lift'](parts['recover'](
            warm_f, parts['reduced_solve'](warm_z0, warm_fp)[0][0], warm_z0)[0]))
        report['warmup'].append(dict(intervals=n, seconds=time.perf_counter() - warm_start,
                                     repetitions=cfg['warmup'],
                                     scope='every shape compiled and warmed once per mesh; shapes do not vary with the case'))

        # ---- parity of the staged query against the retained native query ----
        native_field, native_row = correction_query(sources[0], ops, engine, cfg)
        staged_field = np.asarray(jax.device_get(parts['complete'](warm_source)[0]))
        parity = pc.relative(staged_field, native_field)
        report['setup'][-1]['staged_vs_native_relative_difference'] = parity
        report['setup'][-1]['staged_vs_native_bitwise_identical'] = bool(
            np.array_equal(staged_field, native_field))
        assert parity <= cfg['staged_parity_tolerance'], parity
        save()

        # ---- native diagnostic row per case, outside every timer -------------
        latent_start = {}
        for case in range(ncase):
            field, row = correction_query(sources[case], ops, engine, cfg)
            report['native_rows'].append(dict(
                intervals=n, case=case, group=groups[case],
                **{k: row[k] for k in ('reason', 'attempts', 'accepted', 'jacobians', 'residual',
                                       'initial_residual', 'stationarity', 'stationary',
                                       'reduced_stationarity', 'reduced_stationary',
                                       'max_linear_backward_error', 'fallback_count',
                                       'linear_recovery_backward_error', 'projected_jacobian_rank',
                                       'projected_jacobian_rank_valid', 'solver_valid')},
                physical_error=pc.relative(field, refs[n, case]),
                field_sha256=pc.sha(field),
                scope='native fused query with its charged diagnostics; the source of every stopping and rank verdict'))
            f_value, fp_value, z0_value, _ = jax.block_until_ready(
                parts['contract'](jax.device_put(sources[case])))
            latent_start[case] = (z0_value, fp_value)
        save()

        # ---- timed ladder ----------------------------------------------------
        schedule = subjects + [dict(name='cached_reduced_solve', method='cached')]
        for rep in range(cfg['repetitions']):
            for case in range(ncase):
                truth = refs[n, case]
                truth_obs = lc.restrict(truth[None], obs)[0]
                delta = report['references'][case]['deltas'][str(n)]
                for index in order_rng.permutation(len(schedule)):
                    subject = schedule[index]
                    pc.burn(cfg['burn_seconds'])
                    if subject['method'] == 'cached':
                        z0_value, fp_value = latent_start[case]
                        value, seconds = lc.timed(parts['reduced_solve'], z0_value, fp_value)
                        answer, stats, _ = jax.device_get(value)
                        z, residual, initial, jacobians, accepted, attempts, reason = answer
                        report['cached_invocations'].append(dict(
                            intervals=n, case=case, rep=rep, cached_reduced_seconds=seconds,
                            reason=int(reason), attempts=int(attempts), accepted=int(accepted),
                            jacobians=int(jacobians), residual=float(residual),
                            initial_residual=float(initial),
                            max_linear_backward_error=float(stats[0]), fallback_count=int(stats[1]),
                            latent_sha256=lc.sha_array(np.asarray(z)),
                            scope='reduced nonlinear solve alone; excludes the source contraction and the dense decode'))
                        continue
                    value, timing = measure(computes[subject['name']], sources[case])
                    if subject['method'] == 'rom':
                        field = np.asarray(value[0])
                        answer, stats = value[1], value[2]
                        extra = dict(reason=int(answer[6]), attempts=int(answer[5]),
                                     accepted=int(answer[4]), jacobians=int(answer[3]),
                                     residual=float(answer[1]), initial_residual=float(answer[2]),
                                     max_linear_backward_error=float(stats[0]),
                                     fallback_count=int(stats[1]),
                                     selected_training_code_index=int(value[3]))
                    elif subject['method'] == 'fom_direct':
                        field = np.asarray(value)
                        extra = dict(solver='exact DST-I diagonalisation of the five-point operator',
                                     tolerance=None)
                    else:
                        field = np.asarray(value[0])
                        iterations, true_relative, recursive, converged = value[1]
                        extra = dict(iterations=int(iterations),
                                     true_relative_residual=float(true_relative),
                                     recursive_relative_residual=float(recursive),
                                     cg_converged=bool(converged),
                                     cg_tolerance_satisfied=bool(converged))
                    error = pc.relative(field, truth)
                    finite = bool(np.isfinite(field).all())
                    report['invocations'].append(dict(
                        intervals=n, nodes_per_axis=n + 1, interior_unknowns=(n - 1) ** 2,
                        case=case, group=groups[case], rep=rep, **subject, **timing, **extra,
                        physical_error_requested_grid=error,
                        physical_error_common_grid=pc.relative(lc.restrict(field[None], obs)[0], truth_obs),
                        reference_delta=delta,
                        conservative_physical_error=(error + delta) / (1 - delta),
                        output_bytes=int(field.nbytes), field_sha256=store(field, n),
                        source_sha256=pc.sha(sources[case]), finite=finite))
            print(f'TIMED n={n} rep={rep} elapsed={time.perf_counter() - begin:.1f}s', flush=True)
            save()

        # ---- staged components of one invocation, diagnostic only ------------
        for case in range(ncase):
            pc.burn(cfg['burn_seconds'])
            start = time.perf_counter()
            source = jax.device_put(sources[case])
            jax.block_until_ready(source)
            t1 = time.perf_counter()
            f_value, fp_value, z0_value, _ = jax.block_until_ready(parts['contract'](source))
            t2 = time.perf_counter()
            answer, stats, _ = jax.block_until_ready(parts['reduced_solve'](z0_value, fp_value))
            t3 = time.perf_counter()
            coefficients, y, _ = jax.block_until_ready(parts['recover'](f_value, answer[0], z0_value))
            t4 = time.perf_counter()
            field_device = jax.block_until_ready(parts['lift'](coefficients))
            t5 = time.perf_counter()
            field = np.asarray(jax.device_get(field_device))
            t6 = time.perf_counter()
            fused = np.asarray(jax.device_get(parts['complete'](source)[0]))
            report['component_invocations'].append(dict(
                intervals=n, case=case, input_transfer_s=t1 - start,
                source_contraction_s=t2 - t1, reduced_solve_s=t3 - t2,
                linear_recovery_s=t4 - t3, dense_decode_s=t5 - t4, output_transfer_s=t6 - t5,
                staged_total_s=t6 - start,
                fused_relative_difference=pc.relative(field, fused),
                interpretation=('separately synchronized stages of one staged invocation; diagnostic '
                                'only, never substituted into the complete-query column')))
        save()

        np.savez_compressed(out / f'dense_n{n}_case0.npz',
                            rom=np.asarray(jax.device_get(parts['complete'](
                                jax.device_put(sources[0]))[0])),
                            dst=np.asarray(jax.device_get(pc.dst_solve(
                                jax.device_put(sources[0]), lam))),
                            reference=refs[n, 0])
        del ops, engine, parts, cg, lam, computes, sources, latent_start
        gc.collect()
        jax.clear_caches()

    report['checkpoint_sha256_after'] = lc.sha_file(checkpoint)
    assert report['checkpoint_sha256_after'] == checkpoint_sha
    expected = ncase * len(meshes) * cfg['repetitions'] * len(report['declared_subjects']) // len(meshes)
    assert len(report['invocations']) == expected, (len(report['invocations']), expected)
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print(f'POISSON LADDER COMPLETE seconds={report["elapsed_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
