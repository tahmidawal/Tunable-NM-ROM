"""Frozen-checkpoint Burgers 2D mesh ladder in one job on one GPU.

The checkpoint, latent dimension, bank rank, weak-mode count, quadrature budget,
solver policy and stopping rule are all fixed. Only the mesh changes. New-grid
truth is used for evaluation only; nothing here tunes, selects or retrains.

Three costs are measured separately, each from its own completed device
computation, and never by subtracting one from another:

* cached reduced solve - the latent evolution alone,
* complete device query - supplied device field to six dense device fields,
* offline per-mesh setup - bank evaluation, operator rebuild and EQ refit.

Host transfers are timed inside the same invocation as the complete device query
and reported as separate columns.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
sys.path.insert(0, str(ROOT / 'experiments/mr-burgers2d'))

import engines as e                      # noqa: E402
import accuracy_paths as ap              # noqa: E402
import iterative_paths as ip             # noqa: E402
import ladder_common as lc               # noqa: E402


def make_rom_parts(params, L, dt, trust, ic_budget=400, step_budget=180, gtol=1e-6):
    """The retained stationarity-stopped solver, expressed as separable stages.

    Every line is the corresponding line of ``accuracy_paths.make_rom``; the only
    change is that the initial fit, the latent evolution and the dense decode are
    exposed as three separately compilable functions so each can be timed on its
    own. ``query`` composes exactly those three stages, and the driver asserts
    that it reproduces ``accuracy_paths.make_rom`` on the same inputs.
    """
    K = params['h_lin'].shape[0]
    ic = ap.make_stationary_lm(lambda z, y, R: R @ e.sc.head(params, z) - y, K, ic_budget, gtol=gtol)
    lm = ap.make_stationary_lm(lambda z, p, nu, data: e.weak(z, p, nu, data, params, L, dt),
                               K, step_budget, trust, gtol)
    steps = int(round(.25 / dt))
    stride = int(round(.05 / dt))

    def initialize(u0, data, cold):
        xy, w, Q, R, Hrot, Hnorm = cold
        ui = e.sample_field(u0, xy, L) * w
        y = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ y)
        z, icrn, icit, icreason, icgn = ic(data[7][idx], (y, R), 0.)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(w))
        return z, icit, icreason, scale, icgn

    def evolve(z, nu, scale, data):
        def step(carry, _):
            z, zprev = carry
            p = data[1] @ e.sc.head(params, z)
            ze = z + (z - zprev)
            r0 = jnp.linalg.norm(e.weak(z, p, nu, data, params, L, dt))
            re = jnp.linalg.norm(e.weak(ze, p, nu, data, params, L, dt))
            zi = jnp.where(jnp.isfinite(re) & (re < r0), ze, z)
            z2, rn, it, reason, gn = lm(zi, (p, nu, data), 1e-9 * scale)
            return (z2, z), (z2, rn, it, reason, gn)
        _, (zs, rn, it, reason, gn) = jax.lax.scan(step, (z, z), None, length=steps)
        return jnp.concatenate((z[None], zs)), it, rn, reason, gn

    def decode(internal, data):
        Z = internal[::stride]
        return jax.vmap(lambda z: e.output_field(data[0] @ e.sc.head(params, z), L, L))(Z), Z

    def query(u0, nu, data, cold):
        z, icit, icreason, scale, icgn = initialize(u0, data, cold)
        internal, it, rn, reason, gn = evolve(z, nu, scale, data)
        fields, Z = decode(internal, data)
        return fields, it, rn, reason, Z, icit, icreason, internal, gn, icgn

    return jax.jit(query), dict(initialize=jax.jit(initialize), evolve=jax.jit(evolve),
                                decode=jax.jit(decode))


def reference_settings(mesh, step):
    """Six independently converged solves: three spatial levels and three temporal."""
    return [(mesh // 4, 2 * step), (mesh // 2, 2 * step), (mesh // 2, step),
            (mesh, 4 * step), (mesh, 2 * step), (mesh, step)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()

    environment = lc.preflight()
    checkpoint_path = ROOT / cfg['checkpoint']
    checkpoint_sha = lc.sha_file(checkpoint_path)
    assert checkpoint_sha == cfg['checkpoint_sha256'], checkpoint_sha
    checkpoint = pickle.load(open(checkpoint_path, 'rb'))
    params_host = lc.host(checkpoint['params'])
    params = jax.tree_util.tree_map(jnp.asarray, params_host)
    Z = checkpoint['Z_tr']
    K = Z.shape[1]
    R = params_host['h_lin'].shape[1]
    M, m = 4 * K, 16 * K
    meshes = list(cfg['meshes'])
    obs = cfg['observation_intervals']
    largest = max(meshes)
    dt = cfg['dt']
    assert all(L % obs == 0 for L in meshes) and largest % obs == 0
    assert cfg['reference_mesh'] % largest == 0

    physical = np.concatenate((e.params_draw(cfg['seed'], cfg['cases']),
                               e.params_draw(cfg['fresh_seed'], cfg['fresh_cases'])))
    ncase = len(physical)

    report = dict(
        pde='burgers', config=cfg, **environment,
        checkpoint=cfg['checkpoint'], checkpoint_sha256=checkpoint_sha,
        checkpoint_cfg={k: v for k, v in checkpoint['cfg'].items() if k not in ('hfit_pick', 'train')},
        K=K, R=R, M=M, m=m,
        frozen_weight_transfer=('one checkpoint, one latent dimension, one bank rank, one weak-mode '
                                'count, one quadrature budget, one solver policy and one stopping rule '
                                'across every mesh; new-grid truth evaluates only'),
        network_weights_frozen=True, final_cohort_unopened=True,
        physical_cases=physical.tolist(),
        cohort_roles=['opened development'] * cfg['cases'] + ['fresh development'] * cfg['fresh_cases'],
        output_times=[0, .05, .10, .15, .20, .25],
        cost_contract=dict(
            cached_reduced_solve='latent evolution alone, all reduced operators already resident',
            complete_device_query='supplied dense field already on the device to six dense output fields still on the device',
            offline_setup='bank evaluation on the new mesh, operator rebuild and EQ refit from decoder outputs',
            host_transfers='timed in the same invocation as the complete device query and reported separately',
            rule='each cost is its own completed device computation; no cost is obtained by subtraction'),
        restriction=dict(operator='nested-node injection (stride selection)',
                         common_observation_intervals=obs,
                         reason='every ladder rung and the refined reference share the coarsest grid as nested nodes'),
        verification=ip.verify(),
        reference=[], reference_metrics={}, restriction_checks={}, mesh_setup=[],
        invocations=[], cached_invocations=[], component_invocations=[],
        same_grid_discrepancy=[], declared_subjects=[], compile_warmup=[], complete=False)
    save = lambda: lc.dump(out / 'result.json', report)
    save()

    # ---------------------------------------------------------------- references
    refs = {}
    rf, rt = cfg['reference_mesh'], cfg['reference_dt']
    for L, step in reference_settings(rf, rt):
        print(f'REFERENCE L={L} dt={step}', flush=True)
        query, _ = e.make_fom(L, step)
        for case, phys in enumerate(physical):
            start = time.perf_counter()
            fields, iterations, residual = lc.host(
                query(jnp.asarray(e.initial(L, phys)), float(phys[4]),
                      cfg['reference_newton_tolerance'], cfg['reference_linear_tolerance']))
            worst = float(np.max(residual))
            assert np.isfinite(fields).all() and worst <= cfg['reference_residual_limit'], (L, step, case, worst)
            coarse = lc.restrict(fields, largest)
            if (L, step) == (rf, rt) and case == 0:
                report['restriction_checks'] = lc.validate_restriction(fields, meshes + [obs])
            refs[L, step, case] = coarse
            # The primary reference is archived on the finest ladder rung, so every
            # requested-grid error is re-derivable; the five refinement controls are
            # archived on the common observation grid, which is where their margins
            # are read. Both retain the full-resolution digest.
            primary = (L, step) == (rf, rt)
            name = f'ref_L{L}_dt{step}_case{case}.npz'
            np.savez_compressed(out / name,
                                fields=coarse if primary else lc.restrict(coarse, obs))
            report['reference'].append(dict(intervals=L, dt=step, case=case, artifact=name,
                                            role='primary' if primary else 'refinement control',
                                            archived_intervals=largest if primary else obs,
                                            restricted_to=largest, max_relative_residual=worst,
                                            newton_iterations=iterations.tolist(),
                                            seconds=time.perf_counter() - start,
                                            field_sha256=lc.sha_array(coarse),
                                            observation_sha256=lc.sha_array(lc.restrict(coarse, obs))))
            save()
        del query
        jax.clear_caches()

    # Reference uncertainty at every grid the ladder actually reports on.
    target = cfg['target_fixed_initial'] * cfg['reference_margin_fraction']
    for grid in sorted(set(meshes + [obs])):
        rows = []
        for case in range(ncase):
            def difference(a, b, c, d):
                return e.errors(lc.restrict(refs[a, b, case], grid),
                                lc.restrict(refs[c, d, case], grid), grid)['fixed_initial_max']
            space = difference(rf // 2, rt, rf, rt)
            temporal = difference(rf, 2 * rt, rf, rt)
            space_coarse = difference(rf // 4, 2 * rt, rf // 2, 2 * rt)
            space_fine = difference(rf // 2, 2 * rt, rf, 2 * rt)
            time_coarse = difference(rf, 4 * rt, rf, 2 * rt)
            rows.append(dict(case=case, space_difference=space, time_difference=temporal,
                             margin=space + temporal,
                             spatial_coarse=space_coarse, spatial_fine=space_fine,
                             temporal_coarse=time_coarse,
                             observed_space_order=float(np.log2(space_coarse / space_fine)),
                             observed_time_order=float(np.log2(time_coarse / temporal)),
                             decrease=bool(space_fine < space_coarse and temporal < time_coarse)))
        report['reference_metrics'][str(grid)] = dict(
            rows=rows, worst_margin=max(r['margin'] for r in rows),
            margin_budget=target,
            resolved=bool(max(r['margin'] for r in rows) <= target and all(r['decrease'] for r in rows)),
            interpretation=('space and time refinement differences are empirical development evidence, '
                            'not a rigorous continuum error bound'))
    save()

    # -------------------------------------------------------------- the ladder
    order_rng = np.random.default_rng(cfg['order_seed'])
    for L in meshes:
        print(f'MESH L={L}', flush=True)
        truth = {case: lc.restrict(refs[rf, rt, case], L) for case in range(ncase)}
        truth_obs = {case: lc.restrict(refs[rf, rt, case], obs) for case in range(ncase)}

        # ---- offline per-mesh setup, charged separately ----------------------
        decoder = e.sc.SeparableDecoder(params, K, R)
        bank_start = time.perf_counter()
        bank = jax.block_until_ready(decoder.feat_at(e.coords(L), chunk=8192))
        bank_seconds = time.perf_counter() - bank_start
        bank_bytes = int(bank.nbytes)
        del bank
        jax.clear_caches()
        operator_start = time.perf_counter()
        basis, eigenvalues, _ = e.modes(L, M)
        basis_device = jnp.asarray(basis)
        jax.block_until_ready(basis_device)
        operator_seconds = time.perf_counter() - operator_start
        del basis, basis_device, eigenvalues
        jax.clear_caches()
        data, info = e.build_rom(params, Z, L, M, m,
                                 candidate_cap=cfg['candidate_cap'], fit_states=cfg['fit_states'])
        cold, cold_info = e.build_gauss_cold(params, data[7])
        info.update(
            model='frozen', cold=cold_info,
            bank_evaluation_seconds=bank_seconds, bank_bytes=bank_bytes,
            weak_mode_construction_seconds=operator_seconds,
            total_offline_setup_seconds=info['setup_seconds'] + cold_info['setup_seconds'],
            component_note=('bank_evaluation_seconds and weak_mode_construction_seconds are separate '
                            'standalone measurements of the same work, reported as components; the '
                            'authoritative per-mesh setup total is setup_seconds plus the cold setup'))
        np.savez_compressed(out / f'operators_L{L}.npz', A=np.asarray(data[1]), lam=np.asarray(data[2]),
                            G5=np.asarray(data[3]), Pq=np.asarray(data[4]),
                            candidate_Z=np.asarray(data[7]), cold_xy=np.asarray(cold[0]),
                            cold_w=np.asarray(cold[1]), cold_Q=np.asarray(cold[2]),
                            cold_R=np.asarray(cold[3]))
        report['mesh_setup'].append(info)
        save()

        # ---- solvers ---------------------------------------------------------
        trust = info['trust_radius']
        staged_query, parts = make_rom_parts(params, L, dt, trust, **cfg['strict'])
        retained_query = ap.make_rom(params, L, dt, trust, **cfg['strict'])
        diagnostics = ap.make_diagnostics(params, L, dt)

        subjects = [dict(name='rom_frozen_stationary', method='rom', solver_intervals=L,
                         output_intervals=L, dt=dt, **cfg['strict'])]
        seen = set()
        foms = {}
        for setting in cfg['fom_settings']:
            scale = setting['solver_scale']
            grid = L if scale == 1 else (obs if scale == 0 else L // scale)
            if grid < 8 or L % grid:
                continue
            key = (grid, setting['ntol'], setting['ltol'])
            if key in seen:
                continue
            seen.add(key)
            foms[setting['name']] = e.make_fom(grid, dt, target=L)[0]
            subjects.append(dict(name=setting['name'], method='fom', solver_intervals=grid,
                                 output_intervals=L, dt=dt, newton_tolerance=setting['ntol'],
                                 linear_tolerance=setting['ltol']))
        report['declared_subjects'] += [dict(intervals=L, **s) for s in subjects]

        inputs = [e.initial(L, phys) for phys in physical]

        def invoke(subject, u0, case):
            nu = float(physical[case, 4])
            if subject['method'] == 'rom':
                return staged_query(u0, nu, data, cold)
            return foms[subject['name']](u0, nu, subject['newton_tolerance'], subject['linear_tolerance'])

        # ---- same-grid tight FOM, for the ROM-vs-same-grid discrepancy --------
        tight_query, _ = e.make_fom(L, dt)
        same_grid = {}
        for case, phys in enumerate(physical):
            fields, iterations, residual = lc.host(
                tight_query(jnp.asarray(inputs[case]), float(phys[4]),
                            cfg['same_grid_fom']['ntol'], cfg['same_grid_fom']['ltol']))
            worst = float(np.max(residual))
            assert np.isfinite(fields).all() and worst <= cfg['reference_residual_limit'], (L, case, worst)
            same_grid[case] = np.asarray(fields)
            report['same_grid_discrepancy'].append(dict(
                intervals=L, case=case, **cfg['same_grid_fom'], max_relative_residual=worst,
                physical_error_common_grid=e.errors(fields, truth_obs[case], obs),
                physical_error_requested_grid=e.errors(fields, truth[case], L),
                role='converged same-grid full-order solve; the ROM is compared against it and against the refined reference'))
        del tight_query
        save()

        # ---- compile and warm every shape before any timed block -------------
        warm_u0 = jnp.asarray(inputs[0])
        compile_start = time.perf_counter()
        for subject in subjects:
            jax.block_until_ready(invoke(subject, warm_u0, 0))
        warm_state = parts['initialize'](warm_u0, data, cold)
        jax.block_until_ready(warm_state)
        jax.block_until_ready(parts['evolve'](warm_state[0], float(physical[0, 4]), warm_state[3], data))
        jax.block_until_ready(parts['decode'](
            parts['evolve'](warm_state[0], float(physical[0, 4]), warm_state[3], data)[0], data))
        jax.block_until_ready(retained_query(warm_u0, float(physical[0, 4]), data, cold))
        report['compile_warmup'].append(dict(intervals=L, seconds=time.perf_counter() - compile_start))

        # ---- parity of the staged solver against the retained selected solver -
        staged_value = lc.host(staged_query(warm_u0, float(physical[0, 4]), data, cold))
        retained_value = lc.host(retained_query(warm_u0, float(physical[0, 4]), data, cold))
        parity = float(np.linalg.norm(staged_value[0] - retained_value[0])
                       / max(np.linalg.norm(retained_value[0]), 1e-300))
        report['mesh_setup'][-1]['staged_vs_retained_relative_difference'] = parity
        report['mesh_setup'][-1]['staged_vs_retained_identical'] = bool(
            np.array_equal(staged_value[0], retained_value[0]))
        assert parity <= 1e-12, parity
        save()

        # ---- cached reduced solve: latent evolution alone --------------------
        latent_start = {}
        for case in range(ncase):
            state = jax.block_until_ready(parts['initialize'](jnp.asarray(inputs[case]), data, cold))
            latent_start[case] = (state[0], state[3])

        artifacts = {}
        # The cached reduced solve takes its turn in the same randomized order as the
        # complete queries, so no measurement sits at a fixed position in the sequence.
        schedule = subjects + [dict(name='cached_reduced_solve', method='cached')]
        for rep in range(cfg['reps']):
            for case in range(ncase):
                nu = float(physical[case, 4])
                for index in order_rng.permutation(len(schedule)):
                    subject = schedule[index]
                    e.burn(cfg['burn_seconds'])
                    if subject['method'] == 'cached':
                        value, seconds = lc.timed(parts['evolve'], latent_start[case][0], nu,
                                                  latent_start[case][1], data)
                        internal, iterations, residual, reason, gradient = lc.host(value)
                        report['cached_invocations'].append(dict(
                            intervals=L, case=case, rep=rep, cached_reduced_seconds=seconds,
                            time_steps=int(len(iterations)), lm_iterations=int(np.sum(iterations)),
                            iterations=iterations.tolist(), stop_reasons=reason.tolist(),
                            max_normalized_stationarity=float(np.max(gradient)),
                            stationary=bool(np.max(gradient) <= cfg['strict']['gtol'] * (1 + 1e-7)),
                            latent_sha256=lc.sha_array(internal),
                            scope='latent evolution alone; excludes the initial fit and the dense decode'))
                        continue
                    transfer_start = time.perf_counter()
                    u0 = jax.device_put(np.array(inputs[case], copy=True))
                    jax.block_until_ready(u0)
                    input_transfer = time.perf_counter() - transfer_start
                    device_start = time.perf_counter()
                    value = jax.block_until_ready(invoke(subject, u0, case))
                    device_seconds = time.perf_counter() - device_start
                    output_start = time.perf_counter()
                    fields = np.asarray(value[0])
                    output_transfer = time.perf_counter() - output_start
                    rest = lc.host(value[1:])
                    finite = bool(np.isfinite(fields).all())
                    digest = lc.sha_array(fields)
                    row = dict(intervals=L, nodes_per_axis=L + 1, interior_unknowns=(L - 1) ** 2,
                               case=case, cohort=report['cohort_roles'][case], rep=rep, **subject,
                               complete_device_seconds=device_seconds,
                               input_transfer_seconds=input_transfer,
                               output_transfer_seconds=output_transfer,
                               host_to_host_seconds=input_transfer + device_seconds + output_transfer,
                               output_bytes=int(fields.nbytes), field_sha256=digest, finite=finite,
                               physical_error_common_grid=e.errors(fields, truth_obs[case], obs) if finite else None,
                               physical_error_requested_grid=e.errors(fields, truth[case], L) if finite else None,
                               same_grid_discrepancy_common=e.errors(fields, lc.restrict(same_grid[case], obs), obs) if finite else None,
                               same_grid_discrepancy_requested=e.errors(fields, same_grid[case], L) if finite else None,
                               iterations=rest[0].tolist(), residuals=rest[1].tolist())
                    if subject['method'] == 'rom':
                        gradient = rest[7]
                        row.update(stop_reasons=rest[2].tolist(), latent_states=rest[3].tolist(),
                                   ic_iterations=int(rest[4]), ic_reason=int(rest[5]),
                                   step_normalized_stationarity=gradient.tolist(),
                                   ic_normalized_stationarity=float(rest[8]),
                                   stationary=bool(max(float(np.max(gradient)), float(rest[8]))
                                                   <= cfg['strict']['gtol'] * (1 + 1e-7)))
                        audit_gn, audit_icgn = lc.host(diagnostics(jnp.asarray(rest[6]), u0, nu, data, cold))
                        row['audit_max_stationarity_difference'] = float(
                            max(np.max(abs(audit_gn - gradient)), abs(float(audit_icgn) - float(rest[8]))))
                    else:
                        row['nonlinear_tolerance_satisfied'] = bool(
                            finite and np.max(rest[1]) <= subject['newton_tolerance'] * (1 + 1e-9))
                    key = (subject['name'], case)
                    if key not in artifacts:
                        name = f'L{L}_{subject["name"]}_case{case}_rep{rep}.npz'
                        np.savez_compressed(out / name, observation_fields=lc.restrict(fields, obs))
                        artifacts[key] = (name, digest)
                    row['observation_artifact'] = artifacts[key][0]
                    row['matches_first_field_sha256'] = digest == artifacts[key][1]
                    report['invocations'].append(row)
            print(f'TIMED L={L} rep={rep} elapsed={time.perf_counter() - begin:.1f}s', flush=True)
            save()

        # ---- staged components of one invocation, diagnostic only ------------
        for case in range(ncase):
            nu = float(physical[case, 4])
            e.burn(cfg['burn_seconds'])
            start = time.perf_counter()
            u0 = jax.device_put(np.array(inputs[case], copy=True))
            jax.block_until_ready(u0)
            t1 = time.perf_counter()
            state = jax.block_until_ready(parts['initialize'](u0, data, cold))
            t2 = time.perf_counter()
            evolved = jax.block_until_ready(parts['evolve'](state[0], nu, state[3], data))
            t3 = time.perf_counter()
            decoded = jax.block_until_ready(parts['decode'](evolved[0], data))
            t4 = time.perf_counter()
            fields = np.asarray(decoded[0])
            t5 = time.perf_counter()
            fused = np.asarray(jax.block_until_ready(staged_query(u0, nu, data, cold))[0])
            report['component_invocations'].append(dict(
                intervals=L, case=case, input_transfer_s=t1 - start, initial_fit_s=t2 - t1,
                latent_evolution_s=t3 - t2, dense_decode_s=t4 - t3, output_transfer_s=t5 - t4,
                staged_total_s=t5 - start,
                fused_relative_difference=float(np.linalg.norm(fields - fused)
                                                / max(np.linalg.norm(fused), 1e-300)),
                interpretation=('separately synchronized stages of one staged invocation; diagnostic '
                                'only, never substituted into the complete-query column')))
        save()

        np.savez_compressed(out / f'dense_L{L}_rom_case0.npz',
                            fields=np.asarray(lc.host(staged_query(jnp.asarray(inputs[0]),
                                                                   float(physical[0, 4]), data, cold))[0]))
        del data, cold, staged_query, retained_query, parts, diagnostics, foms, same_grid, decoder
        jax.clear_caches()

    report['checkpoint_sha256_after'] = lc.sha_file(checkpoint_path)
    assert report['checkpoint_sha256_after'] == checkpoint_sha
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print(f'BURGERS LADDER COMPLETE seconds={report["elapsed_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
