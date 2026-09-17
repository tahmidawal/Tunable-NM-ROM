"""w-ladder: the reflective-wave correction ladder from one frozen bank and head.

Every frozen wave module is imported unchanged (pilot, acceleration, modal_projection,
iterative_replay, dynamics, heads32, fresh-wave-head).  This file adds arms only:
nested q-enrichments of the retained K=32 head, integrator variants of the linear bank,
POD-Galerkin at matched ranks, a coarse-mesh direct FOM, and the three-layer decomposition.
One mesh per job.  All timing is same-job; all scoring is against the same-grid DST truth.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

WAVE = Path(__file__).resolve().parents[1] / 'multiresolution-wave'
sys.path.insert(0, str(WAVE))
import pilot as base                                    # noqa: E402  (inserts fresh-wave-head)
from pilot import jax, jnp, Grid, localized_initial, parameter_rows, metrics, restrict  # noqa: E402
import iterative_replay as previous                     # noqa: E402
import acceleration as fast                             # noqa: E402
import acceleration_replay as ar                        # noqa: E402
import modal_projection as modal                        # noqa: E402
import dynamics as dyn                                  # noqa: E402
from fresh_models import tree_from_npz, head_apply, head_geometry  # noqa: E402
from fresh_learning import fit_batch                    # noqa: E402
from fresh_fom import provenance, integrate_balance, energy  # noqa: E402

FIXED = 6            # fixed training-code starts, as in pilot.cold_fit
STARTS = 2 + FIXED   # affine, zero, six fixed codes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- training data
def regenerate_training_fields(inputs, cfg):
    """Second streaming pass over the 64 training trajectories, retaining normalised fields.

    Same generator, seed, mesh, CFL and stride as fresh_learning.generate_data; the stacked
    displacement/velocity bytes are hashed and gated against data_manifest.json.
    """
    original = json.loads((inputs / 'campaign-config.json').read_text())
    expected = json.loads((inputs / 'data_manifest.json').read_text())['splits']['train']
    grid = Grid(original['n'], 'dirichlet', 'dirichlet')
    pars = parameter_rows(original['train_seed'], original['train_count'])
    case_stride = int(cfg.get('pod_training_case_stride', 1))
    stride = int(np.ceil(original['observation_dt'] / (original['fom_cfl'] * grid.h / 1.15)))
    dt = original['observation_dt'] / stride
    nobs = int(round(original['end_time'] / original['observation_dt']))
    us, vs, scales = [], [], []
    start = time.perf_counter()
    for ci, par in enumerate(pars):
        if ci % case_stride:
            continue
        u0, v0 = localized_initial(grid, par)
        u, v, _ = integrate_balance(u0, v0, par[5], dt, grid=grid, steps=nobs * stride, stride=stride)
        e0 = float(energy(u0, v0, grid, par[5]))
        u, v = np.asarray(u), np.asarray(v)
        us.append(u.reshape(len(u), -1)); vs.append(v.reshape(len(v), -1))
        scales.append((float(np.sqrt(np.sum(grid.mass() * np.asarray(u0) ** 2))), float(np.sqrt(2 * e0))))
        print('pod_training_case', ci, flush=True)
    us, vs, scales = np.stack(us), np.stack(vs), np.asarray(scales)
    uh, vh = hashlib.sha256(us.tobytes()).hexdigest(), hashlib.sha256(vs.tobytes()).hexdigest()
    np.testing.assert_allclose(scales, np.asarray(expected['scales'])[::case_stride], rtol=1e-12, atol=1e-14)
    if case_stride == 1:
        match = uh == expected['u_sha256'] and vh == expected['v_sha256']
        if not match:
            raise RuntimeError('Regenerated POD training fields do not match data_manifest.json')
    else:
        match = None   # smoke-only subsample: the campaign hash covers all 64 cases and is not checked
    return grid, us, vs, scales, dict(u_sha256=uh, v_sha256=vh, hashes_match=match, case_stride=case_stride,
                                      seconds_including_first_compile=time.perf_counter() - start,
                                      seed=original['train_seed'], count=len(us), intervals=grid.n, dt=dt)


@partial(jax.jit, static_argnames=('modes',))
def pod_gram_modes(snapshots, sqrt_mass, modes):
    """Method of snapshots in the mass inner product; returns mass-orthonormal leading modes."""
    weighted = snapshots * sqrt_mass[None, :]
    gram = weighted @ weighted.T
    eigen, vectors = jnp.linalg.eigh(gram)
    order = jnp.argsort(-eigen)
    eigen, vectors = eigen[order][:modes], vectors[:, order][:, :modes]
    singular = jnp.sqrt(jnp.maximum(eigen, 0.))
    phi = (weighted.T @ vectors) / singular[None, :]
    return phi / sqrt_mass[:, None], jnp.sqrt(jnp.maximum(eigen, 0.)), jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(gram), 0.))


def build_pod(inputs, out, modes, cfg):
    grid, us, vs, scales, manifest = regenerate_training_fields(inputs, cfg)
    snapshots = np.concatenate((us / scales[:, 0, None, None], vs / scales[:, 1, None, None]), axis=1)
    snapshots = snapshots.reshape(-1, snapshots.shape[-1])
    sqrt_mass = jnp.sqrt(jnp.asarray(grid.mass().ravel()))
    phi, singular, all_singular = pod_gram_modes(jnp.asarray(snapshots), sqrt_mass, modes)
    jax.block_until_ready(phi)
    phi, singular, all_singular = np.asarray(phi), np.asarray(singular), np.asarray(all_singular)
    mass = grid.mass().ravel()
    defect = float(np.linalg.norm(phi.T @ (mass[:, None] * phi) - np.eye(modes)))
    np.savez_compressed(out / 'pod_basis_256.npz', phi=phi, singular_values=singular, all_singular_values=all_singular,
                        case_scales=scales, snapshot_count=snapshots.shape[0])
    manifest.update(modes=modes, snapshot_count=int(snapshots.shape[0]), orthogonality_defect_256=defect,
                    singular_values=singular.tolist(), energy_fraction_captured={
                        str(k): float(np.sum(singular[:k] ** 2) / np.sum(all_singular ** 2)) for k in (16, 40, 64, 128) if k <= modes},
                    construction='Displacement and velocity snapshots of the training trajectories at 256 intervals, each case '
                                 'normalised by its initial displacement mass-norm and sqrt(2 E0); EXACT method of snapshots (eigh of the '
                                 'Gram matrix) in the trapezoid-mass inner product; modes are the leading left singular vectors. This is '
                                 'not the campaign-config randomized SVD.')
    if defect > 1e-8:
        raise RuntimeError('POD orthogonality defect at 256')
    return phi, manifest


def dst_prolong(fields, target):
    """Sine interpolant of Dirichlet-grid fields (..., n-1, n-1) evaluated on the target grid."""
    n = fields.shape[-1] + 1
    coefficients = base.dst2(jnp.asarray(fields))
    pad = [(0, 0)] * (fields.ndim - 2) + [(0, target - n), (0, target - n)]
    return base.dst2(jnp.pad(coefficients, pad)) * (target / n)


def transfer_modes(phi256, grid):
    """Evaluate each 256-grid mode's sine interpolant on the query grid (nodal on nested coarse grids)."""
    modes = phi256.T.reshape(phi256.shape[1], 255, 255)
    if grid.n == 256:
        return jnp.asarray(phi256)
    if grid.n < 256:
        return jnp.asarray(restrict(modes, 256, grid.n, 'dirichlet').reshape(modes.shape[0], -1).T)
    return dst_prolong(modes, grid.n).reshape(modes.shape[0], -1).T


def sine_transfer_check(grid):
    """Pure discrete sine modes (low and near the 256-grid Nyquist) must transfer exactly; prolongation must round-trip."""
    xy256 = Grid(256).coordinates(); xy = grid.coordinates(); worst = 0.
    for kx, ky in ((3, 2), (200, 37), (251, 253)):
        mode = np.sin(kx * np.pi * xy256[..., 0]) * np.sin(ky * np.pi * xy256[..., 1])
        exact = np.sin(kx * np.pi * xy[..., 0]) * np.sin(ky * np.pi * xy[..., 1])
        got = np.asarray(transfer_modes(mode.reshape(-1, 1), grid)).reshape(grid.shape)
        worst = max(worst, float(np.max(abs(got - exact))))
        if grid.n > 256:
            back = restrict(np.asarray(dst_prolong(mode[None], grid.n))[0], grid.n, 256, 'dirichlet')
            worst = max(worst, float(np.max(abs(back - mode))))
    return worst


def pod_banks(phi256, grid, ranks, mass, boundary):
    raw = transfer_modes(phi256, grid)
    q, r = jnp.linalg.qr(jnp.sqrt(mass)[:, None] * raw, mode='reduced')
    phi = q / jnp.sqrt(mass)[:, None]
    kk, dd = base.assemble(phi, mass, boundary, grid=grid)
    jax.block_until_ready((phi, kk, dd))
    audits = dict(transfer_qr_diagonal_min_over_max=float(jnp.min(abs(jnp.diag(r))) / jnp.max(abs(jnp.diag(r)))),
                  orthogonality=float(jnp.linalg.norm(phi.T @ (mass[:, None] * phi) - jnp.eye(phi.shape[1]))),
                  stiffness_symmetry=float(jnp.max(abs(kk - kk.T))), stiffness_min_eigenvalue=float(jnp.linalg.eigvalsh(kk)[0]),
                  sine_mode_transfer_error=sine_transfer_check(grid))
    if audits['orthogonality'] > 1e-9 or audits['sine_mode_transfer_error'] > 1e-12 or audits['stiffness_min_eigenvalue'] <= 0:
        raise RuntimeError('POD transfer audit failed: ' + json.dumps(audits))
    banks = {}
    for k in ranks:
        sub = kk[:k, :k]
        eigen, vectors = jnp.linalg.eigh(sub)
        banks[f'pod_k{k}'] = dict(g=phi[:, :k], mass=mass, k=sub, d=dd[:k, :k], prepared=dict(eigen=eigen, vectors=vectors),
                                  dimension=k)
    return phi, np.asarray(kk), banks, audits


# ----------------------------------------------------------------------------- nested q heads
def nested_model(full, head32, ladder, q, sample_codes_rng):
    """h_{32+q}(z,y) = h32(z) + B_q y with the SCALED training-PCA directions 32:32+q (nested_head.build's convention)."""
    p = head32['p']
    basis = jnp.asarray(ladder['standardized_linear'][:, 32:32 + q])
    enriched = {**p, 'linear': jnp.concatenate((p['linear'], basis), axis=1),
                'l1': {**p['l1'], 'w': jnp.concatenate((p['l1']['w'], jnp.zeros((q, p['l1']['w'].shape[1]))), axis=0)}}
    codes = jnp.concatenate((jnp.asarray(head32['codes']), jnp.zeros((len(head32['codes']), q))), axis=1)
    errors, ranks = [], []
    for ii in np.linspace(0, len(codes) - 1, FIXED, dtype=int):
        z = jnp.asarray(head32['codes'][ii]); w = jnp.asarray(sample_codes_rng.normal(size=32))
        zz, ww = jnp.concatenate((z, jnp.zeros(q))), jnp.concatenate((w, jnp.zeros(q)))
        a, b, jac, curve = head_geometry(p, head32['frozen'], z, w, 'mlp')
        aa, bb, jj, cc = head_geometry(enriched, head32['frozen'], zz, ww, 'mlp')
        errors.append(max(float(jnp.max(abs(a - aa))), float(jnp.max(abs(b - bb))), float(jnp.max(abs(jac - jj[:, :32]))),
                          float(jnp.max(abs(curve - cc))), float(jnp.max(abs(jj[:, 32:] - basis))) if q else 0.))
        s = np.linalg.svd(np.asarray(jj), compute_uv=False)
        st = np.linalg.svd(np.asarray(full['transform']) @ np.asarray(jj), compute_uv=False)
        ranks.append(float(min(s[-1] / s[0], st[-1] / st[0])))
    if max(errors) > 1e-10 or min(ranks) <= 1e-8:
        raise RuntimeError(f'Nested inclusion check failed at q={q}: {errors} {ranks}')
    transform = jnp.asarray(full['transform'])
    linear = ladder['standardized_linear'][:, :32 + q]
    candidate = dict(full, p=base.transform_head(enriched, transform), frozen=head32['frozen'],
                     common_inverse=jnp.linalg.pinv(transform @ jnp.asarray(linear)), common_center=transform @ jnp.asarray(ladder['center']),
                     fixed_codes=codes[np.linspace(0, len(codes) - 1, FIXED, dtype=int)])
    record = dict(q=q, internal_configuration_dimension=32 + q, internal_phase_dimension=2 * (32 + q),
                  max_inclusion_error=max(errors), minimum_sampled_rank_ratio=min(ranks),
                  basis_sha256=base.array_sha(np.asarray(basis)), directions='scaled training-PCA columns 32:32+q (standardized_linear), as in nested_head.build',
                  column_norms=np.linalg.norm(np.asarray(basis), axis=0).tolist(), rank_ratio_definition='min over untransformed and mesh-transformed joint Jacobian',
                  y_starts='affine projection start; zero start; six fixed training codes carry y=0')
    return previous.numerical_bank(candidate), record


def retained_nested40(full, inputs):
    endpoint = tree_from_npz(inputs / 'trained_nested40.npz')
    with np.load(inputs / 'initializer_trained_nested40.npz') as f:
        linear, center = f['linear'], f['center']
    transform = jnp.asarray(full['transform']); latent = endpoint['p']['linear'].shape[1]
    candidate = dict(full, p=base.transform_head(endpoint['p'], transform), frozen=endpoint['frozen'],
                     common_inverse=jnp.linalg.pinv(transform @ jnp.asarray(linear[:, :latent])), common_center=transform @ jnp.asarray(center),
                     fixed_codes=endpoint['codes'][np.linspace(0, len(endpoint['codes']) - 1, FIXED, dtype=int)])
    return previous.numerical_bank(candidate)


# ----------------------------------------------------------------------------- linear arms
@partial(jax.jit, static_argnames=('integrator', 'steps', 'stride'))
def linear_stepping(kk, a0, b0, c, dt, integrator, *, steps, stride):
    """Time-stepped evolution of a'' = -c^2 K a in the bank; integrator 0 = Crank-Nicolson, 1 = RK4."""
    dim = a0.shape[0]
    stiffness = c * c * kk
    if integrator == 0:
        aa = jnp.block([[jnp.zeros((dim, dim)), jnp.eye(dim)], [-stiffness, jnp.zeros((dim, dim))]])
        propagator = jnp.linalg.solve(jnp.eye(2 * dim) - dt / 2 * aa, jnp.eye(2 * dim) + dt / 2 * aa)
        def step(state, _):
            state = propagator @ state
            return state, None
    else:
        def rhs(state):
            return jnp.concatenate((state[dim:], -stiffness @ state[:dim]))
        def step(state, _):
            k1 = rhs(state); k2 = rhs(state + dt / 2 * k1); k3 = rhs(state + dt / 2 * k2); k4 = rhs(state + dt * k3)
            return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, None
    def block(state, _):
        state, _ = jax.lax.scan(step, state, None, length=stride)
        return state, state
    initial = jnp.concatenate((a0, b0))
    _, states = jax.lax.scan(block, initial, None, length=steps // stride)
    states = jnp.concatenate((initial[None], states))
    return states[:, :dim], states[:, dim:]


def linear_query(name, bank, supplied, grid, cfg, integrator=None):
    """Timing contract of modal_projection.device_query, for POD arms and stepped bank variants."""
    u0, v0, c = supplied
    times = jnp.arange(round(cfg['end_time'] / cfg['observation_dt']) + 1) * cfg['observation_dt']
    jax.block_until_ready(times)
    t0 = time.perf_counter()
    if integrator is None:
        aux = modal.linear_evolution(bank, bank['prepared'], u0, v0, c, times)
        steps = stride = 0
    else:
        dt = cfg['head_dt']; steps, stride = round(cfg['end_time'] / dt), round(cfg['observation_dt'] / dt)
        a0 = bank['g'].T @ (bank['mass'] * u0.ravel()); b0 = bank['g'].T @ (bank['mass'] * v0.ravel())
        a, b = linear_stepping(bank['k'], a0, b0, c, dt, integrator, steps=steps, stride=stride)
        aux = dict(coefficients=a, velocity_coefficients=b, a0=a0, b0=b0)
    jax.block_until_ready(aux); t1 = time.perf_counter()
    u, v = modal.decode_coefficients(aux['coefficients'], aux['velocity_coefficients'], bank['g'])
    u, v = u.reshape(-1, *grid.shape), v.reshape(-1, *grid.shape)
    jax.block_until_ready((u, v)); t2 = time.perf_counter()
    u, v = np.asarray(u), np.asarray(v); t3 = time.perf_counter(); aux = jax.tree.map(np.asarray, aux)
    dim = bank['g'].shape[1]
    record = dict(method=name, setting=0. if integrator is None else cfg['head_dt'], boundary=grid.bx, intervals=grid.n,
                  steps=steps, observation_stride=stride, internal_configuration_dimension=dim, internal_phase_dimension=2 * dim,
                  integrator='exact modal' if integrator is None else ('crank_nicolson' if integrator == 0 else 'rk4'),
                  seconds=dict(initialization_and_parameter_projection=0., evolution=t1 - t0, dense_device_output=t2 - t1, complete_device_query=t2 - t0),
                  timer_components_note='evolution includes source projection, speed-dependent preparation and propagation',
                  post_timer_host_transfer_seconds=t3 - t2, device_plus_output_transfer_seconds=t3 - t0,
                  output_sha256=dict(u=base.array_sha(u), v=base.array_sha(v)), output_bytes=u.nbytes + v.nbytes,
                  completed=bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v))))
    return u, v, record, dict(coefficients=aux['coefficients'], velocity_coefficients=aux['velocity_coefficients'], projection=aux)


def reduced_energy_drift(a, b, kk, c):
    e = .5 * (np.sum(b * b, axis=1) + c * c * np.einsum('ti,ij,tj->t', a, kk, a))
    return dict(reduced_energy=e.tolist(), max_relative_drift=float(np.max(abs(e / e[0] - 1.))))


# ----------------------------------------------------------------------------- coarse FOM
@partial(jax.jit, static_argnames=('coarse', 'fine'))
def coarse_dst(u0, v0, c, times, coarse, fine):
    ratio = fine // coarse
    sl = slice(ratio - 1, None, ratio)
    uc, vc = base.spectral_propagate(u0[sl, sl], v0[sl, sl], c, times)
    return dst_prolong(uc, fine), dst_prolong(vc, fine)


def coarse_query(supplied, grid, cfg, coarse):
    u0, v0, c = supplied
    times = jnp.arange(round(cfg['end_time'] / cfg['observation_dt']) + 1) * cfg['observation_dt']
    jax.block_until_ready(times)
    t0 = time.perf_counter()
    u, v = coarse_dst(u0, v0, c, times, coarse, grid.n)
    jax.block_until_ready((u, v)); t2 = time.perf_counter()
    u, v = np.asarray(u), np.asarray(v); t3 = time.perf_counter()
    record = dict(method=f'dst_coarse{coarse}', setting=0., boundary=grid.bx, intervals=grid.n, steps=0, observation_stride=0,
                  coarse_intervals=coarse, seconds=dict(initialization_and_parameter_projection=0., evolution=t2 - t0, dense_device_output=0., complete_device_query=t2 - t0),
                  timer_components_note='evolution includes nodal restriction, coarse DST propagation and sine-interpolant prolongation of every output',
                  post_timer_host_transfer_seconds=t3 - t2, device_plus_output_transfer_seconds=t3 - t0,
                  output_sha256=dict(u=base.array_sha(u), v=base.array_sha(v)), output_bytes=u.nbytes + v.nbytes,
                  completed=bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v))))
    return u, v, record, {}


# ----------------------------------------------------------------------------- decomposition
@partial(jax.jit, static_argnames=('iterations',))
def best_found_fits(bank, targets, velocities, scale, *, iterations):
    n, k = len(targets), bank['p']['linear'].shape[1]
    affine = (targets - bank['common_center']) @ bank['common_inverse'].T
    starts = jnp.concatenate((affine[:, None], jnp.zeros_like(affine)[:, None], jnp.broadcast_to(bank['fixed_codes'], (n, FIXED, k))), axis=1)
    raw = fit_batch(bank['p'], bank['frozen'], jnp.repeat(targets, STARTS, axis=0), jnp.full(n * STARTS, scale), starts.reshape(-1, k), kind='mlp', iterations=iterations)
    zz, objective, gradient, counts, damping, stationarity, rank, finite = [x.reshape(n, STARTS, *x.shape[1:]) for x in raw]
    selected = jnp.argmin(jnp.where(finite, objective, jnp.inf), axis=1)
    z = zz[jnp.arange(n), selected]
    def tangent(z, velocity):
        fun = lambda x: head_apply(bank['p'], bank['frozen'], x, 'mlp')
        a = fun(z); jac = jax.jacfwd(fun)(z)
        q, r = jnp.linalg.qr(jac, mode='reduced')
        w = jax.scipy.linalg.solve_triangular(r, q.T @ velocity, lower=False)
        return a, jac @ w
    a, b = jax.vmap(tangent)(z, velocities)
    pick = lambda x: x[jnp.arange(n), selected]
    return a, b, z, dict(objective=pick(objective), gradient=pick(gradient), iterations=pick(counts), stationarity=pick(stationarity),
                         rank_ratio=pick(rank), finite=pick(finite), selected=selected)


def decomposition(models, linear_banks, su, sv, grid, c, cfg):
    """Bank/POD floors and best-found manifold fits against the truth at all observation times."""
    mass = jnp.asarray(grid.mass().ravel())
    flat_u, flat_v = jnp.asarray(su.reshape(len(su), -1)), jnp.asarray(sv.reshape(len(sv), -1))
    rows = {}
    for name, bank in linear_banks.items():
        au, av = flat_u @ (mass[:, None] * bank['g']), flat_v @ (mass[:, None] * bank['g'])
        pu, pv = np.asarray(au @ bank['g'].T).reshape(su.shape), np.asarray(av @ bank['g'].T).reshape(sv.shape)
        rows[name] = dict(layer='projection_floor', dimension=int(bank['g'].shape[1]), metrics=metrics(pu, pv, su, sv, grid, c, cfg))
    bank = next(iter(models.values()))
    targets = flat_u @ (mass[:, None] * bank['g']); velocities = flat_v @ (mass[:, None] * bank['g'])
    scale = jnp.sqrt(jnp.sum(mass * flat_u[0] ** 2))
    for name, model in models.items():
        a, b, z, fits = best_found_fits(model, targets, velocities, scale, iterations=cfg['fit_iterations'])
        pu, pv = np.asarray(a @ model['g'].T).reshape(su.shape), np.asarray(b @ model['g'].T).reshape(sv.shape)
        fits = jax.tree.map(np.asarray, fits)
        stationary = fits['finite'] & (fits['rank_ratio'] > 1e-8) & (fits['gradient'] <= 1e-7) & ((fits['stationarity'] <= 1e-6) | (fits['objective'] <= 1e-20))
        rows[name] = dict(layer='best_found', dimension=int(model['p']['linear'].shape[1]), metrics=metrics(pu, pv, su, sv, grid, c, cfg),
                          stationary_count=int(np.sum(stationary)), fit_count=int(len(stationary)),
                          selected_objectives=fits['objective'].tolist(), selected_gradients=fits['gradient'].tolist(),
                          selected_iterations=fits['iterations'].tolist(), selected_starts=fits['selected'].tolist(),
                          velocity_note='tangent least squares at the fitted code; displacement fit is the retained 8-start trust-region LM')
    return base.clean(rows)


# ----------------------------------------------------------------------------- dispatch
def query(name, supplied, c, grid, cfg, ctx):
    if name == 'head_q0':
        u, v, row, aux = ar.query('chol_guard', cfg['head_dt'], supplied, c, grid, cfg, ctx['models']['head_q0'])
        row.update(method=name, q=0, internal_configuration_dimension=32, internal_phase_dimension=64)
        return u, v, row, aux
    if name == 'trained_nested40':
        u, v, row, aux = ar.query('trained_nested40', cfg['head_dt'], supplied, c, grid, cfg, ctx['models'][name])
        row.update(q=8, directions='scaled training-PCA columns 32:40 (retained selection)')
        return u, v, row, aux
    if name.startswith('nested_q'):
        q = int(name.removeprefix('nested_q'))
        u, v, row, aux = ar.query('chol_guard', cfg['head_dt'], supplied, c, grid, cfg, ctx['models'][name])
        row.update(method=name, q=q, internal_configuration_dimension=32 + q, internal_phase_dimension=2 * (32 + q))
        return u, v, row, aux
    if name == 'linear_bank64':
        u, v, row, aux = modal.device_query('linear_bank64', ctx['bank'], supplied, grid, cfg)
        row.update(q=64, integrator='exact modal')
        return u, v, row, aux
    if name in ('linear_bank64_cn', 'linear_bank64_rk4'):
        bank = dict(ctx['bank'], prepared=ctx['bank']['prepared_l2'])
        u, v, row, aux = linear_query(name, bank, supplied, grid, cfg, integrator=0 if name.endswith('cn') else 1)
        row.update(q=64)
        return u, v, row, aux
    if name.startswith('pod_k'):
        return linear_query(name, ctx['pod'][name], supplied, grid, cfg)
    if name == 'rk4_fom':
        u, v, row, aux = previous.device_query('rk4', cfg['fom_cfl'], supplied, float(c), grid, cfg)
        row.update(method=name, cfl=cfg['fom_cfl'])
        return u, v, row, aux
    if name.startswith('dst_coarse'):
        return coarse_query(supplied, grid, cfg, int(name.removeprefix('dst_coarse')))
    if name == 'dst' or name.startswith('cg_') or name.startswith('cgdt_'):
        setting = 0. if name == 'dst' else float(name.split('tol_')[1]) if name.startswith('cgdt_') else float(name.split('_')[1])
        return ar.query(name, setting, supplied, c, grid, cfg, ctx['bank'])
    raise ValueError(name)


def arm_list(cfg, n):
    arms = ['head_q0', 'trained_nested40'] + [f'nested_q{q}' for q in cfg['nested_q']]
    arms += ['linear_bank64', 'linear_bank64_cn', 'linear_bank64_rk4'] + [f'pod_k{k}' for k in cfg['pod_ranks']]
    arms += ['dst', 'rk4_fom'] + [f'cg_{tol:g}' for tol in cfg['cg_tolerances']]
    arms += [f"cgdt_{a['dt']:g}_tol_{a['tolerance']:g}" for a in cfg['cg_timestep_arms']]
    if n > cfg['coarse_fom_intervals']:
        arms.append(f"dst_coarse{cfg['coarse_fom_intervals']}")
    return arms


def is_rom(name):
    return name.startswith(('head_', 'trained_', 'nested_', 'linear_', 'pod_'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--gates', type=Path, default=None)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text()); out = args.out
    out.mkdir(parents=True, exist_ok=False)
    meta = provenance(); print(json.dumps(meta), flush=True)
    assert meta['jax_backend'] == 'gpu' and meta['x64'] and meta['matmul_precision'] == 'highest'
    assert not cfg['final_test_opened'] and cfg['boundaries'] == ['dirichlet']
    frozen = json.loads((base.FRESH / 'FROZEN-MATH.json').read_text())
    for name, expected in frozen['sha256'].items():
        assert base.sha(base.FRESH / name) == expected, name
    n = cfg['mesh']; grid = Grid(n, 'dirichlet', 'dirichlet')
    data = dict(config=cfg, provenance=meta, frozen_mathematics=frozen, mesh=n,
                input_sha256={str(p.relative_to(args.inputs)): base.sha(p) for p in args.inputs.rglob('*') if p.is_file()},
                training={}, pod={}, mesh_audits={}, nested_models=[], references=[], invocations=[], warmups=[],
                decomposition=[], consistency=[], gates={}, final_test_opened=False, complete=False)
    save = lambda: base.save_json(out / 'result.json', data)
    # --- training ladder (PCA directions) and POD basis, both from the regenerated training data
    linear, center, manifest = dyn.regenerate_ladder(args.inputs, 'dirichlet', out, cfg)
    assert manifest['data_hashes_match'] and manifest['saved_initialization_matched'], manifest
    with np.load(out / 'training_ladder_dirichlet.npz') as f:
        ladder = {k: f[k] for k in ('basis', 'standardized_linear', 'center', 'scales', 'singular_values')}
    with np.load(args.inputs / 'initializer32.npz') as f:
        saved32 = f['linear']
    manifest['saved_initializer32_max_abs_difference'] = float(np.max(abs(ladder['standardized_linear'][:, :32] - saved32)))
    np.testing.assert_allclose(ladder['standardized_linear'][:, :32], saved32, atol=1e-12, rtol=0)
    with np.load(args.inputs / 'initializer_trained_nested40.npz') as f:
        saved40 = f['linear']
    manifest['saved_initializer40_max_abs_difference'] = float(np.max(abs(ladder['standardized_linear'][:, :40] - saved40[:, :40])))
    np.testing.assert_allclose(ladder['standardized_linear'][:, :40], saved40[:, :40], atol=1e-10, rtol=0)
    data['training'] = manifest; save()
    phi256, pod_manifest = build_pod(args.inputs, out, max(cfg['pod_ranks']), cfg)
    data['pod'] = pod_manifest; save()
    # --- models on this mesh
    full, models = previous.load_models(args.inputs, grid)
    bank = models[cfg['primary_method']]
    bank = dict(bank, prepared_l2=modal.prepare(bank, 'l2'))
    head32 = tree_from_npz(args.inputs / 'head32.npz')
    rng = np.random.default_rng(691217)
    head_models = {'head_q0': models[cfg['primary_method']], 'trained_nested40': retained_nested40(full, args.inputs)}
    for q in cfg['nested_q']:
        head_models[f'nested_q{q}'], record = nested_model(full, head32, ladder, q, rng)
        data['nested_models'].append(record)
    phi, kk_pod, pods, pod_audits = pod_banks(phi256, grid, cfg['pod_ranks'], bank['mass'], jnp.asarray(base.damping_ratio(grid, 1.).ravel()))
    eig = np.linalg.eigvalsh(np.asarray(bank['k']))
    bank_audits = dict(full['audits'], stiffness_max_eigenvalue=float(eig[-1]), stiffness_min_eigenvalue_numpy=float(eig[0]),
                       omega_max_dt_at_c1p15=float(1.15 * np.sqrt(eig[-1]) * cfg['head_dt']), rk4_linear_stability_bound=2 * np.sqrt(2.))
    assert eig[0] > 0, 'bank stiffness not positive definite'
    data['mesh_audits'] = dict(bank=bank_audits, bank_assembly_seconds=full['assembly_seconds_including_first_compile'], pod=pod_audits); save()
    np.savez_compressed(out / f'mesh_{n}.npz', g=np.asarray(bank['g']), mass=np.asarray(bank['mass']), stiffness=np.asarray(bank['k']),
                        damping=np.asarray(bank['d']), transform=full['transform'], pod_phi=np.asarray(phi), pod_stiffness=kk_pod)
    linear_banks = {'linear_bank64': bank, **pods}
    ctx = dict(models=head_models, bank=bank, pod=pods)
    arms = arm_list(cfg, n)
    data['arms'] = arms
    stiffness_of = {**{name: np.asarray(bank['k']) for name in ('linear_bank64', 'linear_bank64_cn', 'linear_bank64_rk4')},
                    **{name: np.asarray(b['k']) for name, b in pods.items()}}
    # --- cases
    for cohort in cfg['cohorts']:
        pars = parameter_rows(cohort['seed'], max(cohort['indices']) + 1)
        for ci in cohort['indices']:
            par = pars[ci]; case = f"{cohort['name']}_{ci}"; c = float(par[5])
            u0, v0 = localized_initial(grid, par); supplied = (u0, v0, jnp.asarray(par[5])); jax.block_until_ready(supplied)
            su, sv, _, _ = query('dst', supplied, c, grid, cfg, ctx)
            full_fields = case in cfg['full_field_cases']
            save_fields = lambda u, v: dict(u=u, v=v) if full_fields else dict(u=restrict(u, n, cfg['saved_intervals'], 'dirichlet'), v=restrict(v, n, cfg['saved_intervals'], 'dirichlet'))
            np.savez_compressed(out / f'reference_{n}_{case}.npz', u0=np.asarray(u0), v0=np.asarray(v0), parameters=par, **save_fields(su, sv))
            data['references'].append(dict(intervals=n, case=case, cohort=cohort['name'], seed=cohort['seed'], parameters=par, retained_subset=ci in (0, 1),
                                           input_sha256=dict(u=base.array_sha(np.asarray(u0)), v=base.array_sha(np.asarray(v0))),
                                           reference_sha256=dict(u=base.array_sha(su), v=base.array_sha(sv)), reference_kind='exact semidiscrete DST',
                                           fields_saved='full' if full_fields else f"common {cfg['saved_intervals']} grid"))
            for name in arms:
                print('warmup', n, case, name, flush=True)
                _, _, row, _ = query(name, supplied, c, grid, cfg, ctx)
                data['warmups'].append(dict(case=case, **row))
            first = {}
            for rep in range(cfg['repetitions']):
                for order, name in enumerate(arms if rep % 2 == 0 else arms[::-1]):
                    print('timed', n, case, rep, name, flush=True); base.burn()
                    u, v, row, aux = query(name, supplied, c, grid, cfg, ctx)
                    ident = f'{n}_{case}_{name}'
                    common = Grid(cfg['saved_intervals'], 'dirichlet', 'dirichlet')
                    row.update(case=case, cohort=cohort['name'], seed=cohort['seed'], parameters=par, repetition=rep, order=order,
                               invocation_id=f'{ident}_{rep}', comparison_eligible=True, retained_subset=ci in (0, 1),
                               same_grid_discrepancy=metrics(u, v, su, sv, grid, par[5], cfg),
                               common_grid_discrepancy=metrics(*(restrict(x, n, common.n, 'dirichlet') for x in (u, v)),
                                                               *(restrict(x, n, common.n, 'dirichlet') for x in (su, sv)), common, par[5], cfg),
                               field_artifact=ident + '.npz')
                    if name in stiffness_of and 'coefficients' in aux:
                        row['reduced_energy'] = reduced_energy_drift(np.asarray(aux['coefficients']), np.asarray(aux['velocity_coefficients']), stiffness_of[name], c)
                    if rep == 0:
                        # ROM arms: coefficients only (fields = coefficients @ g.T, reconstructed by the audit from
                        # mesh_<n>.npz).  FOM arms: full fields on the designated case, common-grid fields otherwise.
                        # The dst arm is the reference itself and is saved once in reference_<n>_<case>.npz.
                        arrays = {}
                        if is_rom(name):
                            arrays.update(coefficients=np.asarray(aux['coefficients']), velocity_coefficients=np.asarray(aux['velocity_coefficients']))
                            if 'rollout' in aux:
                                arrays.update({'rollout_' + k: np.asarray(x) for k, x in aux['rollout'].items()})
                        elif name != 'dst':
                            arrays.update(save_fields(u, v))
                        if arrays:
                            np.savez_compressed(out / (ident + '.npz'), **arrays)
                        first[ident] = row['output_sha256']
                    else:
                        assert first[ident] == row['output_sha256'], 'Nondeterministic output: ' + ident
                    if name == 'dst':
                        row['field_artifact'] = f'reference_{n}_{case}.npz'
                    row['artifact_relation'] = 'Actual first timed output' if rep == 0 else 'Byte-identical full timed fields, both hashes checked'
                    if not name.startswith(('head_', 'trained_', 'nested_')):
                        assert row['completed'], 'non-head arm did not complete: ' + ident
                    if name.startswith(('cg_', 'cgdt_')):
                        assert row['cg_all_converged'] and row['cg_cap_exits'] == 0, 'CG did not meet its tolerance: ' + ident
                    data['invocations'].append(row); save()
            print('decomposition', n, case, flush=True)
            data['decomposition'].append(dict(intervals=n, case=case, layers=decomposition(head_models, linear_banks, su, sv, grid, par[5], cfg))); save()
            # consistency: nested_q8 vs trained_nested40; nested_q32 vs linear_bank64_rk4; rk4 vs exact modal
            fields = {}
            for name in ('trained_nested40', 'nested_q8', 'nested_q32', 'linear_bank64', 'linear_bank64_rk4', 'linear_bank64_cn'):
                if name in arms:
                    with np.load(out / f'{n}_{case}_{name}.npz') as f:
                        fields[name] = (f['coefficients'], f['velocity_coefficients'])
            def discrepancy(x, y):
                return float(max(np.max(abs(fields[x][0] - fields[y][0])) / np.max(abs(fields[y][0])), np.max(abs(fields[x][1] - fields[y][1])) / np.max(abs(fields[y][1]))))
            pairs = [p for p in (('nested_q8', 'trained_nested40'), ('nested_q32', 'linear_bank64_rk4'), ('linear_bank64_rk4', 'linear_bank64'), ('linear_bank64_cn', 'linear_bank64')) if p[0] in fields and p[1] in fields]
            data['consistency'].append(dict(intervals=n, case=case, **{f'{a}_vs_{b}': discrepancy(a, b) for a, b in pairs},
                                            note='relative max-abs coefficient discrepancy over all observation times; reported, not gated'))
            save()
            del su, sv, u, v
    # --- retained-value gates (same A100 class as the archived jobs)
    if args.gates is not None:
        gates = json.loads(args.gates.read_text())['gates']; checks = []
        for g in gates:
            if g['intervals'] != n:
                continue
            rows = [r for r in data['invocations'] if r['case'] == g['case'] and r['method'] == g['method'] and r['repetition'] == 0]
            if not rows:
                continue
            row = rows[0]
            extra = {}
            if g['metric'] == 'selected_fit_objective':
                # DESIGN A4: tie-invariant. Eight starts reach the same minimum; argmin breaks the tie
                # by index, so the selected START may differ between identical runs while the objective
                # reached does not. This is the strict check.
                cf = row['cold_fit']; got = cf['objective'][cf['selected']]; tolerance = 1e-9
                extra = dict(measured_selected_start=int(cf['selected']),
                             selected_start_matches_archive=int(cf['selected']) == g.get('archived_selected_start'))
            else:
                got = row['same_grid_discrepancy'][g['metric']]['max_initial_normalized']
                # A trajectory error inherits the start-tie flip, so it is gated loosely for the
                # fit-based arms and tightly for the deterministic linear ones.
                tolerance = 1e-7 if 'cold_fit' in row else 1e-9
            rel = abs(got - g['value']) / abs(g['value'])
            checks.append(dict(**g, **extra, measured=got, relative_difference=rel, tolerance=tolerance, passed=rel <= tolerance))
        data['gates'] = dict(retained_value_checks=checks, all_passed=all(x['passed'] for x in checks), count=len(checks))
        print('gates', json.dumps(data['gates']), flush=True); save()
        if not data['gates']['all_passed']:
            raise RuntimeError('Retained-value gate failed; see result.json gates')
    data['complete'] = True; data['output_sha256'] = {p.name: base.sha(p) for p in out.glob('*.npz')}; save()
    print('w_ladder_complete', flush=True)


if __name__ == '__main__':
    main()
