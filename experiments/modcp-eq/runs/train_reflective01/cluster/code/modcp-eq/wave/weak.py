"""Decoder-output EQ and overdetermined weak first-order wave CN residuals."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import jax
import jax.numpy as jnp
from common.decoders import decode_points, decode_grid, prepare_points, decode_cached
from common.quadrature import fit_quadrature
from common.lm import make_lm
from physics import Grid, smooth_tests, face_data, active_fields
from data import save_json


def fit_integrands(values, tests, masses, candidates, count):
    """Row RMS normalization is independent of potentially cancelling targets."""
    total = float(masses.sum())
    target = np.einsum('pm,spc,p->scm', tests, values, masses, optimize=True).reshape(-1)
    scale = np.sqrt(np.einsum('pm,spc,p->scm', tests*tests, values*values, masses, optimize=True)/total).reshape(-1)
    floor = max(float(np.max(scale))*1e-12, 1e-14)
    scale = np.maximum(scale, floor)
    design = np.einsum('pm,spc->scmp', tests[candidates], values[:, candidates], optimize=True).reshape(len(target), len(candidates))
    design /= scale[:, None]
    target /= scale
    design = np.vstack((np.ones((1, len(candidates))), design))
    target = np.r_[total, target]
    ids, weights, info = fit_quadrature(design, target, count, candidate_ids=candidates)
    info['expected_measure'] = total
    info['measure_error'] = float(abs(weights.sum()-total))
    # Report every original, unsummed weak row. This is not a residual-snapshot fit.
    info['row_scaling'] = 'full_grid_mass_weighted_RMS_of_decoder_output_times_test'
    return ids, weights, info


def build_rule(params, codes, dc, grid, cfg, multiplier, out):
    """NNLS weights refitted independently for each checkpoint, mesh and rule."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = cfg['weak_modes']*multiplier
    phi, eigen, modes = smooth_tests(grid, cfg['weak_modes'])
    points = grid.coordinates().reshape(-1, 2)
    mass = grid.mass().ravel()
    chosen = np.unique(np.linspace(0, len(codes)-1, min(cfg['eq_fit_codes'], len(codes)), dtype=int))
    # Architecture and coordinates only; no PDE parameter or time conditions.
    decoder = jax.jit(lambda p, z: jax.lax.map(lambda zz: decode_grid(p, zz, grid.n, dc), z))
    fields = np.asarray(decoder(params, jnp.asarray(codes[chosen])))
    if grid.bx == 'dirichlet':
        fields = fields[:, 1:-1, 1:-1]
    values = fields.reshape(len(chosen), -1, 2)
    side = min(grid.shape[0], int(np.floor(np.sqrt(cfg['eq_candidate_count']))))
    axis = np.unique(np.linspace(0, grid.shape[0]-1, side, dtype=int))
    candidates = (axis[:, None]*grid.shape[1]+axis[None, :]).ravel()
    if len(candidates) < m:
        candidates = np.arange(len(points))
    print('EQ_VOLUME', dc.architecture, grid.bx, grid.n, m, flush=True)
    ids, weights, info = fit_integrands(values, phi, mass, candidates, min(m, len(candidates)))
    face_rules, face_infos = [], []
    faces = face_data(grid, modes)
    fbudget = min(grid.n+1, multiplier*max(int(np.max(modes)), 4))
    for fi, (fxy, fw, ft) in enumerate(faces):
        fv = np.asarray(jax.jit(lambda p, z, xy: jax.vmap(lambda zz: decode_points(p, zz, xy, dc))(z))(
            params, jnp.asarray(codes[chosen]), jnp.asarray(fxy)))
        print('EQ_FACE', dc.architecture, grid.n, multiplier, fi, fbudget, flush=True)
        inds, w, diagnostics = fit_integrands(fv, ft, fw, np.arange(len(fxy)), fbudget)
        face_rules.append((fxy[inds], w, ft[inds]))
        face_infos.append(diagnostics)
    rule = {'xy': points[ids], 'active_ids': ids, 'weight': weights, 'test': phi[ids], 'eigen': eigen,
            'modes': modes, 'face_xy': np.concatenate([r[0] for r in face_rules]) if face_rules else np.empty((0, 2)),
            'face_weight': np.concatenate([r[1] for r in face_rules]) if face_rules else np.empty(0),
            'face_test': np.concatenate([r[2] for r in face_rules]) if face_rules else np.empty((0, len(eigen)))}
    np.savez_compressed(out/'rule.npz', **rule)
    audit = {'architecture': dc.architecture, 'intervals': grid.n, 'weak_modes': len(eigen),
             'fit_code_indices': chosen.tolist(), 'fit_rule': 'training_decoder_outputs_only',
             'volume': info, 'faces': face_infos, 'volume_target': m,
             'face_target_each': fbudget if faces else 0,
             'total_selected_nodes_including_face_duplicates': len(weights)+sum(len(r[1]) for r in face_rules),
             'corner_rule': 'Separate physical faces retain both corner half-weights.'}
    # Independent nearby latent perturbations expose fit over-specialization.
    zcheck = np.asarray(codes[chosen[:min(4, len(chosen))]])
    zcheck = zcheck+.01*np.sin(np.arange(zcheck.size).reshape(zcheck.shape)+.3)
    actual = np.asarray(decoder(params, jnp.asarray(zcheck)))
    if grid.bx == 'dirichlet':
        actual = actual[:, 1:-1, 1:-1]
    actual = actual.reshape(len(zcheck), -1, 2)
    truth = np.einsum('pm,spc,p->smc', phi, actual, mass, optimize=True)
    pred = np.einsum('pm,spc,p->smc', phi[ids], actual[:, ids], weights, optimize=True)
    denom = np.maximum(np.linalg.norm(truth, axis=1), 1e-14)
    audit['perturbed_mass_moment_relative'] = (np.linalg.norm(pred-truth, axis=1)/denom).tolist()
    save_json(out/'audit.json', audit)
    return rule, audit


def numerical_rule(params, dc, rule):
    return {'cache': prepare_points(params, jnp.asarray(rule['xy']), dc),
            'projection': jnp.asarray(rule['test'].T*rule['weight'][None, :]),
            'face_cache': prepare_points(params, jnp.asarray(rule['face_xy']), dc),
            'face_projection': jnp.asarray(rule['face_test'].T*rule['face_weight'][None, :]),
            'fit_weights': jnp.asarray(rule['weight']/max(float(np.sum(rule['weight'])), 1e-30)),
            'eigen': jnp.asarray(rule['eigen']), 'ids': jnp.asarray(rule['active_ids'])}


def moments(params, z, rule, dc):
    q = decode_cached(params, z, rule['cache'], dc)
    mass = rule['projection']@q
    if rule['face_projection'].shape[1]:
        faces = rule['face_projection']@decode_cached(params, z, rule['face_cache'], dc)
    else:
        faces = jnp.zeros_like(mass)
    return mass, faces


def make_query(dc, grid, cfg, cap, dt):
    """No physical descriptors/time enter decoder or latent initialization."""
    def residual(z, params, rule, oldmass, oldface, speed, scales):
        mass, face = moments(params, z, rule, dc)
        ru = mass[:, 0]-oldmass[:, 0]-dt*(mass[:, 1]+oldmass[:, 1])/2
        rv = (mass[:, 1]-oldmass[:, 1]+dt*speed*(face[:, 1]+oldface[:, 1])/2 +
              dt*speed*speed*rule['eigen']*(mass[:, 0]+oldmass[:, 0])/2)
        return jnp.concatenate((ru/scales[0], rv/scales[1]))
    solve = make_lm(residual, cap=cap)
    def fitres(z, params, rule, target, scales):
        # Pointwise FIELD fit is valid initialization; PDE remains weak.
        values = decode_cached(params, z, rule['cache'], dc)
        weights = jnp.sqrt(jnp.maximum(rule['fit_weights'], 0.))
        return ((values-target)/scales[None, :]*weights[:, None]).ravel()
    fit = make_lm(fitres, cap=cfg['initial_fit_cap'])
    stride = int(round(cfg['observation_dt']/dt))
    blocks = int(round(cfg['end_time']/cfg['observation_dt']))

    @jax.jit
    def initialize(params, rule, u0, v0, starts, scales):
        values = jnp.stack((u0.ravel()[rule['ids']], v0.ravel()[rule['ids']]), -1)
        # Training-code candidate selection uses sampled supplied fields only.
        results = jax.lax.map(lambda z: fit(z, (params, rule, values, scales), cfg['initial_fit_tolerance']), starts)
        norms = results[4]
        valid = jnp.isfinite(norms) & (results[2] != 3) & (results[2] != 5)
        selected = jnp.argmin(jnp.where(valid, norms, jnp.inf))
        return results[0][selected], {'iterations': results[1], 'reason': results[2],
                                      'stationarity': results[3], 'residual_norm': results[4], 'selected': selected}

    @jax.jit
    def evolve(params, rule, z0, speed, scales, tolerance):
        def block(carry, _):
            z, alive = carry
            def one(s, _):
                z, alive = s
                oldmass, oldface = moments(params, z, rule, dc)
                zn, it, reason, stationarity, rn = solve(z, (params, rule, oldmass, oldface, speed, scales), tolerance)
                valid = jnp.all(jnp.isfinite(zn)) & (reason != 3) & (reason != 5)
                alive = alive & valid
                return (zn, alive), (it, reason, stationarity, rn, alive)
            (z, alive), diagnostic = jax.lax.scan(one, (z, alive), None, length=stride)
            return (z, alive), (z, diagnostic)
        _, (z, diagnostics) = jax.lax.scan(block, (z0, jnp.asarray(True)), None, length=blocks)
        return jnp.concatenate((z0[None], z)), tuple(x.reshape(-1) for x in diagnostics)

    @jax.jit
    def reconstruct(params, codes):
        fields = jax.lax.map(lambda z: decode_grid(params, z, grid.n, dc), codes)
        if grid.bx == 'dirichlet':
            fields = fields[:, 1:-1, 1:-1]
        return fields[..., 0], fields[..., 1]
    return initialize, evolve, reconstruct, residual
