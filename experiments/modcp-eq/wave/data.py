"""Seeded on-cluster data generation with fresh reference diagnostics."""
import hashlib
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
from physics import (Grid, parameter_rows, localized_initial, spectral_propagate,
                     integrate_balance, energy, damping_ratio, metrics)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return clean(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def reference(grid, parameter, cfg, refine=False):
    u0, v0 = localized_initial(grid, parameter)
    times = jnp.arange(int(round(cfg['end_time']/cfg['observation_dt']))+1)*cfg['observation_dt']
    speed = jnp.asarray(parameter[5])
    if grid.bx == 'dirichlet':
        u, v = spectral_propagate(u0, v0, speed, times)
        balance = (energy(u, v, grid, speed)-energy(u0, v0, grid, speed))/energy(u0, v0, grid, speed)
        audit = {'kind': 'exact_semidiscrete_sine', 'max_relative_energy_drift': float(jnp.max(jnp.abs(balance))),
                 'temporal_reference_passed': bool(jnp.max(jnp.abs(balance)) < 1e-10)}
    else:
        observations = len(times)-1
        def run(cfl):
            stride = int(np.ceil(cfg['observation_dt']*float(parameter[5])/(cfl*grid.h)))
            dt = cfg['observation_dt']/stride
            u, v, flux = integrate_balance(u0, v0, speed, dt, grid=grid, steps=observations*stride, stride=stride)
            e0 = energy(u0, v0, grid, speed)
            balance = (energy(u, v, grid, speed)+flux-e0)/e0
            invariant = jnp.sum(jnp.asarray(grid.mass())*(v+damping_ratio(grid, speed)*u), axis=(-2, -1))
            return u, v, {'dt': dt, 'max_relative_energy_balance': float(jnp.max(jnp.abs(balance))),
                          'max_invariant_drift': float(jnp.max(jnp.abs(invariant-invariant[0])))}
        u, v, audit = run(cfg['reference_cfl'])
        audit['kind'] = 'fresh_RK4_with_balance'
        audit['temporal_reference_passed'] = audit['max_relative_energy_balance'] < 1e-5 and audit['max_invariant_drift'] < 1e-10
        if refine:
            uf, vf, fine = run(cfg['reference_refinement_cfl'])
            discrepancy = metrics(u, v, uf, vf, grid, speed)
            audit['temporal_refinement'] = clean(discrepancy)
            audit['fine'] = fine
            audit['temporal_reference_passed'] &= float(discrepancy['maximum']) <= cfg['reference_temporal_target']
            u, v = uf, vf
    if not audit['temporal_reference_passed'] or not bool(jnp.all(jnp.isfinite(u)) & jnp.all(jnp.isfinite(v))):
        raise RuntimeError(f'Fresh reference failed: {clean(audit)}')
    return u, v, audit


def training_data(cfg, bc, out):
    """One trajectory at a time into a regenerable local-job memmap."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid(cfg['train_intervals'], bc, bc)
    pars = parameter_rows(cfg['train_seed'], cfg['train_count'])
    nt = int(round(cfg['end_time']/cfg['observation_dt']))+1
    shape = (len(pars)*nt, int(np.prod(grid.shape)), 2)
    path = out/'training_fields.npy'
    # Every cluster attempt regenerates from its recorded seed. A resumed stage
    # can reuse only this exact attempt's checksummed complete data.
    manifest_path = out/'training_data.json'
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['complete'] and manifest['seed'] == cfg['train_seed'] and manifest['intervals'] == grid.n and manifest['boundary'] == bc:
            return grid, np.load(path, mmap_mode='r'), np.asarray(manifest['fixed_component_scales']), manifest
    states = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=shape)
    audits, scales = [], []
    for i, p in enumerate(pars):
        print('DATA', bc, 'train', i, flush=True)
        u, v, audit = reference(grid, p, cfg)
        e0 = float(energy(u[0], v[0], grid, p[5]))
        us = float(jnp.sqrt(jnp.sum(jnp.asarray(grid.mass())*u[0]**2)))
        scales.append([us, np.sqrt(2*e0)])
        states[i*nt:(i+1)*nt] = np.stack((np.asarray(u).reshape(nt, -1), np.asarray(v).reshape(nt, -1)), -1)
        audits.append({'case': i, 'parameters': p.tolist(), **audit})
    states.flush()
    # Fixed TRAINING-only physical component normalization, shared all arms.
    fixed_scales = np.sqrt(np.mean(np.asarray(scales)**2, axis=0))
    manifest = {'complete': True, 'seed': cfg['train_seed'], 'intervals': grid.n,
                'boundary': bc, 'shape': shape, 'parameters': pars.tolist(), 'audits': audits,
                'fixed_component_scales': fixed_scales.tolist(), 'case_scales': scales,
                'array_sha256': array_sha(states), 'evaluation_generated': False}
    save_json(manifest_path, manifest)
    return grid, states, fixed_scales, manifest
