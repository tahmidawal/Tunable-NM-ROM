"""Exact periodic integer translations, fixed before opening any final cases.

One membership table is shared by bank, head, POD and every neural operator.
Translations never mix times, velocity components, viscosities or base cases.
"""
from __future__ import annotations
import hashlib
import numpy as np


def membership(base_count, n, copies, seed):
    assert copies >= 1 and copies <= n**3
    rng = np.random.default_rng(seed)
    rows = []
    for base in range(base_count):
        offsets = [(0, 0, 0)]
        while len(offsets) < copies:
            candidate = tuple(int(x) for x in rng.integers(0, n, 3))
            if candidate not in offsets:
                offsets.append(candidate)
        rows.extend((base, *offset) for offset in offsets)
    return np.asarray(rows, dtype=np.int64)


def augment(states, table):
    assert states.ndim == 6 and states.shape[2] == 3
    result = np.empty((len(table), *states.shape[1:]), dtype=np.float64)
    for index, (base, *offset) in enumerate(table):
        result[index] = np.roll(states[base], tuple(offset), axis=(-3, -2, -1))
    return result


def manifest(table, n, seed, base_states, base_parameters):
    digest = lambda x: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    return dict(kind='shared deterministic periodic integer translations',
                base_cases=len(base_states), augmented_cases=len(table), n=n,
                augmentation_seed=seed, membership=table.tolist(),
                membership_sha256=digest(table), base_states_sha256=digest(base_states),
                base_parameter_sha256=digest(base_parameters),
                columns=['base_case', 'roll_x', 'roll_y', 'roll_z'],
                applies_to=['learned_bank', 'nonlinear_head', 'POD', 'FNO', 'U-Net', 'DeepONet', 'Transolver'],
                viscosity_rule='base-case viscosity unchanged',
                held_out_rule='development and final cohorts unchanged',
                includes_original=True, snapshots_per_case=int(base_states.shape[1]))


def verify(n=12):
    """Production CNAB2 plus independent advective/RK4 symmetry checks."""
    import jax.numpy as jnp
    import ns3d_fom as F
    import ns3d_independent as I
    relative = lambda a, b: float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b), 1e-300))
    offset = (2, 3, -1)
    shift = lambda x: np.roll(np.asarray(x), offset, axis=(-3, -2, -1))
    par = F.parameters(202609240, 1)[0]
    shifted_par = par.copy(); shifted_par[:3] = (par[:3]+np.asarray(offset)/n) % 1
    geom = F.geometry(n); spec = I.setup(n)
    raw = F.initial_raw(n, par)
    u = F.initial(n, par); su = shift(u)
    checks = dict(initial_family=relative(F.initial(n, shifted_par), su),
        production_projection=relative(F.project_field(jnp.asarray(shift(raw)), geom), shift(F.project_field(jnp.asarray(raw), geom))),
        independent_projection=relative(I.physical(I.solenoidal(I.transform(shift(raw)), spec)), shift(I.physical(I.solenoidal(I.transform(raw), spec)))),
        production_nonlinearity=relative(F.ifft(F.nonlinear(F.fft(jnp.asarray(su)), geom)), shift(F.ifft(F.nonlinear(F.fft(jnp.asarray(u)), geom)))),
        independent_nonlinearity=relative(I.physical(I.advective(I.transform(su), spec)), shift(I.physical(I.advective(I.transform(u), spec)))))
    solver = F.make_solver(.002, 10, 2)
    checks['production_complete_trajectory'] = relative(solver(jnp.asarray(su), par[-1], geom), shift(solver(jnp.asarray(u), par[-1], geom)))
    checks['independent_complete_trajectory'] = relative(I.solve(su, par[-1], .002, 10, 2), shift(I.solve(u, par[-1], .002, 10, 2)))
    base = np.stack((np.asarray(solver(jnp.asarray(u), par[-1], geom)),)*2)
    table = membership(2, n, 4, 100)
    augmented = augment(base, table)
    checks['norm_preservation'] = float(np.max(np.abs(np.linalg.norm(augmented.reshape(8, -1), axis=1)/np.linalg.norm(base[0])-1)))
    return dict(passed=max(checks.values()) < 1e-10, relative_tolerance=1e-10, checks=checks,
                membership_deterministic=bool(np.array_equal(table, membership(2,n,4,100))),
                axes='last three spatial axes only', n=n)
