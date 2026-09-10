"""Bounded check of the post-smoke reconstruction/Jacobian diagnostic addition."""
import argparse
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from compare import load_models, Queries, representation_diagnostic
from weak import numerical_rule
from physics import Grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads((args.archive/'out/smoke/config.json').read_text())
    models = load_models(args.archive/'inputs', cfg, 'dirichlet')
    grid = Grid(16)
    with np.load(args.archive/'out/smoke/comparison/quadrature/cp_16_4/rule.npz') as a:
        raw = {key: a[key] for key in a.files}
    rule = numerical_rule(models['cp']['params'], models['cp']['config'], raw)
    queries = Queries(cfg, grid, models, {('cp', 4): rule})
    setting = {'method': 'cp', 'id': 'cp_eq4_cap2_tol0.001_dt0.005', 'multiplier': 4, 'cap': 2, 'tol': .001, 'dt': .005}
    path = args.archive/'out/smoke/comparison/fields/evaluation_16_0_cp_eq4_cap2_tol0.001_dt0.005.npz'
    with np.load(path) as a:
        ut, vt, speed = jnp.asarray(a['truth_u']), jnp.asarray(a['truth_v']), jnp.asarray(a['speed'])
    supplied = (ut[0], vt[0], speed)
    _, _, _, aux = queries.query(setting, supplied)
    result = representation_diagnostic(queries, setting, supplied, (ut, vt), aux)
    assert len(result['snapshot_fits']) == len(ut)
    assert len(result['residual_plus_jacobian_seconds']) == 7
    assert all(len(r['weak_tangent_singular_values']) == cfg['latent'] for r in result['snapshot_fits'])
    print('WAVE_RECONSTRUCTION_GEOMETRY_DIAGNOSTIC_PASSED', flush=True)


if __name__ == '__main__':
    main()
