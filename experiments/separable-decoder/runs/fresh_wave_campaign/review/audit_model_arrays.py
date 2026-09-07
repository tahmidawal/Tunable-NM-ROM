"""Independent NumPy/SciPy check of exported neural banks and head derivatives.

This reads result artifacts only; it never imports fresh-wave implementation,
trains a model, or evolves a trajectory.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit


def bank_value(parameters, xy, reflective):
    phase = 2*np.pi*(xy@parameters['frequency'].T)
    x = np.concatenate((2*xy-1, np.sin(phase), np.cos(phase)), axis=-1)
    for layer in ('l1', 'l2'):
        x = x@parameters[f'p/{layer}/w']+parameters[f'p/{layer}/b']
        x = x*expit(x)
    x = x@parameters['p/out/w']+parameters['p/out/b']
    if reflective:
        x *= (np.sin(np.pi*xy[:, 0])*np.sin(np.pi*xy[:, 1]))[:, None]
    return x


def head_value_jacobian(parameters, z, kind):
    a = parameters['p/bias']+z@parameters['p/linear'].T
    jac = np.broadcast_to(parameters['p/linear'], (len(z), *parameters['p/linear'].shape)).copy()
    scale = float(parameters['frozen/output_scale'])
    if kind == 'quadratic':
        ii, jj = np.triu_indices(z.shape[-1])
        products = z[:, ii]*z[:, jj]
        derivative = np.zeros((len(z), len(ii), z.shape[-1]))
        for p, (i, j) in enumerate(zip(ii, jj)):
            derivative[:, p, i] += z[:, j]
            derivative[:, p, j] += z[:, i]
        a += scale*(products@parameters['p/quadratic'])
        jac += scale*np.einsum('tpk,pr->trk', derivative, parameters['p/quadratic'])
    elif kind == 'mlp':
        x = z
        dx = np.broadcast_to(np.eye(z.shape[-1]), (len(z), z.shape[-1], z.shape[-1]))
        for layer in ('l1', 'l2'):
            w = parameters[f'p/{layer}/w']
            pre = x@w+parameters[f'p/{layer}/b']
            sig = expit(pre)
            dx = np.einsum('tik,ij->tjk', dx, w)*(sig+pre*sig*(1-sig))[:, :, None]
            x = pre*sig
        a += scale*(x@parameters['p/out/w']+parameters['p/out/b'])
        jac += scale*np.einsum('tik,ir->trk', dx, parameters['p/out/w'])
    else:
        raise ValueError(kind)
    return a, jac


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('boundary_dir', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    directory = args.boundary_dir
    manifest = json.loads((directory/'data_manifest.json').read_text())
    n, bc = manifest['n'], manifest['bc']
    reflective = bc == 'dirichlet'
    axis = np.arange(1, n)/n if reflective else np.arange(n+1)/n
    xy = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    weights = np.ones(len(axis))
    if not reflective:
        weights[[0, -1]] = .5
    mass = (weights[:, None]*weights[None, :]/n**2).ravel()
    with np.load(directory/'bank_tables.npz') as table, np.load(directory/'bank_parameters.npz') as parameters:
        g, rr = table['g'], table['qr_r']
        raw = np.concatenate([bank_value(parameters, xy[start:start+4096], reflective) for start in range(0, len(xy), 4096)])
        np.testing.assert_allclose(mass, table['mass'], rtol=1e-14, atol=0)
        errors = {'bank_export': relative(g@rr, raw), 'mass_identity': relative(g.T@(mass[:, None]*g), np.eye(g.shape[1]))}
        field = g.reshape(len(axis), len(axis), -1)
        if reflective:
            field = np.pad(field, ((1, 1), (1, 1), (0, 0)))
            dx, dy = np.diff(field, axis=0), np.diff(field, axis=1)
            damping = np.zeros((g.shape[1], g.shape[1]))
        else:
            dx = np.diff(field, axis=0)*np.sqrt(weights)[None, :, None]
            dy = np.diff(field, axis=1)*np.sqrt(weights)[:, None, None]
            damping = sum(face.T@(weights[:, None]*face)/n for face in (field[0], field[-1], field[:, 0], field[:, -1]))
        dx, dy = dx.reshape(-1, g.shape[1]), dy.reshape(-1, g.shape[1])
        stiffness = dx.T@dx+dy.T@dy
        errors['stiffness_edges'] = relative(stiffness, table['stiffness_unit'])
        errors['damping_faces'] = relative(damping, table['damping_unit']) if not reflective else float(np.max(abs(table['damping_unit'])))
    assert max(errors.values()) < 1e-10, errors
    head_rows = []
    for path in sorted(directory.glob('*/reconstruction.npz')):
        arm = path.parent
        if not (arm/'head.npz').exists():
            continue
        kind = 'quadratic' if arm.name.startswith('quadratic') else 'mlp'
        with np.load(arm/'head.npz') as parameters, np.load(path) as saved:
            predictions, jac = head_value_jacobian(parameters, saved['fitted_z'], kind)
            tangent = np.einsum('trk,tk->tr', jac, saved['tangent_w'])
            row = dict(arm=arm.name, independent_prediction_error=relative(predictions, saved['predictions']),
                       independent_tangent_error=relative(tangent, saved['reconstructed_velocity']))
            assert row['independent_prediction_error'] < 1e-10, row
            assert row['independent_tangent_error'] < 1e-10, row
            with np.load(arm/'latent_fits.npz') as fitted:
                assert fitted['initial_starts'].shape[1] == 8
                np.testing.assert_array_equal(fitted['initial_starts'], fitted['doubled_initial_starts'])
                row['identical_eight_starts'] = True
        head_rows.append(row)
    report = dict(boundary_dir=str(directory.resolve()), bc=bc, intervals=n,
                  bank_operator_discrepancies=errors, heads=head_rows, passed=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
