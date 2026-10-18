"""Independent affine-image lower bound for the frozen CP validation failure.

No experiment code or JAX is imported. This is a full-grid reconstruction
diagnostic, excluded from online timing and configuration selection.
"""
import json
from pathlib import Path
import pickle
import numpy as np

from modcp_audit import digest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/'worktrees'/'2026-09-10-modcp-burgers2d'/'experiments'/'modcp-eq'


def main():
    checkpoint = EXPERIMENT/'runs'/'diagnose01'/'out'/'checkpoints'/'cp.pkl'
    fields = EXPERIMENT/'diagnostics'/'validation_initial'/'validation_cp_cap10_dt0.00125_q4_tol0.0001_L256_case3.npz'
    projection = EXPERIMENT/'runs'/'diagnose01'/'out'/'cp_case3_free_span_projection.npz'
    model = pickle.loads(checkpoint.read_bytes())
    cfg, params = model['config'], model['params']
    assert cfg['architecture'] == 'cp' and cfg['boundary'] == 'dirichlet' and cfg['outputs'] == 1
    n, rank = cfg['intervals'], cfg['rank']
    with np.load(fields, allow_pickle=False) as archive:
        truth, online, same = archive['truth_u'], archive['u'], archive['same_grid_u']
    assert truth.shape[1:] == (n+1, n+1) and np.array_equal(truth[0], same[0])
    factors = params['factors'][:, 0]
    assert factors.dtype == np.float64 and factors.shape == (2, rank, n+1)
    mask = np.ones((n+1, n+1))
    mask[[0, -1], :] = 0
    mask[:, [0, -1]] = 0
    basis = (np.einsum('ri,rj->ijr', factors[0], factors[1])*mask[..., None]).reshape(-1, rank)
    bias = (float(params['bias'][0])*mask).reshape(-1, 1)
    target = truth.reshape(len(truth), -1).T-bias
    coefficients, _, numerical_rank, singular = np.linalg.lstsq(basis, target, rcond=1e-12)
    residual = basis@coefficients-target
    scale = np.linalg.norm(truth[0])
    errors = np.linalg.norm(residual, axis=0)/scale
    orthogonality = np.linalg.norm(basis.T@residual)/(np.linalg.norm(basis)*np.linalg.norm(residual))
    with np.load(projection, allow_pickle=False) as saved:
        coefficient_difference = float(np.max(np.abs(coefficients-saved['coefficients'])))
        residual_difference = float(np.max(np.abs(residual.T.reshape(truth.shape)-saved['residual'])))
        np.testing.assert_allclose(errors, saved['errors'], rtol=1e-12, atol=1e-13)
    assert numerical_rank == rank and orthogonality < 1e-12
    result = dict(scope='Validation-only affine-image lower bound for this frozen CP checkpoint; no online cost claim.',
                  inputs={str(p.relative_to(ROOT)): digest(p) for p in (checkpoint, fields, projection)},
                  intervals=n, case=3, rank=int(numerical_rank), rcond=1e-12,
                  singular_values=singular.tolist(), condition_number=float(singular[0]/singular[-1]),
                  fixed_bias=float(params['bias'][0]), relative_projection_errors=errors.tolist(),
                  normalized_orthogonality=float(orthogonality),
                  saved_coefficient_max_difference=coefficient_difference,
                  saved_residual_max_difference=residual_difference,
                  initial_reference_identity=True)
    stem = ROOT/'reports'/'2026-09-10-modified-cp-span-audit'
    stem.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fitted = (basis@coefficients+bias).T.reshape(truth.shape)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    bound = max(float(np.max(np.abs(x))) for x in (truth[0], fitted[0], online[0]))
    online_error = np.linalg.norm(online[0]-truth[0])/scale
    for ax, field, title in zip(axes, (truth[0], fitted[0], online[0]),
                               ('Supplied initial field', f'Free CP coefficients: {100*errors[0]:.3g}% error',
                                f'Online latent fit: {100*online_error:.3g}% error')):
        plot = ax.imshow(field.T, origin='lower', extent=(0, 1, 0, 1), vmin=-bound, vmax=bound, cmap='RdBu_r')
        ax.set(title=title, xlabel='x', ylabel='y')
    fig.colorbar(plot, ax=axes, shrink=.75, label='Scalar field')
    fig.suptitle(f'Burgers validation case {result["case"]} — initial state — {n} intervals\n'
                 'Post-hoc diagnosis of a validation failure; excluded from evaluation selection and timing')
    for suffix in ('.png', '.pdf'):
        fig.savefig(stem.with_suffix(suffix), dpi=170, bbox_inches='tight',
                    metadata={'CreationDate': None, 'ModDate': None} if suffix == '.pdf' else None)
    plt.close(fig)
    print(json.dumps({key: result[key] for key in ('rank', 'condition_number', 'relative_projection_errors', 'normalized_orthogonality')}, indent=2))


if __name__ == '__main__':
    main()
