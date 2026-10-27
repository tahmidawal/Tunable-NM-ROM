"""Persist the independent saved-step audit with explicit comparison denominators."""
from pathlib import Path
import json


def render(here, load, sha, table):
    record = load(here / 'data/POD-A3-ADJOINT-REVIEW.json')
    a3 = load(here / 'data/lane-pod-export.json')['a3_replay']
    result_path = Path(a3['result_path'])
    assert sha(result_path) == a3['result_sha256'] == record['result_sha256']
    result = json.loads(result_path.read_text())
    assert result['artifacts']['files']['Phi.npz'] == record['Phi_sha256']
    assert result['artifacts']['files']['HR~ecsw~512~k16.npz'] == record['k16_sha256']
    assert sha(result_path.parent / record['array_relative_path']) == record['array_sha256']
    assert a3['selected']['scheme'] == 'CN' and a3['selected']['dt'] == record['dt']
    text = '# POD A3: independent saved-step adjoint audit\n\n'
    text += record['status'] + '. ' + record['supersedes'] + '.\n\n'
    text += (f"Validation case {record['case']}, state indices {record['state_indices']}, "
             f"step {record['dt']}, spacing {record['h']}, viscosity {record['nu']}; "
             f"first {record['basis_columns']} columns of the pinned POD basis.\n\n")
    text += r'''With zero-Dirichlet central differences, write $D_\Sigma=D_x+D_y+D_z$ and
$F(u)=u\odot D_\Sigma u-\nu\Delta_hu$. For $u_j=\Phi c_j$ and $\tau=\Delta t/2$,

$$
r_F=u_1-u_0+\tau[F(u_1)+F(u_0)],
$$

$$
g_F=\Phi^\top\!\left[r_F+\tau\{(D_\Sigma u_1)\odot r_F-D_\Sigma(u_1\odot r_F)-\nu\Delta_hr_F\}\right].
$$

The direct stencil row is

$$
J_i=\Phi_i+\tau\{(D_\Sigma u_1)_i\Phi_i+u_{1,i}(D_\Sigma\Phi)_i-\nu(\Delta_h\Phi)_i\},
\qquad g_E=\sum_{i\in\mathcal I}w_iJ_i^\top r_{F,i}.
$$

The full gradient uses the discrete adjoint above. The sampled gradient uses direct
selected rows. This restricts the state to the trial span while retaining the full
residual; it does not first Galerkin-project that residual.

'''
    text += (f"The Jacobian Frobenius norm was accumulated in {record['jacobian_column_block']}-column blocks. "
             f"The common full-metric denominator is $\\|J_F\\|_F\\|r_F\\|={record['full_metric_denominator']}$. "
             'Target-norm-relative columns instead divide by the corresponding saved target norm.\n\n')
    rows = [[key, value['absolute_difference'], value['full_metric_normalized'] or 'not used',
             value['saved_target_norm'], value['target_norm_relative']] for key, value in record['comparisons'].items()]
    text += table(['Compared quantity', 'Absolute difference norm', 'Full-metric-normalized difference',
                   'Saved target norm', 'Target-norm-relative difference'], rows) + '\n\n'
    text += record['precision_note'] + '\n\n'
    text += f"Result SHA256 `{record['result_sha256']}`; array SHA256 `{record['array_sha256']}`. "
    text += ('Basis/support hashes and exact fresh output are in '
             '[the immutable review record](../../data/POD-A3-ADJOINT-REVIEW.json).\n\n'
             '## Glossary\n\n- Adjoint: transpose of the residual derivative applied to a vector.\n'
             '- Trial span: fields represented by the frozen basis times coefficients.\n'
             '- Full metric: the full Jacobian Frobenius norm multiplied by the full residual norm.\n'
             '- Target-relative: difference divided by the corresponding saved target norm.\n')
    path = here / 'audits/coordinator/POD-A3-ADJOINT-SUPPLEMENT.md'
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    return path
