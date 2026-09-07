"""Generate the protected-head record exclusively from saved run JSONs."""
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs/b3d_architecture'
REPO = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value):
    return f'{100*value:.6f}%'


def exact_tree(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            exact_tree(left, right)
    else:
        np.testing.assert_array_equal(actual, expected)


def audit(screen):
    directory = RUNS / 'screen40'
    config = screen['config']
    assert screen['complete'] and config['model'] == 'b3d_arch_anchor'
    assert config['backend'] == 'gpu' and config['x64']
    assert config['matmul_precision'] == 'highest' and not config['test_table_opened']
    assert config['optimizer_seeds'] == [200, 201]
    source = config['source_config']
    assert (source['N'], source['k'], source['r'], source['seed']) == (33, 32, 128, 0)
    assert source['n_train_traj'] == 512 and source['max_snaps'] == 8192
    assert source['val_rows'] == [512, 575]
    assert sha(ROOT / 'b3d_arch_anchor.py') == config['model_sha256']
    assert sha(ROOT / 'runs/b3d_repair/head33/out/refined_checkpoint.pkl') == config['source_sha256']
    with (ROOT / 'runs/b3d_repair/head33/out/refined_checkpoint.pkl').open('rb') as stream:
        bank_source = pickle.load(stream)
    assert (directory / 'EXIT_CODE').read_text().strip() == '0'
    assert (directory / 'COMMIT.txt').read_text().strip() == config['commit']
    stdout = (directory / 'logs' / f"{config['slurm_job']}.out").read_text()
    stderr = (directory / 'logs' / f"{config['slurm_job']}.err").read_text()
    assert 'jax_backend=gpu' in stdout and 'DONE ' in stdout
    assert not stderr.strip(), 'inspect any warning or error before accepting this job'
    subprocess.run(['sha256sum', '-c', 'PULL.sha256', '--quiet'], cwd=directory, check=True)
    source_checks = []
    for line in (directory / 'INPUTS.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name.startswith('code/'):
            content = subprocess.check_output(['git', '-C', str(REPO), 'show',
                f"{config['commit']}:experiments/separable-decoder/{Path(name).name}"])
            assert hashlib.sha256(content).hexdigest() == digest
            source_checks.append(name)
    expected_ids = (np.arange(512, 576)[:, None]*51+np.asarray([0,10,25,50])).ravel()
    np.testing.assert_array_equal(screen['validation_states']['sid'], expected_ids)
    for key, value in screen['provenance']['fields'].items():
        if key in {'nu', 's_star'}:
            assert value['max_relative'] <= screen['provenance']['derived_rtol']
        else:
            assert value['exact']
    checkpoints = []
    for run in screen['runs']:
        training = run['training']
        assert training['steps'] == 60000 and training['lr'] == 3e-4
        assert training['batch'] == 4096 and training['norm'] == 'global'
        path = directory / 'out' / f"protected_anchor_seed{run['seed']}.pkl"
        assert sha(path) == run['checkpoint_sha256']
        with path.open('rb') as stream:
            payload = pickle.load(stream)
        assert payload['model'] == config['model'] and payload['codes'].shape == (8192,32)
        assert set(payload['trainable']) == {'bias', 'residual'}
        assert payload['source'] == config
        exact_tree(payload['bank_params'], bank_source['params'])
        u = np.asarray(payload['frozen']['anchor'])
        expected_anchor, ru = np.linalg.qr((np.asarray(bank_source['params']['h_lin']) @ payload['r'].T).T,
                                         mode='reduced')
        expected_anchor *= np.where(np.diag(ru) < 0, -1., 1.)[None, :]
        np.testing.assert_allclose(u, expected_anchor, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(u.T @ u, np.eye(32), rtol=1e-12, atol=1e-12)
        for stage in run['fits']:
            errors = np.asarray(stage['error'])
            assert len(errors) == 256 and np.all(np.isfinite(errors))
            assert abs(errors.mean()-stage['summary']['mean']) < 1e-14
            assert abs(errors.max()-stage['summary']['worst']) < 1e-14
            assert np.sum(errors > .15) == stage['outliers_above_15pct']
            assert np.sum(np.asarray(stage['optimality']) > 1e-6) == stage['nonstationary']
        checkpoints.append(dict(seed=run['seed'], sha256=run['checkpoint_sha256']))
    return dict(passed=True, job=config['slurm_job'], source_commit=config['commit'],
                checked_source_files=source_checks, checked_checkpoints=checkpoints,
                cohort_and_precision_checked=True, gpu_and_completion_logged=True,
                stderr_empty=True, role='local result-integrity audit; independent scientific review remains separate')


def main():
    source_path = RUNS / 'screen40/out/result.json'
    screen = json.loads(source_path.read_text())
    audited = audit(screen)
    (RUNS / 'screen40/local_audit.json').write_text(json.dumps(audited, indent=2)+'\n')
    lines = ['# Burgers 3D protected-anchor screen', '',
        'Generated from saved raw results. These are provisional validation-only architecture results pending independent result review; no pilot, rollout or wave-transfer result is claimed.', '',
        'All rows retain the learned spatial bank and use the common affine initialization. The anchor arm fixes the linear directions and projects its nonlinear correction away from them. The MLP controls are the previously saved campaign controls.', '',
        '| Head | Seed | Training mean | Validation mean | Median | Worst | Above worst gate | Unconverged | Tangent mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label in ['mlp128', 'mlp192', 'screen40']:
        data = screen if label == 'screen40' else json.loads((RUNS/label/'out/result.json').read_text())
        name = 'Protected anchor' if label == 'screen40' else f"MLP width {data['config']['model_config']['width']}"
        for run in data['runs']:
            fit = run['fits'][-1]
            s = fit['summary']
            lines.append(f"| {name} | {run['seed']} | {percent(run['training']['final']['mean'])} | {percent(s['mean'])} | {percent(s['median'])} | {percent(s['worst'])} | {fit['outliers_above_15pct']} | {fit['nonstationary']} | {percent(fit['tangent_summary']['mean'])} |")
    lines += ['', 'Equal updates are not equal compute. Optimizer repeats share the same data; no cross-job timing comparison is made. The legacy normalized-gradient criterion defines the unconverged count; invariant stationarity is reported separately below.', '',
        '| Anchor seed | Initial-state mean | Later-state mean | Maximum gradient | Maximum invariant stationarity | Budget change |',
        '|---|---:|---:|---:|---:|---:|']
    for run in screen['runs']:
        fit = run['fits'][-1]
        maximum_eta = max(g['invariant_stationarity'] for g in fit['geometry'])
        lines.append(f"| {run['seed']} | {percent(fit['groups']['initial']['mean'])} | {percent(fit['groups']['later']['mean'])} | {max(fit['optimality']):.8e} | {maximum_eta:.8e} | {run['budget_change_max']:.8e} |")
    lines += ['', '| Anchor seed | Recovery error | Anchor-Jacobian error | Minimum Gram eigenvalue | Minimum singular value | Maximum condition number |',
        '|---|---:|---:|---:|---:|---:|']
    for run in screen['runs']:
        d = run['architecture_diagnostics']
        lines.append(f"| {run['seed']} | {d['coordinate_recovery_max_absolute']:.8e} | {d['anchor_jacobian_identity_max_absolute']:.8e} | {d['minimum_gram_eigenvalue']:.8e} | {d['minimum_jacobian_singular_value']:.8e} | {d['maximum_jacobian_condition']:.8e} |")
    lines += ['', '## Necessary screen checks', '',
        'These checks do not replace the full inherited pilot, POD comparison or negative controls. Passing them alone would not permit rollout promotion.', '',
        '| Anchor seed | Mean within gate | Worst within gate | Gradient within gate | Stable budget |',
        '|---|---|---|---|---|']
    for run in screen['runs']:
        fit = run['fits'][-1]
        checks = [fit['summary']['mean'] <= .05, fit['summary']['worst'] <= .15,
                  max(fit['optimality']) <= 1e-6, run['budget_change_max'] < .01]
        lines.append('| '+str(run['seed'])+' | '+' | '.join('PASS' if check else 'FAIL' for check in checks)+' |')
    config = screen['config']
    cancelled = json.loads((RUNS / 'screen/cancellation.json').read_text())
    lines += ['', '## Provenance and limits', '',
        f"Source commit `{config['commit']}`; job `{config['slurm_job']}` on `{config['node']}`, {config['gpu']}. Backend `{config['backend']}`, x64 `{config['x64']}`, matmul precision `{config['matmul_precision']}`.", '',
        f"Bank validation mean {percent(screen['bank_error']['mean'])}; truth residual maximum {screen['bank']['truth_max_residual']:.8e}; backward-Euler velocity identity maximum {screen['bank']['velocity_be_identity_max']:.8e}.", '',
        'The source/checkpoint hashes, output pull, precision flags and validation membership passed the local integrity audit. Actual data were regenerated from seed; archived parameter metadata supplied only the already bounded provenance reference.', '',
        f"Earlier queued job `{cancelled['job_id']}` was cancelled before execution to use the available GPU class. It produced no numerical result; its submission, cancellation and checked metadata pull remain archived. Replacement job `{cancelled['replacement_job_id']}` retained the same scientific settings.", '',
        'The protected head imposes a graph over the chosen fixed anchor. Recovery and rank protection do not establish adequate physical tangent coverage, good conditioning or successful dynamics. A failed screen concerns this anchor and bounded optimization protocol; local multistart fits are not certified global minima.', '',
        '[Raw result](runs/b3d_architecture/screen40/out/result.json), [local integrity audit](runs/b3d_architecture/screen40/local_audit.json), [implementation and geometry tests](B3D-ANCHOR-DESIGN.md).', '',
        '## Glossary', '',
        '- **Head / bank:** latent-to-coefficient map, and frozen learned spatial features.',
        '- **Seed:** optimizer randomness; each repeat uses the same data cohort.',
        '- **Training / validation:** snapshots used to fit the model, and unseen snapshots used to assess it.',
        '- **Mean / median / worst:** relative full-field reconstruction errors over the indicated states.',
        '- **Above worst gate / unconverged:** counts exceeding the error or normalized-gradient criteria.',
        '- **Tangent mean:** mean relative truth velocity outside the decoder Jacobian range.',
        '- **Initial / later:** snapshots before or after physical time stepping.',
        '- **Gradient / invariant stationarity:** legacy normalized objective gradient, and residual projected onto available field directions.',
        '- **Budget change:** maximum relative reconstruction change when latent-fit attempts are doubled.',
        '- **Recovery / anchor-Jacobian error:** numerical violations of exact latent recovery or its derivative identity.',
        '- **Gram / singular value / condition number:** derivative metric, directional stretching, and ratio of largest to smallest stretching.',
        '- **Gate / pilot:** a required criterion, and the inherited complete validation procedure.',
        '- **POD:** a linear comparator fitted from training snapshots.',
        '- **Backward Euler:** the implicit time discretization used for truth trajectories.',
        '- **Rollout / wave transfer:** online evolution, and any later evaluation on wave equations.', '']
    (ROOT/'B3D-ANCHOR-NOTES.md').write_text('\n'.join(lines))
    print(json.dumps(audited, indent=2))


if __name__ == '__main__':
    main()
