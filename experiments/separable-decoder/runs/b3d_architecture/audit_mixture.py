"""Audit the completed mixture screen using hashes and independent NumPy values."""
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def compare_tree(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare_tree(a, b)
    else:
        np.testing.assert_array_equal(left, right)


def parameter_count(tree):
    if isinstance(tree, dict):
        return sum(parameter_count(x) for x in tree.values())
    if isinstance(tree, (tuple, list)):
        return sum(parameter_count(x) for x in tree)
    return np.asarray(tree).size


def independent_forward(p, f, z):
    x = np.asarray(z) / f['latent_scale']
    logits = x @ p['gate'][0] + p['gate'][1]
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    experts = []
    for layers in p['experts']:
        hidden = x
        for w, b in layers[:-1]:
            hidden = hidden @ w + b
            hidden *= .5 + .5*np.tanh(.5*hidden)
        experts.append(f['output_scale'] * (hidden @ layers[-1][0] + layers[-1][1]))
    experts = np.stack(experts, axis=1)
    return p['bias'] + np.asarray(z) @ p['linear'].T + np.sum(weights[:, :, None]*experts, axis=1), weights, experts


def main():
    record = HERE / 'screen40'
    report = json.loads((record/'out/result.json').read_text())
    config = report['config']
    assert report['complete'] and report['kind'] == 'b3d_architecture'
    assert config['model'] == 'b3d_arch_mixture'
    assert config['backend'] == 'gpu' and config['x64']
    assert config['matmul_precision'] == 'highest'
    assert config['optimizer_seeds'] == [200, 201] and len(report['runs']) == 2
    assert config['test_table_opened'] is False
    assert (record/'EXIT_CODE').read_text().strip() == '0'
    assert (record/'COMMIT.txt').read_text().strip() == config['commit']
    for manifest in ['PULL.sha256', 'RESULTS.sha256']:
        subprocess.run(['sha256sum','-c',manifest,'--quiet'],cwd=record,check=True)
    assert (record/'INPUTS.sha256').read_bytes() == (record/'MANIFEST.sha256').read_bytes()
    source_hashes = {}
    input_hashes = {}
    for line in (record/'INPUTS.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        input_hashes[name] = expected
        if name.startswith('code/'):
            content = subprocess.check_output(['git','-C',str(REPO),'show',
                f'{config["commit"]}:experiments/separable-decoder/{name[5:]}'])
            actual = hashlib.sha256(content).hexdigest()
            assert actual == expected
            source_hashes[name] = actual
    assert source_hashes['code/b3d_arch_mixture.py'] == config['model_sha256']
    assert input_hashes['in/checkpoint.pkl'] == config['source_sha256']
    logs = '\n'.join(path.read_text() for path in (record/'logs').glob('*'))
    assert 'jax_backend=gpu' in logs and 'DONE ' in logs
    for text in ['captured constant', 'out of memory', 'no space left', 'traceback', 'time limit']:
        assert text not in logs.lower(), text
    assert all(report['provenance']['fields'][key]['exact'] for key in
               ['seed','m','B','c','w','rho','A'])
    states = report['validation_states']
    expected_sid = np.asarray([row*51+step for row in range(512,576) for step in [0,10,25,50]])
    np.testing.assert_array_equal(states['sid'], expected_sid)
    cfg = config['source_config']
    assert (cfg['N'],cfg['k'],cfg['r'],cfg['seed']) == (33,32,128,0)
    control = json.loads((HERE/'mlp128/out/result.json').read_text())
    with (HERE/'mlp128/out/mlp_control_seed200.pkl').open('rb') as stream:
        control_checkpoint = pickle.load(stream)
    assert control['config']['source_sha256'] == config['source_sha256']
    cross_job = {}
    for key, value in states.items():
        a, b = np.asarray(value), np.asarray(control['validation_states'][key])
        if key in ['sid','blob_count']:
            np.testing.assert_array_equal(a,b)
        else:
            np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
        cross_job[key] = dict(exact=bool(np.array_equal(a,b)),max_absolute=float(np.max(np.abs(a-b))))
    runs = []
    for run in report['runs']:
        path = record/'out'/f'smooth_mixture_seed{run["seed"]}.pkl'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run['checkpoint_sha256']
        with path.open('rb') as stream:
            checkpoint = pickle.load(stream)
        compare_tree(checkpoint['bank_params'],control_checkpoint['bank_params'])
        np.testing.assert_array_equal(checkpoint['r'],control_checkpoint['r'])
        for key in checkpoint['frozen']:
            np.testing.assert_allclose(checkpoint['frozen'][key],control_checkpoint['frozen'][key],rtol=1e-12,atol=1e-12)
        assert checkpoint['codes'].shape == (8192,32)
        train = run['training']
        assert (train['steps'],train['lr'],train['batch'],train['norm']) == (60000,3e-4,4096,'global')
        assert train['initialization_code_max_difference'] <= 1e-12
        assert train['initialization_field_max_difference'] <= 1e-12
        count = parameter_count(checkpoint['trainable'])
        assert train['trainable_parameter_count'] == count
        assert [fit['budget'] for fit in run['fits']] == [400,800]
        fit = run['fits'][-1]
        output, weights, experts = independent_forward(checkpoint['trainable'],checkpoint['frozen'],fit['z'])
        errors = np.sqrt((np.sum((output-np.asarray(states['target']))**2,axis=1)
                          +np.asarray(states['perpendicular2']))/np.asarray(states['norm2']))
        np.testing.assert_allclose(errors,fit['error'],rtol=1e-11,atol=1e-12)
        diag = run['architecture_diagnostics']
        np.testing.assert_allclose(weights,diag['routing_weights'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(weights.mean(axis=0),diag['mean_usage'],rtol=1e-12,atol=1e-12)
        entropy = -np.sum(weights*np.log(np.maximum(weights,np.finfo(float).tiny)),axis=1)
        np.testing.assert_allclose(entropy,diag['routing_entropy'],rtol=1e-12,atol=1e-12)
        disagreement = np.linalg.norm(experts[:,0]-experts[:,1],axis=1)
        np.testing.assert_allclose(disagreement,diag['expert_disagreement_absolute'],rtol=1e-11,atol=1e-12)
        assert bool(weights.mean(axis=0).min()<.05) == diag['routing_collapse']
        assert (float(np.sqrt(np.mean((disagreement/diag['expert_disagreement_reference_scale'])**2))) <= 1e-6) == diag['identical_expert_collapse']
        runs.append(dict(seed=run['seed'],trainable_parameters=count,
                         independent_error_difference_max=float(np.max(np.abs(errors-fit['error']))),
                         routing_and_disagreement_independently_verified=True,
                         rank_min=min(x['rank'] for x in fit['geometry'])))
    audit = dict(status='verified_by_implementation_agent',independent_coordinator_audit='pending',
                 source_commit=config['commit'],job=config['slurm_job'],gpu=config['gpu'],
                 output_and_log_checksums_verified=True,source_hashes=source_hashes,
                 backend='gpu',x64=True,matmul_precision='highest',same_bank_and_qr_as_control=True,
                 same_state_membership_as_control=True,cross_job_validation=cross_job,
                 host_affinity_warning='hwloc_set_cpubind' in logs,runs=runs)
    audit['remote_directory_deleted_after_verified_pull'] = (
        (record/'completion.log').exists() and
        'verified_pull_remote_directory_deleted' in (record/'completion.log').read_text())
    (record/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
