"""Compare real development inference with the retained training-checkpoint fields.

This reads existing development fields only; it never generates final parameters.
The absolute tolerance is declared before the development replay is executed.
"""
import argparse, hashlib, json, pickle
from pathlib import Path
import numpy as np

ATOL = 1e-10


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(training, panel, destination):
    training, panel = Path(training), Path(panel)
    source = json.loads((training/'result.json').read_text())
    result = json.loads((panel/'result.json').read_text())
    assert source['complete'] and result['complete']
    assert not source['final_cohort_opened'] and not result['final_cohort_opened']
    cfg = result['config']; n = cfg['train_intervals']
    assert cfg['representation_oracles']
    for key in ('train_count', 'train_seed', 'validation_count', 'validation_seed',
                'diffusivity', 'times', 'bank_rank', 'representation_fit_starts',
                'representation_fit_budget', 'representation_fit_tolerance'):
        assert cfg[key] == source['config'][key], key
    source_cohorts = json.loads((training/'cohorts.json').read_text())
    target_cohorts = json.loads((panel/'cohorts.json').read_text())
    for key in ('training_parameters', 'validation_parameters'):
        np.testing.assert_array_equal(source_cohorts[key], target_cohorts[key])
    checks = []; files = ['bank.pkl']
    for k in cfg['latent_dimensions']:
        files.append(f'head_K{k}.pkl')
        first = np.load(training/f'head_K{k}_development.npz')
        replay = np.load(panel/'fields'/f'N{n}_K{k}_best_found.npz')
        error = float(np.max(np.abs(first['prediction']-replay['prediction'])))
        np.testing.assert_allclose(first['prediction'], replay['prediction'], rtol=0, atol=ATOL)
        checks.append(dict(kind='head_best_found_fields', k=k, maximum_absolute_difference=error,
                           cases=len(first['prediction'])))
    for name in cfg['frozen_operators']:
        checkpoint = f'operators/{name}/adapter.pkl'; files.append(checkpoint)
        adapter = pickle.loads((training/checkpoint).read_bytes())
        first = np.load(training/f'operators/{name}/development.npz')['prediction']
        # Training uses BXYZT normalized evolved fields; query uses T/XYZ physical fields.
        expected = np.moveaxis(first, -1, 1)*adapter['physical_scale']
        error = 0.
        for case, reference in enumerate(expected):
            replay = np.load(panel/'fields'/f'N{n}_case{case}_{name}.npz')['prediction'][1:]
            error = max(error, float(np.max(np.abs(reference-replay))))
            np.testing.assert_allclose(reference, replay, rtol=0, atol=ATOL)
        checks.append(dict(kind='operator_native_fields', name=name,
                           maximum_absolute_difference=error, cases=len(expected)))
    hashes = {}
    for name in files:
        hashes[name] = sha(training/name)
        assert hashes[name] == sha(panel/name), (name, 'model bytes changed during reuse')
    data = dict(passed=True, tolerance_absolute=ATOL, final_data_generated=False,
        source_result_sha256=sha(training/'result.json'), replay_result_sha256=sha(panel/'result.json'),
        source_commit=source['source_commit'], replay_commit=result['source_commit'],
        source_job_id=source['job_id'], replay_job_id=result['job_id'],
        checkpoint_sha256=hashes, checks=checks,
        scope='real native development inference and truth-assisted representation replay on unchanged checkpoint bytes; not a cross-job runtime comparison')
    Path(destination).write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training'); parser.add_argument('panel'); parser.add_argument('destination')
    args = parser.parse_args(); audit(args.training, args.panel, args.destination)
