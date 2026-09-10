"""Audit collected validation provenance and coverage without reopening evaluation."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess


CELL = Path(__file__).resolve().parent
TREE = CELL.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(label):
    run = CELL/'runs'/label
    cluster = run/'cluster'
    out = cluster/'out/comparison'
    submission = json.loads((run/'submission.json').read_text())
    archive = json.loads((run/'ARCHIVE.json').read_text())
    data = json.loads((out/'handoff.json').read_text())
    job = submission['job_id']
    assert submission['remote_cleanup_complete']
    assert (run/'verified-cluster.tar').stat().st_size == archive['bytes']
    assert archive['job_id'] == job
    assert (cluster/'EXIT_CODE').read_text().strip() == '0'
    stdout = (cluster/'logs'/f'{job}.out').read_text()
    stderr = (cluster/'logs'/f'{job}.err').read_text()
    assert 'jax_backend=gpu' in stdout
    assert 'WAVE_VALIDATION_COMPLETE_EVALUATION_SEALED' in stdout
    assert 'Ran 4 tests' in stderr and 'Ran 3 tests' in stderr
    assert len(re.findall(r'^OK$', stderr, re.M)) == 2
    assert not re.search(r'Traceback|out of memory|ResourceExhausted|captured.*constant|disk.*full|cuInit|Warning', stdout+stderr, re.I)
    meta = data['provenance']
    assert meta['job_id'] == job
    assert meta['commit'] == submission['source_commit'] == archive['source_commit']
    assert meta['backend'] == 'gpu' and meta['x64'] and meta['matmul_precision'] == 'highest'
    for rel, expected in submission['source_sha256'].items():
        source = cluster/rel
        assert digest(source) == expected, rel
        git_path = 'experiments/'+str(Path(rel).relative_to('code'))
        original = subprocess.check_output(['git', 'show', f'{meta["commit"]}:{git_path}'], cwd=TREE)
        assert hashlib.sha256(original).hexdigest() == expected, git_path
    assert data['status'] == 'validation_frozen' and data['phase'] == 'validation'
    assert data['evaluation_opened'] is False
    cfg = data['config']
    expected_configurations = 94 if data['case_name'] == 'wave_reflective' else 90
    panels = {}
    for n in cfg['meshes']:
        rows = [r for r in data['invocations'] if r['intervals'] == n]
        proxy = [r for r in data['selection_timings'] if r['intervals'] == n]
        assert len(rows) == expected_configurations*cfg['validation_count']
        by_config = collections.defaultdict(list)
        for row in rows:
            assert row['split'] == 'validation' and row['rep'] == 0
            assert row['field_artifact_kind'] == 'observation_grid_only_full_metrics_computed_before_subsampling'
            assert (out/row['field_artifact']).is_file()
            by_config[row['configuration']].append(row['case'])
        assert len(by_config) == expected_configurations
        assert all(sorted(v) == list(range(cfg['validation_count'])) for v in by_config.values())
        proxy_keys = [(r['configuration'], r['rep']) for r in proxy]
        assert len(proxy_keys) == len(set(proxy_keys)) == expected_configurations*cfg['repetitions']
        assert set(proxy_keys) == {(name, rep) for name in by_config for rep in range(cfg['repetitions'])}
        assert all(r['case'] == 0 and r['split'] == 'validation_selection_proxy' for r in proxy)
        freeze = json.loads((out/f'frozen_selection_{n}.json').read_text())
        assert freeze['evaluation_opened_at_selection'] is False
        assert freeze['training_sha256'] == data['checkpoint_sha256']
        for rel, expected in freeze['quadrature_sha256'].items():
            assert digest(out/rel) == expected
        panels[n] = {'configuration_count': len(by_config), 'cohort_rows': len(rows),
                     'selection_proxy_rows': len(proxy),
                     'numerically_completed_rows': sum(r['completed'] for r in rows),
                     'stationary_rows': sum(r['stationary'] for r in rows),
                     'selected_configuration_count': len(freeze['selected_settings']),
                     'selection_proof_sha256': digest(out/f'frozen_selection_{n}.json')}
    assert len(data['references']) == len(cfg['meshes'])*cfg['validation_count']
    assert all(r['split'] == 'validation' and r['temporal_reference_passed'] for r in data['references'])
    return {'case_name': data['case_name'], 'job_id': job, 'source_commit': meta['commit'],
            'handoff': str((out/'handoff.json').relative_to(CELL)), 'handoff_sha256': digest(out/'handoff.json'),
            'archive_sha256': archive['sha256'], 'archive_bytes': archive['bytes'],
            'source_files_verified_against_git': len(submission['source_sha256']),
            'gpu': meta['gpu'], 'panels': panels, 'references_passed': len(data['references']),
            'evaluation_opened': False, 'remote_cleanup_complete': True, 'passed': True}


if __name__ == '__main__':
    result = {'interpretation': 'Source, runtime, reference-gate and coverage audit. Full-mesh validation errors are not independently reconstructed from the saved observation-grid fields. Archive and member hashes were verified by the collector before remote cleanup.',
              'panels': [audit(label) for label in ('validation_reflective02', 'validation_absorbing02')]}
    target = CELL/'runs/validation-owner-audit.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
