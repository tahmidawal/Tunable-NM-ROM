"""Audit collected final provenance, sealed choices, and every repetition."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from audit_validation import CELL, TREE, digest

sys.path.insert(0, str(CELL/'wave'))
from seal import verify_validation_bundle


def audit(label):
    run = CELL/'runs'/label
    cluster, out = run/'cluster', run/'cluster/out/comparison'
    submission = json.loads((run/'submission.json').read_text())
    archive = json.loads((run/'ARCHIVE.json').read_text())
    data = json.loads((out/'handoff.json').read_text())
    meta, cfg = data['provenance'], data['config']
    job = submission['job_id']
    assert submission['remote_cleanup_complete'] and archive['job_id'] == job
    assert (run/'verified-cluster.tar').stat().st_size == archive['bytes']
    assert (cluster/'EXIT_CODE').read_text().strip() == '0'
    stdout = (cluster/'logs'/f'{job}.out').read_text()
    stderr = (cluster/'logs'/f'{job}.err').read_text()
    assert 'jax_backend=gpu' in stdout and 'WAVE_COMPARISON_COMPLETE' in stdout
    assert 'Ran 4 tests' in stderr and 'Ran 3 tests' in stderr
    assert len(re.findall(r'^OK$', stderr, re.M)) == 2
    assert not re.search(r'Traceback|out of memory|ResourceExhausted|captured.*constant|disk.*full|cuInit|Warning', stdout+stderr, re.I)
    assert meta['job_id'] == job
    assert meta['commit'] == submission['source_commit'] == archive['source_commit']
    assert meta['backend'] == 'gpu' and meta['x64'] and meta['matmul_precision'] == 'highest'
    for rel, expected in submission['source_sha256'].items():
        assert digest(cluster/rel) == expected
        original = subprocess.check_output(['git', 'show', f'{meta["commit"]}:experiments/{Path(rel).relative_to("code")}'], cwd=TREE)
        assert hashlib.sha256(original).hexdigest() == expected
    assert data['status'] == 'complete' and data['phase'] == 'evaluation'
    assert data['evaluation_opened'] is True and data['selection_timings'] == []
    old, panels, proof = verify_validation_bundle(out/'imported_validation', cfg, data['case_name'], data['checkpoint_sha256'])
    assert data['imported_validation_proof'] == proof
    assert data['selections'] == old['selections']
    refs = {(r['intervals'], r['case']): r for r in data['references']}
    assert len(refs) == len(data['references']) == len(cfg['meshes'])*cfg['evaluation_count']
    assert all(r['split'] == 'evaluation' and r['temporal_reference_passed'] for r in refs.values())
    summaries = {}
    for n, settings in panels.items():
        ids = {s['id'] for s in settings}
        rows = [r for r in data['invocations'] if r['intervals'] == n]
        keys = [(r['configuration'], r['case'], r['rep']) for r in rows]
        expected = {(name, case, rep) for name in ids for case in range(cfg['evaluation_count']) for rep in range(cfg['repetitions'])}
        assert len(keys) == len(set(keys)) and set(keys) == expected
        for row in rows:
            assert row['split'] == 'evaluation'
            assert row['field_artifact_kind'] == 'full_grid_with_shared_truth'
            assert (out/row['field_artifact']).is_file()
            reference = refs[(n, row['case'])]
            assert row['reference_artifact'] == reference['reference_artifact']
            assert row['reference_sha256'] == reference['reference_sha256']
            assert (out/row['reference_artifact']).is_file()
            assert set(row['output_sha256']) == {'u', 'v'}
        for rel, expected_hash in json.loads((out/'imported_validation'/f'frozen_selection_{n}.json').read_text())['quadrature_sha256'].items():
            assert digest(out/rel) == expected_hash
        summaries[n] = {'configurations': len(ids), 'invocations': len(rows),
                        'numerically_completed': sum(r['completed'] for r in rows),
                        'stationary': sum(r['stationary'] for r in rows),
                        'prediction_files': len({r['field_artifact'] for r in rows}),
                        'reference_files': len({r['reference_artifact'] for r in rows}),
                        'artifact_relations': dict(collections.Counter(r['artifact_relation'] for r in rows))}
    return {'passed': True, 'case_name': data['case_name'], 'job_id': job,
            'source_commit': meta['commit'], 'gpu': meta['gpu'],
            'handoff_sha256': digest(out/'handoff.json'), 'global_seal_sha256': proof['seal_sha256'],
            'archive_sha256': archive['sha256'], 'archive_bytes': archive['bytes'],
            'source_files_verified_against_git': len(submission['source_sha256']),
            'remote_cleanup_complete': True, 'panels': summaries}


if __name__ == '__main__':
    labels = sys.argv[1:] or ['evaluation_reflective01', 'evaluation_absorbing01']
    results = {'interpretation': 'Owner provenance, seal, coverage and reference-link audit. Full-field physical errors and energy/invariant histories are independently recomputed by the coordinator.',
               'panels': [audit(label) for label in labels]}
    target = CELL/'runs'/('evaluation-owner-audit.json' if len(labels) == 2 else labels[0]+'-owner-audit.json')
    target.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))
