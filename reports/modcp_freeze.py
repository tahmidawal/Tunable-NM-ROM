"""Independently bind evaluation evidence to the global validation freeze."""
import json
from pathlib import Path

from modcp_audit import digest

CASES = {'burgers2d', 'wave_reflective', 'wave_absorbing'}


def verify_evaluation_freeze(path, data):
    if not any(row['split'] == 'evaluation' for row in data.get('invocations', [])):
        return None
    folder = Path(path).parent
    burgers = data['case_name'] == 'burgers2d'
    bundle = folder if burgers else folder/'imported_validation'
    seal_path = bundle/('global_validation_seal.json' if burgers else 'global_seal.json')
    validation_path = bundle/('validation_handoff.json' if burgers else 'handoff.json')
    if not seal_path.is_file() or not validation_path.is_file():
        raise ValueError('Evaluation lacks the saved global validation freeze')
    seal = json.loads(seal_path.read_text())
    if (seal.get('schema') != 'modcp-global-validation-seal-v1' or
            seal.get('evaluation_generated') is not False or set(seal.get('cases', {})) != CASES):
        raise ValueError('Global validation freeze is incomplete or already opened')
    required = {'handoff_sha256', 'selection_proof_sha256', 'evaluation_seed',
                'source_commit', 'validation_job_id', 'selected_configurations'}
    if any(not required <= set(entry) or not entry['selection_proof_sha256']
           for entry in seal['cases'].values()):
        raise ValueError('Global validation freeze lacks a case proof')
    entry = seal['cases'][data['case_name']]
    old = json.loads(validation_path.read_text())
    if (digest(validation_path) != entry['handoff_sha256'] or old['case_name'] != data['case_name'] or
            old['status'] != 'validation_frozen' or old.get('evaluation_opened') or
            any(row['split'] == 'evaluation' for row in old.get('invocations', [])) or
            old.get('physical_cases', {}).get('evaluation')):
        raise ValueError('Saved validation evidence differs from the unopened global freeze')
    if (old['config'] != data['config'] or old['selections'] != data['selections'] or
            entry['selected_configurations'] != data['selections']):
        raise ValueError('Evaluation configurations or selections changed after validation freeze')
    seed = data['config']['seeds']['evaluation'] if burgers else data['config']['evaluation_seed']
    if (entry['evaluation_seed'] != seed or entry['source_commit'] != old['provenance']['commit'] or
            str(entry['validation_job_id']) != str(old['provenance']['job_id'])):
        raise ValueError('Evaluation seed or validation source differs from global freeze')
    checkpoints = lambda value: value.get('checkpoint_hashes', value.get('checkpoint_sha256', {}))
    if set(checkpoints(old)) != {'cp', 'modcp', 'film'} or checkpoints(old) != checkpoints(data):
        raise ValueError('Evaluation checkpoint identities differ from frozen validation')
    names = {'selections.json', 'selection_freeze.json'} if burgers else {
        f'frozen_selection_{n}.json' for n in data['config']['meshes']}
    if set(entry['selection_proof_sha256']) != names:
        raise ValueError('Global freeze omits a required selection proof')
    for name, expected in entry['selection_proof_sha256'].items():
        if digest(bundle/name) != expected:
            raise ValueError('Saved selection proof differs from global freeze')
    actual_hash = digest(seal_path)
    recorded_hash = (data['provenance'].get('global_validation_seal_sha256') if burgers else
                     data.get('imported_validation_proof', {}).get('seal_sha256'))
    if recorded_hash != actual_hash:
        raise ValueError('Evaluation did not record this global validation freeze')
    return dict(path=str(seal_path), sha256=actual_hash,
                validation_handoff=str(validation_path), validation_sha256=entry['handoff_sha256'],
                selection_proof_sha256=entry['selection_proof_sha256'])
