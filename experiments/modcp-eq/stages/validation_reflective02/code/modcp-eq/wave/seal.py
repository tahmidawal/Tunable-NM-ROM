"""Pure host-side global validation seal checks, before drawing evaluation data."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_validation_bundle(folder, cfg, case_name, checkpoint_sha256):
    folder = Path(folder)
    seal_path = folder/'global_seal.json'
    seal = json.loads(seal_path.read_text())
    if (seal.get('schema') != 'modcp-global-validation-seal-v1' or
            seal.get('evaluation_generated') is not False or
            not {'burgers2d', 'wave_reflective', 'wave_absorbing'} <= set(seal.get('cases', {}))):
        raise RuntimeError('Missing complete global validation seal; evaluation stays closed')
    required = {'handoff_sha256', 'selection_proof_sha256', 'evaluation_seed', 'source_commit', 'validation_job_id'}
    if any(not required <= set(entry) or not entry['selection_proof_sha256'] for entry in seal['cases'].values()):
        raise RuntimeError('Another pilot case has an incomplete validation seal')
    entry = seal['cases'][case_name]
    path = folder/'handoff.json'
    old = json.loads(path.read_text())
    if (sha(path) != entry['handoff_sha256'] or old['case_name'] != case_name or
            old['status'] != 'validation_frozen' or old['evaluation_opened'] or
            old['checkpoint_sha256'] != checkpoint_sha256 or
            old['provenance']['commit'] != entry['source_commit'] or
            str(old['provenance']['job_id']) != str(entry['validation_job_id']) or
            entry['evaluation_seed'] != cfg['evaluation_seed'] or
            any(old['config'][key] != value for key, value in cfg.items())):
        raise RuntimeError('Validation proof/configuration/checkpoint differs from global seal')
    panels = {}
    expected_files = {f'frozen_selection_{n}.json' for n in cfg['meshes']}
    if set(entry['selection_proof_sha256']) != expected_files:
        raise RuntimeError('Global seal does not cover every wave mesh')
    for n in cfg['meshes']:
        name = f'frozen_selection_{n}.json'
        proof = json.loads((folder/name).read_text())
        if (sha(folder/name) != entry['selection_proof_sha256'][name] or
                proof['evaluation_opened_at_selection'] is not False or
                proof['intervals'] != n or proof['training_sha256'] != checkpoint_sha256):
            raise RuntimeError('Frozen mesh selection failed integrity check')
        for relative, expected in proof['quadrature_sha256'].items():
            if sha(folder/relative) != expected:
                raise RuntimeError('Frozen quadrature content differs from selected rule')
        panels[n] = proof['selected_settings']
    return old, panels, {'schema': seal['schema'], 'seal_sha256': sha(seal_path),
                         'handoff_sha256': sha(path), 'selection_proof_sha256': entry['selection_proof_sha256'],
                         'created_utc': seal['created_utc'], 'evaluation_generated_at_global_seal': False,
                         'validation_provenance': old['provenance'], 'validated_cases': sorted(seal['cases'])}
