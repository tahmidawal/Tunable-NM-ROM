"""Require the coordinator's global validation freeze before scientific tests."""
import hashlib
import json
from pathlib import Path


def verify_validation_seal(directory,case_name,evaluation_seed):
    root=Path(directory);path=root/'global_validation_seal.json'
    if not path.exists():raise RuntimeError('scientific evaluation requires the global validation seal')
    seal=json.loads(path.read_text())
    if seal.get('schema')!='modcp-global-validation-seal-v1':raise RuntimeError('unknown global validation seal schema')
    if seal.get('evaluation_generated') is not False:raise RuntimeError('global seal does not certify an unopened evaluation cohort')
    cases=seal.get('cases',{})
    required={'burgers2d','wave_reflective','wave_absorbing'}
    if not required.issubset(cases):raise RuntimeError('global validation seal does not cover every planned PDE/boundary case')
    fields={'handoff_sha256','selection_proof_sha256','evaluation_seed','source_commit','validation_job_id'}
    if any(not isinstance(cases[c],dict) or not fields.issubset(cases[c]) for c in required):
        raise RuntimeError('global validation seal contains incomplete case entries')
    own=cases[case_name]
    handoff=root/'validation_handoff.json'
    if not handoff.exists() or hashlib.sha256(handoff.read_bytes()).hexdigest()!=own['handoff_sha256']:
        raise RuntimeError('copied validation handoff does not match global seal')
    metadata=json.loads(handoff.read_text())
    if metadata.get('case_name')!=case_name or metadata.get('status')!='validation_frozen':
        raise RuntimeError('copied handoff is not this case\'s completed validation')
    # The original immutable Burgers validation source predates the explicit
    # evaluation_opened flag. Its validation-only rows/status and absence of
    # evaluation parameters supply the corresponding legacy evidence.
    if metadata.get('evaluation_opened',False) is not False or metadata.get('evaluation_generated',False) is not False:
        raise RuntimeError('copied validation handoff says evaluation was already opened')
    if not metadata.get('invocations') or any(r.get('split')!='validation' for r in metadata['invocations']):
        raise RuntimeError('copied validation contains missing or non-validation invocations')
    if metadata.get('physical_cases',{}).get('evaluation') or any(r.get('split')=='evaluation' for r in metadata.get('references',[])):
        raise RuntimeError('copied validation contains evaluation data')
    configuration=root/'config.json'
    if not configuration.exists() or json.loads(configuration.read_text())!=metadata.get('config'):
        raise RuntimeError('staged configuration differs from sealed validation')
    if int(metadata['config']['seeds']['evaluation'])!=int(evaluation_seed):
        raise RuntimeError('staged evaluation seed differs from sealed validation configuration')
    checkpoints=metadata.get('checkpoint_hashes',{})
    if not {'cp','modcp','film'}.issubset(checkpoints):raise RuntimeError('sealed validation lacks a comparison checkpoint')
    for name,expected in checkpoints.items():
        if Path(name).name!=name:raise RuntimeError('checkpoint key must be a basename')
        source=root/'checkpoints'/f'{name}.pkl'
        if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'staged checkpoint differs from sealed validation: {name}')
    proof=own.get('selection_proof_sha256',{})
    if not proof:raise RuntimeError('global seal lacks frozen selection proofs')
    for name,expected in proof.items():
        if Path(name).name!=name:raise RuntimeError('selection proof must be a basename')
        source=root/name
        if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'frozen selection proof does not match seal: {name}')
    choices=root/'selections.json'
    if not choices.exists() or json.loads(choices.read_text())!=metadata.get('selections'):
        raise RuntimeError('staged selections differ from sealed validation handoff')
    if 'selected_configurations' in own and own['selected_configurations']!=metadata['selections']:
        raise RuntimeError('global seal selections differ from sealed validation handoff')
    if int(own['evaluation_seed'])!=int(evaluation_seed):raise RuntimeError('evaluation seed differs from global freeze')
    provenance=metadata['provenance']
    if str(own['source_commit'])!=str(provenance['commit']):raise RuntimeError('validation source commit differs from freeze')
    if str(own['validation_job_id'])!=str(provenance['job_id']):raise RuntimeError('validation job differs from freeze')
    return dict(global_validation_seal_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                validation_handoff_sha256=own['handoff_sha256'],validation_source_commit=own['source_commit'],
                validation_job_id=own['validation_job_id'])
