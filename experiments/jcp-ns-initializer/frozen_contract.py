"""Read-only enforcement of source/config and untouched-cohort registration."""
import hashlib,json
import support as S

def verify(config,config_path):
    frozen=json.loads((S.HERE/'FROZEN_PROTOCOL.json').read_text())
    for rel,digest in frozen['files'].items():
        if S.sha(S.HERE/rel)!=digest:raise RuntimeError('frozen source/config mismatch: '+rel)
    schemas={'ns-initializer-untouched-confirmation-v2':'configs/fresh-initializers.json',
             'ns-repaired-untouched-phase-v1':'configs/fresh-phase.json'}
    expected=schemas.get(config.get('schema'))
    if expected is None or S.sha(config_path)!=frozen['files'][expected]:
        raise RuntimeError('supplied configuration is not the exact frozen configuration')
    registration=json.loads((S.HERE/'FRESH_COHORT_REGISTRATION.json').read_text())
    if registration['frozen_protocol_sha256']!=S.sha(S.HERE/'FROZEN_PROTOCOL.json'):
        raise RuntimeError('cohort registration is not tied to this frozen protocol')
    if registration['seed']!=2026112301 or registration['count']!=32:
        raise RuntimeError('unregistered fresh cohort')
    if not registration['nonoverlap_pass']:raise RuntimeError('fresh tuple nonoverlap failed')
    if config['eval_seed']!=registration['seed'] or config['eval_cases']!=registration['count']:
        raise RuntimeError('configuration differs from registered cohort')
    return frozen,registration

def parameters(config,registration):
    values=S.F.parameters(config['eval_seed'],config['eval_cases'])
    digest=hashlib.sha256(values.tobytes()).hexdigest()
    if digest!=registration['parameter_sha256']:raise RuntimeError('physical tuple regeneration mismatch')
    return values
