"""Rebuild a development-frozen model/operator without fitting or selecting again."""
from pathlib import Path
import json,hashlib
import numpy as np
from pilot2d import sha_file,_read_arrays,source_digest,model_fingerprint,array_digest
from qualification import require_qualification
from weak import WeakOperator


def input_identity(metadata,probes,bank):
    parameters={'transform':bank.transform}
    for i,(weight,bias) in enumerate(bank.layers):parameters.update({f'weight_{i}':weight,f'bias_{i}':bias})
    return {'prepared_files':metadata.get('coefficient_files'),
            'reference_admission_sha256':metadata.get('admission_evidence',{}).get('reference_admission_sha256'),
            'training_array_sha256':metadata.get('training_array_sha256'),
            'probes_sha256':array_digest(probes),'bank_parameters_sha256':array_digest(parameters),
            'original_training_budget':{k:bank.metadata.get(k) for k in
                ('seed','steps_executed','steps_cap','time_cap_seconds','seconds','last_checkpoint')}}


def read_evidence(path,expected_sha256):
    path=Path(path)
    if sha_file(path)!=expected_sha256:raise ValueError('frozen evidence record differs')
    wrapped=json.loads(path.read_text());array_path=path.parent/wrapped['array_file']
    if Path(wrapped['array_file']).name!=wrapped['array_file']:raise ValueError('local array sidecar required')
    if sha_file(array_path)!=wrapped['array_sha256']:raise ValueError('frozen array sidecar differs')
    arrays=_read_arrays(array_path,2**30)
    def decode(value):
        if isinstance(value,dict) and set(value)=={'array','shape','dtype','content_sha256'}:
            a=arrays[value['array']]
            if list(a.shape)!=value['shape'] or str(a.dtype)!=value['dtype']:raise ValueError('array descriptor differs')
            h=hashlib.sha256();h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes(order='C'))
            if h.hexdigest()!=value['content_sha256']:raise ValueError('frozen array bytes differ')
            return a
        if isinstance(value,dict):return {k:decode(v) for k,v in value.items()}
        if isinstance(value,list):return [decode(v) for v in value]
        return value
    return decode(wrapped['record'])


def load_frozen(path,expected_sha256,qualification):
    record=read_evidence(path,expected_sha256)
    if (record.get('schema')!='ns-obstacle-frozen-rom-development-v1' or record.get('validation_seen') is not False
        or record.get('source_sha256')!=source_digest()):raise ValueError('exact development-only frozen model record required')
    if require_qualification(record['reference_qualification'])!=require_qualification(qualification):
        raise ValueError('frozen model truth scope differs')
    return record


def rebuild(record,name,arrays,metadata,probes,bank):
    from pilot2d import build_model
    if input_identity(metadata,probes,bank)!=record['input_identity']:raise ValueError('frozen model inputs changed')
    specification=record['models'][name]
    if specification['kind']=='bank' and record['bank_gate'].get('passed') is not True:
        raise RuntimeError('bank development floor gate closed')
    model=build_model(specification['kind'],specification['prefix'],arrays,metadata,probes,bank)
    if model_fingerprint(model)!=specification['fingerprint']:raise ValueError('rebuild changed frozen operator/projection')
    frozen=record['fitted_operators'][name]
    operator=None if frozen is None else WeakOperator(**frozen)
    if (operator is None)!=(record['choices'][name]['selected_support'] is None):
        raise ValueError('selected operator/support mismatch')
    if operator is not None and (operator.nv,operator.np)!=(model.operator.nv,model.operator.np):
        raise ValueError('frozen fitted operator dimensions differ')
    return model,operator
