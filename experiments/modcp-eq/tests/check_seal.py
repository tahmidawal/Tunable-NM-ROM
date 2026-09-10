"""Validation seal guards fail closed before any scientific final seed draw."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.seal import verify_validation_seal


def main():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
        p=Path(name)
        def rejected(seed=42):
            try:verify_validation_seal(p,'burgers2d',seed)
            except RuntimeError:return
            raise AssertionError('invalid global seal was accepted')
        rejected()
        config=dict(seeds=dict(evaluation=42));(p/'config.json').write_text(json.dumps(config))
        (p/'checkpoints').mkdir()
        for arm in ('cp','modcp','film'):(p/'checkpoints'/f'{arm}.pkl').write_bytes(arm.encode())
        checkpoints={arm:hashlib.sha256(arm.encode()).hexdigest() for arm in ('cp','modcp','film')}
        validation=dict(case_name='burgers2d',status='validation_frozen',config=config,checkpoint_hashes=checkpoints,
                        invocations=[dict(split='validation')],selections=[],provenance=dict(commit='source',job_id='123'))
        (p/'validation_handoff.json').write_text(json.dumps(validation))
        (p/'selections.json').write_text('[]')
        (p/'selection_freeze.json').write_text('{}')
        digest=lambda file:hashlib.sha256((p/file).read_bytes()).hexdigest()
        entry=dict(handoff_sha256=digest('validation_handoff.json'),selection_proof_sha256={x:digest(x) for x in ('selections.json','selection_freeze.json')},evaluation_seed=42,source_commit='source',validation_job_id='123')
        seal=dict(schema='modcp-global-validation-seal-v1',evaluation_generated=False,cases={c:dict(entry) for c in ('burgers2d','wave_reflective','wave_absorbing')})
        path=p/'global_validation_seal.json';path.write_text(json.dumps(seal))
        verified=verify_validation_seal(p,'burgers2d',42)
        assert verified['global_validation_seal_sha256']==digest(path.name)
        rejected(seed=43)
        for marker in (True,None):
            seal['evaluation_generated']=marker;path.write_text(json.dumps(seal));rejected()
        seal['evaluation_generated']=False;path.write_text(json.dumps(seal))
        (p/'checkpoints/cp.pkl').write_bytes(b'changed');rejected()
        (p/'checkpoints/cp.pkl').write_bytes(b'cp')
        (p/'config.json').write_text('{}');rejected();(p/'config.json').write_text(json.dumps(config))
        for key,value in [('case_name','wave_reflective'),('status','complete'),('evaluation_opened',True),
                          ('invocations',[dict(split='evaluation')]),('physical_cases',dict(evaluation=[[1]]))]:
            changed=dict(validation);changed[key]=value
            (p/'validation_handoff.json').write_text(json.dumps(changed))
            seal['cases']['burgers2d']['handoff_sha256']=digest('validation_handoff.json')
            path.write_text(json.dumps(seal));rejected()
        (p/'validation_handoff.json').write_text(json.dumps(validation))
        seal['cases']['burgers2d']['handoff_sha256']=digest('validation_handoff.json');path.write_text(json.dumps(seal))
        (p/'selections.json').write_text('["tampered"]');rejected()
        (p/'selections.json').write_text('[]')
        del seal['cases']['wave_absorbing'];path.write_text(json.dumps(seal));rejected()
    print('PASS unopened global cohort, own validation identity, checkpoint/config/proof hashes, seed and source guards')


if __name__=='__main__':main()
