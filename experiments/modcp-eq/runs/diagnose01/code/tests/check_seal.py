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
        validation=dict(provenance=dict(commit='source',job_id='123'))
        (p/'validation_handoff.json').write_text(json.dumps(validation))
        (p/'selections.json').write_text('[]')
        (p/'selection_freeze.json').write_text('{}')
        digest=lambda file:hashlib.sha256((p/file).read_bytes()).hexdigest()
        entry=dict(handoff_sha256=digest('validation_handoff.json'),selection_proof_sha256={x:digest(x) for x in ('selections.json','selection_freeze.json')},evaluation_seed=42,source_commit='source',validation_job_id='123')
        seal=dict(schema='modcp-global-validation-seal-v1',cases={c:dict(entry) for c in ('burgers2d','wave_reflective','wave_absorbing')})
        path=p/'global_validation_seal.json';path.write_text(json.dumps(seal))
        verified=verify_validation_seal(p,'burgers2d',42)
        assert verified['global_validation_seal_sha256']==digest(path.name)
        rejected(seed=43)
        (p/'selections.json').write_text('["tampered"]');rejected()
        (p/'selections.json').write_text('[]')
        del seal['cases']['wave_absorbing'];path.write_text(json.dumps(seal));rejected()
    print('PASS global coverage, own handoff/proof hashes, seed, source and missing-seal guard')


if __name__=='__main__':main()
