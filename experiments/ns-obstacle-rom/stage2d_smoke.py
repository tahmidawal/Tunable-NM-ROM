"""Metadata/claim policy smoke only; no dataset, admission or native execution."""
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import stage2d
from qualification import GATE,expected_scope

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);bad=root/'amendment.json';bad.write_text(json.dumps({'qualification_scope':expected_scope()}))
    try:stage2d.build_candidate('prepare',{'training_manifest':str(bad),'preparation_release':str(bad)},root/'rejected.json')
    except RuntimeError:pass
    else:raise AssertionError('amendment admitted')
    assert not (root/'rejected.json').exists()
    prepared=root/'prepared';prepared.mkdir();(prepared/'fixture.txt').write_text('not scientific data')
    inputs={'prepared':str(prepared),'preparation_release':str(bad)}
    q={'gate':GATE,'passed':True,'qualification_scope':expected_scope()}
    with patch('stage2d.qualification',return_value=q):
        candidate=root/'candidate.json';stage2d.build_candidate('train',inputs,candidate)
        release=root/'release.json';record={'schema':'ns-obstacle-rom-stage-release-v1','authorized_stage':'train',
            'candidate_sha256':stage2d.sha_file(candidate),'source_sha256':stage2d.source_digest(),
            'single_execution':True,'reference_qualification':q}
        release.write_text(json.dumps(record));stage2d.verify(candidate,release)
        (prepared/'fixture.txt').write_text('changed bytes')
        try:stage2d.verify(candidate,release)
        except ValueError:pass
        else:raise AssertionError('changed warm input accepted')
    assert 'jax' not in sys.modules and 'ngsolve' not in sys.modules
print('PASS stage metadata policy and input mutation rejection; qualification mocked only in structural fixture, no execution')
