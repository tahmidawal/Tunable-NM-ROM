"""Metadata/context regression with qualification mocked; no applied native release."""
from pathlib import Path
from unittest.mock import patch
import json,tempfile
import capacity_context2d as c
from fe_adapter import CAPS


def main():
    root=Path(__file__).parent;measurement=root/'artifacts/native-capacity04/results/interface/result.json'
    factor=root/'artifacts/native-factor05/results/interface/result.json'
    review=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/CODEX-CROSSP02-CAPACITY04-FACTOR05-RESULTS.md')
    layout=root/'staging/native-capacity04/payload/inputcase/fe_layout.json'
    measured=json.loads(measurement.read_text());proposed=dict(CAPS)
    proposed.update(mixed_dofs=sum(measured['counts'][key] for key in ('velocity_dofs','pressure_dofs')),
        elements=measured['counts']['elements'],volume_points=measured['counts']['volume_points'],
        matrix_nnz=max(v['native_nnz'] for v in measured['metrics'].values()))
    def desc(path):return {'path':str(path.resolve()),'sha256':c.sha(path)}
    with tempfile.TemporaryDirectory() as folder,patch.object(c,'require_qualification',return_value={'synthetic':True}):
        folder=Path(folder);amend=folder/'amendment.json';independent=folder/'review.json'
        record={'schema':'ns-obstacle-rom-count-capacity-amendment-v1','source_sha256':'fixture','scientific_changes':False,
            'reference_qualification':{'synthetic':True},'reference_admission_sha256':'fixture','original_caps':dict(CAPS),'proposed_caps':proposed,
            'measurement':desc(measurement),'factor_reuse':desc(factor),'native_control_review':desc(review)}
        admission={'reference_admission_sha256':'fixture','cases':[{'resolved_directory':str(layout.parent),'files':{'mesh.vol.bin':c.MESH_SHA,'fe_layout.json':c.sha(layout)}}]}
        def release():
            amend.write_text(json.dumps(record));independent.write_text(json.dumps({'schema':'ns-obstacle-rom-count-capacity-review-v1',
                'passed':True,'amendment_sha256':c.sha(amend),'source_sha256':'fixture'}))
            return {'capacity_amendment':desc(amend),'capacity_review':desc(independent),'point_block_volume':True,'verify_primary_volume':True}
        defaults=dict(CAPS);testcaps=dict(CAPS)
        try:
            with c.approved_capacity(folder/'release.json',release(),admission,'fixture',testcaps):
                assert testcaps==proposed and testcaps['factor_estimate_bytes']==defaults['factor_estimate_bytes']
                raise RuntimeError('synthetic preparation failure')
        except RuntimeError:pass
        assert testcaps==defaults and CAPS==defaults
        record['proposed_caps']['factor_estimate_bytes']+=1
        try:c.validate(folder/'release.json',release(),admission,'fixture',testcaps)
        except ValueError:pass
        else:raise AssertionError('factor cap changed')
    print('PASS exact measured-count context, original caps restored after failure, factor cap increase rejected; qualification mocked only')


if __name__=='__main__':main()
