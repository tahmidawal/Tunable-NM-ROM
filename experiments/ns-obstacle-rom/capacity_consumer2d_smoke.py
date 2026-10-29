"""Fresh default-cap reader over1M synthetic points; admission/native result mocked."""
from pathlib import Path
from unittest.mock import patch
import json,tempfile
import numpy as np
import capacity_context2d as c
import pilot2d
from fe_adapter import CAPS,adapter_digest
from prepare2d import preparation_digest


def main():
    root=Path(__file__).parent;measurement=root/'artifacts/native-capacity04/results/interface/result.json'
    factor=root/'artifacts/native-factor05/results/interface/result.json'
    native_review=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/CODEX-CROSSP02-CAPACITY04-FACTOR05-RESULTS.md')
    measured=json.loads(measurement.read_text());proposed=dict(CAPS)
    proposed.update(mixed_dofs=437140,elements=measured['counts']['elements'],volume_points=measured['counts']['volume_points'],matrix_nnz=max(v['native_nnz'] for v in measured['metrics'].values()))
    def desc(path):return {'path':str(path.resolve()),'sha256':c.sha(path)}
    with tempfile.TemporaryDirectory() as folder,patch.object(c,'require_qualification',return_value={'synthetic':True}):
        folder=Path(folder);source=preparation_digest();amend=folder/'amendment.json';review=folder/'capacity-review.json'
        record={'schema':'ns-obstacle-rom-count-capacity-amendment-v1','source_sha256':source,'scientific_changes':False,
            'reference_qualification':{'synthetic':True},'reference_admission_sha256':'fixture','original_caps':dict(CAPS),'proposed_caps':proposed,
            'measurement':desc(measurement),'factor_reuse':desc(factor),'native_control_review':desc(native_review)}
        amend.write_text(json.dumps(record));review.write_text(json.dumps({'schema':'ns-obstacle-rom-count-capacity-review-v1','passed':True,'amendment_sha256':c.sha(amend),'source_sha256':source}))
        release={'capacity_amendment':desc(amend),'capacity_review':desc(review),'point_block_volume':True,'verify_primary_volume':True,'adapter_manifest_sha256':'fixture'}
        metadata={'caps':proposed,'volume_storage':'point-block-v1','reference_qualification':{'synthetic':True},
            'admission_evidence':{'reference_admission_sha256':'fixture'},'source_case_files':[{'files':{'mesh.vol.bin':c.MESH_SHA}}]*9,
            'layout_sha256':c.LAYOUT_SHA,'adapter_source_sha256':adapter_digest(),'coefficient_files':{'volume/manifest.json':'fixture'}}
        normalized={'reference_admission_sha256':'fixture','cases':metadata['source_case_files']}
        proof=c.validate(folder/'release.json',release,normalized,source,CAPS,metadata)
        result=folder/'native-result.json';result.write_text(json.dumps({'schema':'ns-obstacle-native-preparation-result-v1','source_sha256':source,
            'status':'execution_completed_pending_independent_review','capacity_context':proof,'adapter_manifest_sha256':'fixture',
            'reference_admission_sha256':'fixture','reference_qualification':{'synthetic':True}}))
        acceptance=folder/'native-review.json';acceptance.write_text(json.dumps({'schema':'ns-obstacle-native-preparation-review-v1','passed':True,
            'result_sha256':c.sha(result),'source_sha256':source,'adapter_manifest_sha256':'fixture','capacity_amendment_sha256':c.sha(amend),
            'reference_qualification':{'synthetic':True}}))
        release.update(native_preparation_result=desc(result),native_preparation_review=desc(acceptance))
        defaults=dict(CAPS);assert defaults['volume_points']==100000
        geometry={'points_physical':np.zeros((proposed['volume_points'],2))}
        with patch.object(pilot2d,'read_preparation_metadata',return_value=(metadata,release)),patch.object(pilot2d,'_read_arrays',return_value={}),patch('point_store.PointVolumeStore',return_value=geometry):
            arrays,_,_=pilot2d.load_preparation(folder,folder/'release.json')
            assert len(arrays['volume']['points_physical'])==1036497 and CAPS==defaults
            release.pop('native_preparation_review')
            try:pilot2d.load_preparation(folder,folder/'release.json')
            except KeyError:pass
            else:raise AssertionError('missing independent preparation review accepted')
        assert CAPS==defaults
    print('PASS fresh consumer loads1036497 synthetic points only with exact reviewed count provenance; globals unchanged')


if __name__=='__main__':main()
