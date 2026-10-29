"""Schema-only membership fixture; FD1 qualification mocked, no native execution."""
from pathlib import Path
from unittest.mock import patch
import json,tempfile
import materialize_reference2d as m


def main():
    with tempfile.TemporaryDirectory() as folder,patch.object(m,'require_qualification',return_value={'synthetic':True}):
        root=Path(folder);original=root/'original';original.mkdir()
        for name in ('EXPORT-MANIFEST.json','mesh.vol.bin','solution.sol'):(original/name).write_text('synthetic')
        (original/'config.json').write_text(json.dumps({'nu':.001,'release':'G2_steady_control_candidate'}))
        (original/'fe_layout.json').write_text(json.dumps({'sha256':'a'*64}))
        identities={'export_manifest_sha256':m.sha(original/'EXPORT-MANIFEST.json'),'config_sha256':m.sha(original/'config.json'),
            'mesh_sha256':m.sha(original/'mesh.vol.bin'),'layout_sha256':'a'*64,'solution_sha256':m.sha(original/'solution.sol'),'storage_mode':'expanded_full_fields'}
        rows=[]
        for split,values in [('train',[15+1.25*j for j in range(9)]),('development',[16.875,23.125]),('validation',[16.,18.,22.,24.])]:
            rows.extend(dict(identities,Re=re,split=split) for re in values)
        truth=root/'truth.json';release=root/'release.json'
        def write():
            truth.write_text(json.dumps({'synthetic':True,'admitted_references':rows}))
            release.write_text(json.dumps({'schema':'ns-obstacle-fd1-primary-materialization-release-v1',
                'authorized_stage':'training_primary_materialization','source_sha256':m.source_digest(),'split':'training','single_execution':True,
                'reference_admission_sha256':m.sha(truth),'reference_qualification':{'synthetic':True},'original_export_sha256':identities['export_manifest_sha256'],
                'Re':20.,'source_storage_mode':'legacy_expanded_Re20'}))
        write();m.verify(original,truth,release)
        next(row for row in rows if row['Re']==20.)['export_manifest_sha256']='0'*64;write()
        try:m.verify(original,truth,release)
        except ValueError:pass
        else:raise AssertionError('non-adopted control substituted for actual reference')
    print('PASS explicit15-Re membership and adopted Re20 identities; alternate control rejected; synthetic only')


if __name__=='__main__':main()
