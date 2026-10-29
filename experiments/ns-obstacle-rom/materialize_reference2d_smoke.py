"""Pure array/container derivation regression, never a qualification or native replay."""
from pathlib import Path
import hashlib,json,tempfile,shutil
import numpy as np
import materialize_reference2d as m


def digest(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(json.dumps({'dtype':a.dtype.str,'shape':list(a.shape)},sort_keys=True).encode());h.update(memoryview(a).cast('B'));return h.hexdigest()


def main():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);original=root/'original';original.mkdir()
        config={'release':'control','split':'control','nu':.001,'velocity_order':4}
        export={'mesh_filename':'mesh.vol.bin','mesh_format':'synthetic','reload_policy':'synthetic'}
        for name in m.COPIED:(original/name).write_text('synthetic fixture '+name)
        (original/'config.json').write_text(json.dumps(config));(original/'EXPORT-MANIFEST.json').write_text(json.dumps(export))
        (original/'PRIMARY-REFERENCE.json').write_text('synthetic fixture')
        a=np.array([[1.,-0.],[2.,3.]])
        primary={'derived_full_rule_arrays':{'velocity':{'dtype':a.dtype.str,'shape':list(a.shape),'content_sha256':digest(a)}},
                 'primary_sha256':'fixture','immutable_geometry_rules':{'fixture':'not native'}}
        release={'reference_admission_sha256':'synthetic-only','Re':20.,'original_export_sha256':m.sha(original/'EXPORT-MANIFEST.json')}
        before={p.name:m.sha(p) for p in original.iterdir()}
        out=root/'derived';out.mkdir()
        result=m.derive(original,out,config,primary,{'volume':{'velocity':a}}, {'synthetic':True},release,{'synthetic':True},digest)
        assert result['status']=='exact_content_replay_passed_pending_independent_admission'
        assert result['historical_ZIP_identity_claimed'] is False
        changed=json.loads((out/'config.json').read_text());assert changed['nu']==config['nu'] and changed['velocity_order']==4
        assert changed['release']==m.EXPORT_RELEASE and changed['split']=='training'
        bad=root/'failed';bad.mkdir()
        try:m.derive(original,bad,config,primary,{'volume':{'velocity':a+1}}, {},release,{'synthetic':True},digest)
        except ArithmeticError:pass
        else:raise AssertionError('changed field bits accepted')
        assert (bad/'full_rule.npz').is_file() and (bad/'MATERIALIZATION.json').is_file()
        assert not (bad/'EXPORT-MANIFEST.json').exists()
        assert before=={p.name:m.sha(p) for p in original.iterdir()}
        legacy=root/'legacy';legacy.mkdir()
        for source in original.iterdir():
            if source.name!='PRIMARY-REFERENCE.json':shutil.copyfile(source,legacy/source.name)
        np.savez_compressed(legacy/'full_rule.npz',velocity=a)
        before_legacy={p.name:m.sha(p) for p in legacy.iterdir()}
        target=root/'legacy-derived';target.mkdir()
        legacy_release=dict(release,source_storage_mode='legacy_expanded_Re20')
        result=m.derive(legacy,target,config,{'derived_full_rule_arrays':primary['derived_full_rule_arrays']},
            {'volume':{'velocity':a}},{'no_PDE_solve':True,'new_native_replay_performed':False},legacy_release,{'synthetic':True},digest)
        assert result['source_storage_mode']=='legacy_expanded_Re20' and 'primary_manifest_sha256' not in result
        assert result['original_full_rule_archive_sha256']==m.sha(legacy/'full_rule.npz')
        assert json.loads((target/'config.json').read_text())['scope_contract_sha256']==m.SCOPE_SHA256
        assert before_legacy=={p.name:m.sha(p) for p in legacy.iterdir()}

    print('PASS primary and expanded legacy derivation/new containers; original immutable; altered bits retained and rejected; synthetic only')


if __name__=='__main__':main()
