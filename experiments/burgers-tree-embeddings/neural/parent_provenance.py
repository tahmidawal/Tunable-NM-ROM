"""Explicit retained parent selection; absent metadata means the historical B7 parent."""
from common import *
from neural.build_data import resolve
from training_io import read_checkpoint


def validate_expanded_schema(record, parent_record, parent):
    assert parent_record['status']=='complete_expanded_global_selection'
    assert parent_record['latent']==32 and not parent_record['final_test_opened']
    for cohort,count in [('train',160),('dev',52)]:
        labels=record[cohort+'_labels']
        assert len(labels)==count and len(set(labels))==count
        assert parent[cohort+'_labels']==parent_record[cohort+'_labels']==labels
    assert not set(record['train_labels'])&set(record['dev_labels'])
    assert parent['expanded_sources']==parent_record['expanded_sources']==record['expanded_sources']
    codes=np.asarray(parent['training_codes']);encoder=np.asarray(parent['training_encoder'])
    assert codes.shape==(1760,32) and encoder.shape==(1760,256)
    assert np.isfinite(codes).all() and np.isfinite(encoder).all()
    assert np.asarray(parent['network']['modulation']['w']).shape[0]==32
    assert len(parent['data_record']['groups']['train']['labels'])==128
    assert len(parent['data_record']['groups']['dev']['labels'])==44


def load_parent(record, ns):
    job=record.get('parent_job','b7parentrepair')
    assert job and job.replace('-','').isalnum()
    path=resolve(ns/job/'output/stable_parent.pkl',ns);rp=path.parent/'result.json'
    pr=json.loads(rp.read_text());assert digest(path)==pr['selected_sha256']==record['parent_sha256']
    parent=read_checkpoint(path);expanded=job!='b7parentrepair'
    proof=dict(parent_job=job,expanded_parent=expanded,parent_sha256=digest(path),parent_result_sha256=digest(rp))
    if expanded:
        manifest=HERE/'manifests'/f'retention-{job}.json';m=json.loads(manifest.read_text())
        assert m['local_and_remote_verified'] and m['terminal_state']=='COMPLETED'
        assert str(m['job'])==str(pr['meta']['job']) and m['source_commit']==pr['meta']['commit']
        entries={v['file']:v for v in m['files']}
        for name,p in [('output/result.json',rp),('output/stable_parent.pkl',path)]:
            assert p.stat().st_size==entries[name]['bytes'] and digest(p)==entries[name]['sha256']
        validate_expanded_schema(record,pr,parent)
        dp=resolve(ns/'b6parent/output/coordinate_data.npz',ns)
        assert digest(dp)==parent['data_sha256']==record['primary_data_sha256']
        with np.load(dp) as v:encoder=[np.array(v['train_encoder'])]
        for source in record['expanded_sources']:
            sp=resolve(ns/source['job']/'output/result.json',ns);sr=json.loads(sp.read_text())
            assert digest(sp)==source['record_sha256'] and sr['cohort']==source['cohort'] and not sr['final_test_opened']
            cache=sp.parent/'coordinate_data.npz';assert digest(cache)==source['cache_sha256']==sr['coordinate_data']['sha256']
            if source['cohort']=='train':
                with np.load(cache) as v:encoder.append(np.array(v['encoder']))
        assert np.array_equal(np.concatenate(encoder),parent['training_encoder'])
        proof['parent_retention_sha256']=digest(manifest)
    return path,pr,parent,proof
