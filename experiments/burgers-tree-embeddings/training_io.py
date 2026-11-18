"""Verified in-namespace caches and complete small model checkpoints."""
import pickle
from common import *
import data

def checkpoint(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('wb') as f:pickle.dump(jax.device_get(value),f,protocol=5)
    return digest(p)

def read_checkpoint(path):
    with Path(path).open('rb') as f:return pickle.load(f)

def load_reference(directory,cohort):
    directory=Path(directory)
    assert str(directory).startswith(os.environ['STUDY_NAMESPACE']+'/')
    result=json.loads((directory/'result.json').read_text())
    assert result['status']=='spectral_ladders_complete_pending_independent_checks'
    rows=[r for r in result['cases'] if not r['case'].startswith('fixture')]
    if cohort=='train':assert len(rows)==64
    else:assert len(rows)==28
    fields=[];norms=[];params=[];sources=[];labels=[]
    for row in rows:
        assert row['provisional_ladder_pass']
        p=directory/(row['case']+'_training.npz');v=np.load(p)
        assert str(v['reference_sha256'])==row['levels'][-1]['sha256']
        assert np.array_equal(v['parameters'],row['parameters'])
        fields.append(v['field'].reshape(11,-1));norms.append(v['norm']);params.append(v['parameters'])
        sources.append(dict(file=str(p),sha256=digest(p),reference_sha256=str(v['reference_sha256'])))
        labels.append(row['case'])
    xy,w=data.grid(64,True)
    return dict(Y=jnp.asarray(np.concatenate(fields)),norm=jnp.asarray(np.concatenate(norms)),
                parameters=np.asarray(params),labels=labels,xy=jnp.asarray(xy),w=jnp.asarray(w),
                sources=sources,reference_result_sha256=digest(directory/'result.json'))

def errors(pred,target,w):
    return jnp.sqrt(jnp.sum((pred-target)**2*w[None],axis=1)/jnp.sum(target**2*w[None],axis=1))

def summary(err):
    a=np.asarray(err)
    return dict(per_snapshot=a,median=float(np.median(a)),p90=float(np.quantile(a,.9)),
                worst=float(a.max()),failures=int(np.sum(a>.005)))
