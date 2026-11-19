"""Matched training/development coordinate samples from retained qualified references."""
from common import *
import data


def resolve(path,ns):
    p=Path(path);assert str(p).startswith(str(ns)+'/')
    if p.exists():return p
    relative=p.relative_to(ns)
    if relative.parts[0]=='archive':raise FileNotFoundError(p)
    q=ns/'archive'/relative;assert q.exists();return q


def build(out):
    ns=Path(os.environ['STUDY_NAMESPACE']);source=ns/'archive/b1expanded/output/result.json'
    record=json.loads(source.read_text());xy,w=data.grid(128,True);xe,we=data.grid(16,True)
    x=np.unique(xy[:,0]);ex=np.unique(xe[:,0]);synth=jax.jit(data.synth_grid)
    result=dict(source_sha256=digest(source),grid=128,encoder_grid=16,final_test_opened=False,
        normalization='current-time qualified reference physical L2',groups={})
    arrays=dict(xy=xy,w=w,encoder_xy=xe)
    for group,key in [('train','training_sources'),('dev','dev_sources')]:
        fields=[];encoded=[];features=[];norms=[];sources=[];maxnorm=0
        for row in record[key]:
            assert row['case'].startswith(('train',) if group=='train' else ('dev','align'))
            path=resolve(row['file'],ns);assert digest(path)==row['sha256']
            v=np.load(path);b=jnp.asarray(v['coefficients']);p=v['parameters'];t=v['times']
            y=np.asarray(synth(b,jnp.asarray(x))).reshape(11,-1)
            ye=np.asarray(synth(b,jnp.asarray(ex))).reshape(11,-1)
            norm=np.linalg.norm(np.asarray(b).reshape(11,-1),axis=1)/2
            change=np.max(abs(np.sqrt(np.sum(y*y*w,axis=1))/norm-1));maxnorm=max(maxnorm,float(change))
            assert change<.0002,('Reference grid norm unresolved',row['case'],change)
            f=np.column_stack([np.repeat(p[None],11,axis=0),t]);f[:,3]=np.log(f[:,3])
            fields.append(y);encoded.append(ye);features.append(f);norms.append(norm);sources.append(dict(row,file=str(path)))
        arrays.update({group+'_y':np.concatenate(fields),group+'_encoder':np.concatenate(encoded),
                       group+'_features':np.concatenate(features),group+'_norm':np.concatenate(norms)})
        result['groups'][group]=dict(labels=[r['case'] for r in sources],sources=sources,
            max_reference_norm_quadrature_change=maxnorm)
        print('coordinate_data',group,len(sources),maxnorm,flush=True)
    mean=arrays['train_features'].mean(0);scale=np.maximum(arrays['train_features'].std(0),1e-8)
    arrays['feature_mean']=mean;arrays['feature_scale']=scale
    for group in ['train','dev']:arrays[group+'_features']=(arrays[group+'_features']-mean)/scale
    path=out/'coordinate_data.npz';np.savez_compressed(path,**arrays)
    result.update(status='complete_primary_coordinate_data',file=path.name,sha256=digest(path))
    save(out/'coordinate_data.json',result)
    return arrays,result
