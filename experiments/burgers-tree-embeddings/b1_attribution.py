"""Separate parent, local-span and nonlinear-training limitations; no final test."""
import argparse
from common import *
import data,decoder as dec,geometry as geo,local_study as LS,fit
from training_io import *
from evaluate import finer_dataset

def spectral_dataset(directory):
    directory=Path(directory);assert str(directory).startswith(os.environ['STUDY_NAMESPACE']+'/archive/')
    raw=json.loads((directory/'result.json').read_text());rows=[r for r in raw['cases'] if not r['case'].startswith('fixture')]
    fields=[];norm=[];tail=[];labels=[];parameters=[];sources=[]
    for r in rows:
        p=directory/r['levels'][-1]['file'];assert digest(p)==r['levels'][-1]['sha256']
        b=np.load(p)['coefficients'];small=b[:,:128,:128]
        total=np.sum(b*b,axis=(1,2))*.25;kept=np.sum(small*small,axis=(1,2))*.25
        fields.append(small.reshape(11,-1));norm.append(np.sqrt(total));tail.append(np.maximum(total-kept,0))
        labels.append(r['case']);parameters.append(r['parameters']);sources.append(dict(file=str(p),sha256=digest(p)))
    return dict(A=jnp.asarray(np.concatenate(fields)),norm=jnp.asarray(np.concatenate(norm)),tail=jnp.asarray(np.concatenate(tail)),
                labels=labels,parameters=parameters,sources=sources)

def polynomials(x,cells,side):
    blocks=[]
    for c in cells:
        v=geo.local_inputs(x,c)[...,:2]/1.5
        def recur(t):
            p=[jnp.ones_like(t),t]
            for n in range(2,side):p.append(((2*n-1)*t*p[-1]-(n-1)*p[-2])/n)
            return jnp.stack(p[:side],axis=-1)
        px,py=recur(v[:,0]),recur(v[:,1]);b=(px[:,:,None]*py[:,None,:]).reshape(len(x),side*side)
        blocks.append(geo.window(x,c)[:,None]*b)
    return jnp.concatenate(blocks,axis=1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    for p in ['train','dev','global-model','local-model','audit']:ap.add_argument('--'+p,required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    result=dict(meta=preflight(),status='running',normalization='current-time reference L2',final_test_opened=False,attribution=[])
    T=spectral_dataset(a.train);D=spectral_dataset(a.dev)
    normalized=T['A']/(2*T['norm'][:,None])
    U,s,Vh=jax.jit(lambda x:jnp.linalg.svd(x,full_matrices=False))(normalized)
    basis=Vh[:512];Ct=T['A']@basis.T*.5;Cd=D['A']@basis.T*.5
    result['spectral_POD_controls']={}
    for rank in (16,32,64,128,256,384,512):
        delta=D['A']-2*Cd[:,:rank]@basis[:rank]
        err=jnp.sqrt(jnp.sum(delta*delta,axis=1)*.25+D['tail'])/D['norm']
        result['spectral_POD_controls'][str(rank)]=summary(err.reshape(-1,11))
    result['reference_truncation_worst']=float(jnp.max(jnp.sqrt(D['tail'])/D['norm']))
    result['training_sources']=T['sources'];result['dev_sources']=D['sources']
    result['dev_labels']=D['labels'];result['training_labels']=T['labels']
    np.savez_compressed(out/'pod_basis.npz',coefficients=2*np.asarray(basis).reshape(512,128,128),singular_values=np.asarray(s))
    train_missing=jnp.maximum(T['norm']**2-jnp.sum(Ct*Ct,axis=1),0)
    dev_missing=jnp.maximum(D['norm']**2-jnp.sum(Cd*Cd,axis=1),0)
    np.savez_compressed(out/'reduced_data.npz',train_coefficients=np.asarray(Ct),dev_coefficients=np.asarray(Cd),
        train_norm=np.asarray(T['norm']),dev_norm=np.asarray(D['norm']),train_missing=np.asarray(train_missing),
        dev_missing=np.asarray(dev_missing),train_parameters=np.asarray(T['parameters']),dev_parameters=np.asarray(D['parameters']))
    result['basis_sha256']=digest(out/'pod_basis.npz');result['reduced_data_sha256']=digest(out/'reduced_data.npz')
    save(out/'result.json',result);print('POD', {k:v['worst'] for k,v in result['spectral_POD_controls'].items()},flush=True)
    old=read_checkpoint(Path(a.global_model)/'global_model.pkl');gf=read_checkpoint(Path(a.global_model)/'global_head16_selected.pkl')
    root=old['heads']['16'];z=jnp.asarray(gf['train_codes']);c=dec.head(root,z)
    affine=jnp.concatenate([jnp.ones((len(z),1)),z],axis=1);beta=jnp.linalg.lstsq(affine,c,rcond=1e-12)[0]
    result['old_parent_nonlinearity']=dict(latent_std=np.asarray(jnp.std(z,axis=0)),latent_abs_max=float(jnp.max(jnp.abs(z))),
        relative_departure_from_best_affine=float(jnp.linalg.norm(c-affine@beta)/jnp.linalg.norm(c)),
        old_parent_selected_update=read_checkpoint(Path(a.global_model)/'global_head16_selected.pkl')['selected_update'])
    DD=finer_dataset(a.dev,load_reference(a.dev,'dev'),128);x,w,Y=DD['xy'],DD['w'],DD['Y']
    model=read_checkpoint(Path(a.local_model)/'local_model.pkl');pl=model['local'];G0=LS.global_matrix(old,x)
    fits=read_checkpoint(Path(a.audit)/'direct_fine_fits.pkl');pres=next(r for r in fits if r['allocation']=='prescribed')
    # Same trees/checkpoint; only the indicated coefficient constraints are released.
    for kind in ['parent128_local4','parent128_local16','parent16_local16','parent128_poly36']:
        errors_all=[];stat=[];nvar=[]
        for i,p in enumerate(DD['parameters']):
            cs=pres['trees'][i];sl=slice(i*11,(i+1)*11)
            if kind=='parent16_local16':
                f=LS.project_tree(old,pl,None,x,w,Y[sl],jnp.asarray(pres['codes'][i])[:,:16],cs,16,100,3)
                err=f['error'];stat.append(np.asarray(f['stationarity']));nvar.append(16+len(cs)*16)
            else:
                rank=4 if kind=='parent128_local4' else (36 if kind=='parent128_poly36' else 16)
                B=polynomials(x,cs,6) if rank==36 else dec.local_matrix(pl,x,cs,rank)
                G=jnp.concatenate([G0,B],axis=1);c,pred,sv=fit.linear_fit(G,Y[sl],w)
                err=errors(pred,Y[sl],w);nvar.append(G.shape[1])
            errors_all.append(np.asarray(err))
        row=dict(kind=kind,projection=summary(np.asarray(errors_all)),variables=nvar,
                 stationarity=stat,scoring=128,trees='identical R1 prescribed trees',eligible_at_original_cap=False)
        result['attribution'].append(row);save(out/'result.json',result);print(kind,row['projection']['worst'],flush=True)
    result['status']='complete_attribution_and_POD_span';save(out/'result.json',result)

if __name__=='__main__':main()
