"""First paired decoder audit; representation costs are not PDE query timings."""
import argparse
from common import *
import data,decoder as dec,geometry as geo,fit,local_study as LS
from training_io import *

def finer_dataset(directory,D,n=128):
    xy,w=data.grid(n,True);x=np.unique(xy[:,0]);fields=[]
    raw=json.loads((Path(directory)/'result.json').read_text())
    rows={r['case']:r for r in raw['cases']}
    for name in D['labels']:
        row=rows[name];path=Path(directory)/row['levels'][-1]['file'];assert digest(path)==row['levels'][-1]['sha256']
        b=np.load(path)['coefficients'];fields.append(np.asarray(data.synth_grid(jnp.asarray(b),x)).reshape(11,-1))
    E=dict(D);E.update(Y=jnp.asarray(np.concatenate(fields)),xy=jnp.asarray(xy),w=jnp.asarray(w))
    return E

def dense_field(model,pl,child,z,cells,n):
    parent=model['heads']['16'];k=child['linear'].shape[1]
    rawcoeff=jnp.asarray(model['global_transform'])@dec.head(parent,z[:16])
    localcoeff=dec.head(child,z[16:].reshape(len(cells),k))
    @jax.jit
    def block(x,rawcoeff,localcoeff):
        u=dec.global_features(model['global_params'],x)@rawcoeff
        for p,c in enumerate(cells):u=u+dec.local_features(pl,x,c)@localcoeff[p]
        return u
    axis=jnp.linspace(0.,1.,n);chunks=[]
    for i in range(0,n,32):
        xx,yy=jnp.meshgrid(axis[i:i+32],axis,indexing='ij');xy=jnp.stack([xx,yy],axis=-1).reshape(-1,2)
        chunks.append(block(xy,rawcoeff,localcoeff).reshape(-1,n))
    return jnp.concatenate(chunks)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--dev',required=True)
    ap.add_argument('--tree-model',required=True);ap.add_argument('--global-model',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    D=load_reference(a.dev,'dev');md=read_checkpoint(Path(a.tree_model)/'tree_model.pkl')
    model,pl,child,k=md['global_model'],md['local'],md['child'],md['latent'];parent=model['heads']['16']
    root=read_checkpoint(Path(a.global_model)/'global_head16_selected.pkl');zd=jnp.asarray(root['dev_codes'])
    result=dict(meta=preflight(),status='running',comparisons=[],same_checkpoint_allocations=True,
                tree_sha256=digest(Path(a.tree_model)/'tree_model.pkl'),normalization='current-time reference L2')
    stored={}
    for cap in (32,64,128):
        rows,codes=LS.evaluate_pair(model,pl,child,D,zd,k,25,3,cap)
        for row in rows:row.update(model='nonlinear_child',scoring_points_per_axis=64)
        result['comparisons']+=rows;stored[cap]=codes
        checkpoint(out/f'nonlinear_fits_cap{cap}.pkl',codes);save(out/'result.json',result)
        print('nonlinear',cap,[(r['allocation'],r['projection']['worst']) for r in rows],flush=True)
    E=finer_dataset(a.dev,D,128)
    fine,fcodes=LS.evaluate_pair(model,pl,child,E,zd,k,125,3,128)
    result['fine_projection']=fine;checkpoint(out/'fine_fits.pkl',fcodes)
    result['representation_gate']=any(row['projection']['worst']<=.005 for row in fine)
    result['passing_allocations']=[row['allocation'] for row in fine if row['projection']['worst']<=.005]
    # The trained checkpoint must independently preserve zero children and useful directions.
    x,w=E['xy'],E['w'];cells=fcodes[1]['trees'][0];G0=LS.global_matrix(model,x);B=dec.local_matrix(pl,x,cells,16)
    zparent=zd[0];zero=jnp.concatenate([zparent,jnp.zeros(len(cells)*k)])
    coeff=lambda z:dec.coefficients(parent,child,z,len(cells),k)
    field=jnp.concatenate([G0,B],axis=1)@coeff(zero);rootfield=G0@dec.head(parent,zparent)
    zeroerr=float(jnp.max(jnp.abs(field-rootfield)))
    H=jax.jacfwd(lambda z:dec.head(child,z))(jnp.zeros(k))
    J=B@jnp.kron(jnp.eye(len(cells)),H);weighted=J*jnp.sqrt(w[:,None])
    parentJ=(G0@jax.jacfwd(lambda z:dec.head(parent,z))(zparent))*jnp.sqrt(w[:,None])
    Q,_=jnp.linalg.qr(parentJ,mode='reduced');ind=weighted-Q@(Q.T@weighted)
    sv=jnp.linalg.svd(weighted,compute_uv=False);indsv=jnp.linalg.svd(ind,compute_uv=False)
    result['jacobian_zero']=dict(zero_child_max_field_change=zeroerr,singular_values=sv,
           parent_orthogonal_singular_values=indsv,rank=int(jnp.sum(sv>sv[0]*1e-10)),expected=len(cells)*k,
           useful_independent_rank=int(jnp.sum(indsv>indsv[0]*1e-10)))
    assert zeroerr<1e-13
    # Paired cached-feature component costs, all arms in this allocation/GPU.
    x,w=D['xy'],D['w'];G0=LS.global_matrix(model,x);methods={};directions={}
    for dim in (16,32,64):
        h=model['heads'][str(dim)];f=read_checkpoint(Path(a.global_model)/f'global_head{dim}_selected.pkl')
        z=jnp.asarray(f['dev_codes'][0]);co=lambda z,h=h:dec.head(h,z)
        methods[f'global_head{dim}']=(G0,co,z);directions[f'global_head{dim}']=jnp.ones_like(z)/jnp.sqrt(dim)
    for cap in (32,64,128):
        item=next(v for v in stored[cap] if v['allocation']=='prescribed');cs=item['trees'][0];z=jnp.asarray(item['codes'][0][0])
        GG=jnp.concatenate([G0,dec.local_matrix(pl,x,cs,16)],axis=1)
        co=lambda z,n=len(cs):dec.coefficients(parent,child,z,n,k)
        methods[f'tree{cap}']=(GG,co,z);directions[f'tree{cap}']=jnp.ones_like(z)/jnp.sqrt(len(z))
    Graw=dec.global_features(model['global_params'],x)
    for r in (16,32,64,128):
        c,_,_=fit.linear_fit(Graw[:,:r],D['Y'][:1],w)
        methods[f'global_direct{r}']=(Graw[:,:r],lambda z:z,c[0]);directions[f'global_direct{r}']=jnp.ones(r)/jnp.sqrt(r)
    compiled={};records={}
    for name,(GG,co,z) in methods.items():
        ff=jax.jit(lambda G,z,co=co:G@co(z));jj=jax.jit(lambda G,z,v,co=co:G@jax.jvp(co,(z,),(v,))[1])
        value=ff(GG,z);jvalue=jj(GG,z,directions[name]);jax.block_until_ready((value,jvalue))
        np.testing.assert_allclose(value,GG@co(z),rtol=1e-12,atol=1e-12)
        compiled[name]=(ff,jj);records[name]=dict(field_seconds=[],jvp_seconds=[],active_variables=len(z),
                    channels=GG.shape[1],cache_bytes=GG.nbytes,points=len(x))
    burn=jax.jit(lambda A:A@A/512+.01);A=jnp.eye(512)
    for _ in range(200):A=burn(A)
    jax.block_until_ready(A);rng=np.random.default_rng(2026111805)
    for rep in range(5):
        for name in rng.permutation(list(methods)):
            GG,co,z=methods[name];ff,jj=compiled[name]
            _,sec=timer(lambda:ff(GG,z));records[name]['field_seconds'].append(sec)
            _,sec=timer(lambda:jj(GG,z,directions[name]));records[name]['jvp_seconds'].append(sec)
    result['component_timing']=dict(contract='cached-feature field and JVP only; not complete PDE solve',methods=records)
    # Dense output witness uses the same optimized state; full4096, not an upsampled image.
    chosen=min(fine,key=lambda r:r['projection']['worst']);ci=next(i for i,v in enumerate(fcodes) if v['allocation']==chosen['allocation'])
    ii,tt=np.unravel_index(np.argmax(chosen['projection']['per_snapshot']),np.asarray(chosen['projection']['per_snapshot']).shape)
    cs=fcodes[ci]['trees'][ii];z=jnp.asarray(fcodes[ci]['codes'][ii][tt]);label=D['labels'][ii]
    rraw=json.loads((Path(a.dev)/'result.json').read_text());rr=next(r for r in rraw['cases'] if r['case']==label)
    b=jnp.asarray(np.load(Path(a.dev)/rr['levels'][-1]['file'])['coefficients'][tt]);dense={}
    for n in (2048,4096):
        pred,sec=timer(lambda:dense_field(model,pl,child,z,cs,n));ref=data.synth_grid(b,jnp.linspace(0.,1.,n))
        err=float(jnp.linalg.norm(pred-ref)/jnp.linalg.norm(ref));peak=float(jnp.max(jnp.abs(pred-ref))/jnp.max(jnp.abs(ref)))
        location=np.unravel_index(int(jnp.argmax(jnp.abs(pred-ref))),pred.shape)
        dense[str(n)]=dict(shape=list(pred.shape),relative_l2=err,relative_linf=peak,peak_index=location,
                          field_evaluation_including_compilation_seconds=sec)
        if n==4096:
            np.savez_compressed(out/'actual4096.npz',prediction=np.asarray(pred),reference=np.asarray(ref),
                      parameters=D['parameters'][ii],case=label,time=data.TIMES[tt],cells=cs,codes=np.asarray(z))
    result['dense_witness']=dict(case=label,time=data.TIMES[tt],allocation=chosen['allocation'],grids=dense,
         scoring_change=abs(dense['4096']['relative_l2']-dense['2048']['relative_l2']),PDE_solve=False)
    # Also exercise the requested final-time output shape, keeping its projection
    # status explicit: no online method was allowed to read this reference.
    zfinal=jnp.asarray(fcodes[ci]['codes'][ii][-1])
    bfinal=jnp.asarray(np.load(Path(a.dev)/rr['levels'][-1]['file'])['coefficients'][-1])
    pred,sec=timer(lambda:dense_field(model,pl,child,zfinal,cs,4096))
    ref=data.synth_grid(bfinal,jnp.linspace(0.,1.,4096))
    finalerr=float(jnp.linalg.norm(pred-ref)/jnp.linalg.norm(ref))
    np.savez_compressed(out/'actual4096-final.npz',prediction=np.asarray(pred),reference=np.asarray(ref),
                       parameters=D['parameters'][ii],case=label,time=.25,cells=cs,codes=np.asarray(zfinal))
    result['final_time_output_witness']=dict(case=label,time=.25,shape=[4096,4096],relative_l2=finalerr,
           field_evaluation_including_compilation_seconds=sec,reference_assisted_projection=True,PDE_solve=False)
    result['status']='complete_representation_audit';save(out/'result.json',result)
    print('representation_gate',result['representation_gate'],'dense',dense['4096']['relative_l2'],flush=True)
if __name__=='__main__':main()
