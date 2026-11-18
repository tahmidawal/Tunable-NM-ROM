"""Fresh-process component memory and trained-feature arithmetic checks, no PDE rollout."""
from pathlib import Path
import argparse,json,subprocess,sys

MODES=['global_direct128','global_head16','global_head32','global_head64',
       'direct_uniform','direct_prescribed','nonlinear_uniform','nonlinear_prescribed']

def worker(a):
    from common import jax,jnp,np,preflight,save,digest
    import data,decoder as dec,local_study as LS,operators as O
    from training_io import read_checkpoint
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    meta=preflight();device=jax.devices()[0]
    md=read_checkpoint(Path(a.tree_model)/'tree_model.pkl')
    admitted=json.loads((Path(a.audit)/'result.json').read_text())
    assert digest(Path(a.tree_model)/'tree_model.pkl')==admitted['tree_sha256']
    model,pl,child=md['global_model'],md['local'],md['child']
    xy,w=data.grid(128,True);xy=jnp.asarray(xy);parent=model['heads']['16']
    local=False;cells=[]
    if a.mode=='global_direct128':
        G=dec.global_features(model['global_params'],xy);z=jnp.ones(128)*.01
        coefficient=lambda z:z
        used_params=dec.nparameters(model['global_params'])
    elif a.mode.startswith('global_head'):
        k=int(a.mode.removeprefix('global_head'));head=model['heads'][str(k)]
        root=read_checkpoint(Path(a.global_model)/f'global_head{k}_selected.pkl')
        G=LS.global_matrix(model,xy);z=jnp.asarray(root['dev_codes'][0]);coefficient=lambda z:dec.head(head,z)
        used_params=dec.nparameters(model['global_params'])+dec.nparameters(head)
    else:
        local=True;kind,allocation=a.mode.split('_');nonlinear=kind=='nonlinear'
        fits=read_checkpoint(Path(a.audit)/('fine_fits.pkl' if nonlinear else 'direct_fine_fits.pkl'))
        row=next(r for r in fits if r['allocation']==allocation)
        cells=row['trees'][0];z=jnp.asarray(row['codes'][0][0]);activechild=child if nonlinear else None
        rank=16 if nonlinear else 4
        G=jnp.concatenate([LS.global_matrix(model,xy),dec.local_matrix(pl,xy,cells,rank)],axis=1)
        coefficient=lambda z:dec.coefficients(parent,activechild,z,len(cells),4)
        used_params=dec.nparameters(model['global_params'])+dec.nparameters(parent)+dec.nparameters(pl)
        if nonlinear:used_params+=dec.nparameters(child)
    field=jax.jit(lambda matrix,z:matrix@coefficient(z))
    jvp=jax.jit(lambda matrix,z,v:matrix@jax.jvp(coefficient,(z,),(v,))[1])
    value=field(G,z);tangent=jvp(G,z,jnp.ones_like(z)/jnp.sqrt(z.size));jax.block_until_ready((value,tangent))
    np.testing.assert_allclose(value,G@coefficient(z),rtol=1e-12,atol=1e-12)
    stats=device.memory_stats()
    result=dict(meta=meta,mode=a.mode,active_variables=int(z.size),feature_channels=int(G.shape[1]),
        integration_points=len(xy),stored_feature_bytes=int(G.nbytes),network_parameters=int(used_params),
        fixed_global_transform_entries=int(np.asarray(model['global_transform']).size),
        retained_local_nodes=len(cells),inactive_padding_entries=0,
        memory_scope='fresh process through feature construction and synchronized cached field/JVP; includes loading common checkpoint; no projection/PDE/dense4096 output',
        device_memory_stats=stats,timed=False)
    if a.mode=='nonlinear_prescribed':
        # Same physical features and retained parent; zero additions also preserve spatial derivatives.
        def physical(points):
            return jnp.concatenate([LS.global_matrix(model,points),dec.local_matrix(pl,points,cells,16)],axis=1)
        zzero=z.at[16:].set(0.);c0=coefficient(zzero)
        def fullpoint(pt):return (physical(pt[None])@c0)[0]
        def rootpoint(pt):return (LS.global_matrix(model,pt[None])@dec.head(parent,z[:16]))[0]
        points=jnp.asarray([[.25,.35],[.5,.5],[.317,.193],[.125,.31]])
        delta=jax.vmap(fullpoint)(points)-jax.vmap(rootpoint)(points)
        gd=jax.vmap(jax.grad(fullpoint))(points)-jax.vmap(jax.grad(rootpoint))(points)
        hd=jax.vmap(jax.hessian(fullpoint))(points)-jax.vmap(jax.hessian(rootpoint))(points)
        result['zero_child']=dict(value_max=float(jnp.max(jnp.abs(delta))),gradient_max=float(jnp.max(jnp.abs(gd))),hessian_max=float(jnp.max(jnp.abs(hd))))
        assert max(result['zero_child'].values())<1e-10
        L=16;P,lam,x=O.modes(L,64);x=jnp.asarray(x);GG=physical(x);coeff=coefficient(z)
        u=GG@coeff;adv,lap=O.spatial(u,L);A=P.T@GG
        offsets=jnp.asarray([[0,0],[1,0],[-1,0],[0,1],[0,-1]])/L
        G5=physical((x[:,None]+offsets[None]).reshape(-1,2)).reshape(len(x),5,-1)
        previous=.97*coeff;dt=.005;nu=.005
        tested=O.tested(coeff,A@previous,A,lam,G5,P,L,nu,dt)
        dense=(P.T@(u-GG@previous+dt*(adv-nu*lap)))/(1+dt*nu*lam)
        discrepancy=float(jnp.max(jnp.abs(tested-dense)))
        assert discrepancy<1e-11
        result['trained_tested_fixture']=dict(L=L,M=64,points=len(x),fine_stencil_spacing=1/L,
                 max_absolute_discrepancy=discrepancy,production_PDE_qualified=False)
        result['device_memory_after_extra_fixtures']=device.memory_stats()
    save(out/(a.mode+'.json'),result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    for s in ['tree-model','audit','global-model']:ap.add_argument('--'+s,required=True)
    ap.add_argument('--mode',choices=MODES);a=ap.parse_args()
    if a.mode:return worker(a)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for mode in MODES:
        subprocess.run([sys.executable,__file__,'--out',a.out,'--tree-model',a.tree_model,
                       '--audit',a.audit,'--global-model',a.global_model,'--mode',mode],check=True)
    rows=[json.loads((out/(mode+'.json')).read_text()) for mode in MODES]
    (out/'result.json').write_text(json.dumps(dict(status='complete_component_memory_and_trained_fixtures',rows=rows),indent=2)+'\n')
    print('all component memory and trained fixtures complete',flush=True)

if __name__=='__main__':main()
