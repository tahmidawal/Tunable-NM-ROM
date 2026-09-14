"""Frozen-weight cold-fit objective audit; analytic ICs only, no rollout claim.

The full-grid QR is an explicit diagnostic, not a proposed grid-independent
online initializer. Fixed physical midpoint/Gauss rules sample input fields
by charged aligned bilinear interpolation and never sample a PDE residual.
"""
import argparse,hashlib,json,os,pickle,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import engines as e


def sample_field(field,xy,L):
    x=xy*L;lo=jnp.minimum(jnp.floor(x).astype(jnp.int32),L-1);t=x-lo
    i,j=lo[:,0],lo[:,1];a,b=t[:,0],t[:,1]
    return (1-a)*(1-b)*field[i,j]+a*(1-b)*field[i+1,j]+(1-a)*b*field[i,j+1]+a*b*field[i+1,j+1]


def make_fit(params,L,budget,full=False,gram=False,starts=4):
    K=params['h_lin'].shape[0]
    lm=e.make_lm(lambda z,y,R:R@e.sc.head(params,z)-y,K,budget,stall=1e-7)
    def query(input_field,G,xy,weights,Q,R,Gi,Z,Hrot,Hnorm):
        vals=input_field[1:-1,1:-1].reshape(-1)*weights if full else sample_field(input_field,xy,L)*weights
        if gram:
            b=Gi.T@vals;y=jax.scipy.linalg.solve_triangular(R.T,b,lower=True)
        else:y=Q.T@vals
        score=Hnorm-2*Hrot@y;ids=jnp.argsort(score)[:starts]
        zs,rns,its,reasons=jax.vmap(lambda z:lm(z,(y,R),0.))(Z[ids]);best=jnp.argmin(rns)
        z=zs[best];r=R@e.sc.head(params,z)-y;J=jax.jacfwd(lambda z:R@e.sc.head(params,z))(z)
        grad=jnp.linalg.norm(J.T@r)/(jnp.linalg.norm(J)*jnp.linalg.norm(r)+1e-300)
        return e.output_field(G@e.sc.head(params,z),L,L),z,rns[best],its[best],reasons[best],grad
    return jax.jit(query)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--meshes',default='256,512,1024');p.add_argument('--cases',type=int,default=4);p.add_argument('--seed',type=int,default=7090702)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend()=='gpu' and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    ck=pickle.load(open(a.checkpoint,'rb'));params=jax.tree_util.tree_map(jnp.asarray,ck['params']);Z=ck['Z_tr'];K,R=Z.shape[1],params['h_lin'].shape[1]
    dec=e.sc.SeparableDecoder(params,K,R);candidates=jnp.asarray(Z[::max(1,len(Z)//8192)]);H=e.sc.head(params,candidates)
    physical=e.params_draw(a.seed,a.cases);meshes=list(map(int,a.meshes.split(',')));obs=min(meshes)
    d=dict(config=vars(a),commit=os.environ['COMMIT'],job_id=os.environ['SLURM_JOB_ID'],backend=jax.default_backend(),
        gpu=jax.devices()[0].device_kind,x64=jax.config.jax_enable_x64,matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),weights_frozen=True,physical_cases=physical.tolist(),
        scope='analytic initial-state fitting only; no autonomous rollout or physical-query speed claim',mesh_setup=[],rows=[],complete=False)
    def save():(out/'cold_fit.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    save();begin=time.perf_counter()
    for L in meshes:
        print('COLD SETUP',L,flush=True);t0=time.perf_counter();G=dec.feat_at(e.coords(L),chunk=8192);jax.block_until_ready(G)
        nq=48
        edge=np.rint(np.linspace(1,L-1,nq)).astype(int)/L
        mid=(np.arange(nq)+.5)/nq
        nodes,weights=np.polynomial.legendre.leggauss(nq);gauss=(nodes+1)/2;gw=weights/2
        # QR supplies an exact diagnostic for the inherited learned bank, not POD.
        Qfull,Rfull=jax.block_until_ready(jnp.linalg.qr(G/(L-1),mode='reduced'))
        rows=np.arange(0,len(G),max(1,len(G)//4096))
        qrdev=float(jnp.linalg.norm(Qfull[rows]@Rfull-G[rows]/(L-1))/jnp.linalg.norm(G[rows]/(L-1)))
        assert qrdev<1e-11
        d['mesh_setup'].append(dict(intervals=L,setup_seconds=time.perf_counter()-t0,qr_relative_reconstruction=qrdev,
            bank_R=R,latent_K=K,quadrature_axis_points=nq,grid_nodes=(L+1)**2))
        rules=[]
        for name,axis,waxis in [('edge_gram',edge,np.ones(nq)/nq),('edge_qr',edge,np.ones(nq)/nq),
                                ('fixed_midpoint',mid,np.ones(nq)/nq),('fixed_gauss',gauss,gw)]:
            xy=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
            w=np.sqrt(np.outer(waxis,waxis).ravel());xyj=jnp.asarray(xy);wj=jnp.asarray(w)
            Gi=dec.feat_at(xy)*wj[:,None]
            if name=='edge_gram':
                gram=Gi.T@Gi;rot=jnp.linalg.cholesky(gram+1e-12*jnp.trace(gram)/R*jnp.eye(R)).T;Q=jnp.zeros((1,1))
            else:Q,rot=jnp.linalg.qr(Gi,mode='reduced')
            hr=H@rot.T
            rules.append((name,xyj,wj,Q,rot,Gi,hr,jnp.sum(hr*hr,1)))
        hr=H@Rfull.T
        rules.append(('full_qr',jnp.zeros((1,2)),jnp.asarray(1/(L-1)),Qfull,Rfull,jnp.zeros((1,1)),hr,jnp.sum(hr*hr,1)))
        inputs=[e.initial(L,p) for p in physical]
        floors=[]
        for u0 in inputs:
            fine=jnp.asarray(u0[1:-1,1:-1].reshape(-1)/(L-1));coef=Qfull.T@fine
            floors.append(float(jnp.sqrt(jnp.maximum(jnp.dot(fine,fine)-jnp.dot(coef,coef),0))/jnp.linalg.norm(fine)))
        for name,xy,w,Q,rot,Gi,hr,hnorm in rules:
            for budget in [60,180]:
                for starts in [1,4]:
                    fun=make_fit(params,L,budget,full=name=='full_qr',gram=name=='edge_gram',starts=starts)
                    jax.block_until_ready(fun(jnp.asarray(inputs[0]),G,xy,w,Q,rot,Gi,candidates,hr,hnorm));e.burn(.4)
                    for case,u0 in enumerate(inputs):
                        t=time.perf_counter();field,z,rn,it,reason,grad=jax.tree_util.tree_map(np.asarray,
                            fun(jnp.asarray(u0),G,xy,w,Q,rot,Gi,candidates,hr,hnorm));elapsed=time.perf_counter()-t
                        err=float(np.linalg.norm(field-u0)/np.linalg.norm(u0))
                        artifact=f'L{L}_{name}_budget{budget}_starts{starts}_case{case}.npz'
                        np.savez_compressed(out/artifact,field=field[::L//obs,::L//obs],truth=u0[::L//obs,::L//obs],latent=z)
                        d['rows'].append(dict(intervals=L,rule=name,budget=budget,starts=starts,case=case,seconds=elapsed,
                            relative_full_grid_error=err,relative_common_grid_error=float(np.linalg.norm((field-u0)[::L//obs,::L//obs])/np.linalg.norm(u0[::L//obs,::L//obs])),
                            unrestricted_bank_floor=floors[case],weighted_fit_norm=float(rn),iterations=int(it),stop_reason=int(reason),
                            relative_gradient=float(grad),artifact=artifact,minimum=float(field.min())))
                    print('COLD',L,name,budget,starts,'done',flush=True);save()
        del G,Qfull,Rfull,rules; jax.clear_caches()
    d['complete']=True;d['total_seconds']=time.perf_counter()-begin;save();print('ALL-DONE',flush=True)


if __name__=='__main__':main()
