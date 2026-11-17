"""Fixed-space PDE diagnostic. Upwind operator exact on the stated internal mesh.
No truth-based tree choice, no final-panel admission, and complete final output cost recorded.
"""
from campaign import *
from owned_inputs import verified
import local_bank as B
import qcore as Q
import argparse,pickle
sys.path.insert(0,str(T/'jcp-fom2d'));import psjax2d as PJ
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
params=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('repair01','best.pkl'),'rb')))
V=jnp.asarray(np.load(verified('representation01','global_pod.npz'))['basis']);train,dev=draw()
rep=dict(meta=meta(),rows=[],contract='common resident 4096 input to resident 4096 final field; diagnostic single cold/warm pair',internal_mesh=128)
from spectral_cost import run_panel
rep['spectral']=run_panel(o)
# Continuous sine extension of trained coarse POD, with source-grid normalization retained.
Vc=jax.vmap(lambda v:P.analysis(v.reshape(63,63),64),in_axes=1)(V[:,:192])
L=128;xy=jnp.asarray(E.coords(L,True));patches=B.uniform(1)

def make_decoder(r,k,kind):
 @jax.jit
 def dense(c,params,Vc):
  coeff=jnp.einsum('r,rij->ij',c[:r],Vc[:r]);out=PJ.synth_mesh(coeff,4095)
  if k:
   # Chunked decoder, no persistent 4096-by-r feature table; all network work is charged.
   def body(block,out):
    xx=(block*32+jnp.arange(32))/4095;yy=jnp.arange(4096)/4095
    coords=jnp.stack(jnp.meshgrid(xx,yy,indexing='ij'),axis=-1).reshape(-1,2)
    local=B.features(coords,patches,k,kind,params)@c[r:]
    return jax.lax.dynamic_update_slice(out,jax.lax.dynamic_slice(out,(block*32,0),(32,4096))+local.reshape(32,4096),(block*32,0))
   out=jax.lax.fori_loop(0,128,body,out)
  return out.at[0,:].set(0).at[-1,:].set(0).at[:,0].set(0).at[:,-1].set(0)
 return dense
for name,r,k,kind in [('pod128',128,0,'poly'),('pod192',192,0,'poly'),('localpoly',64,32,'poly'),('localneural',64,32,'neural')]:
 base=jax.vmap(lambda b:PJ.synth_mesh(b,L)[1:-1,1:-1].reshape(-1))(Vc[:r]).T
 bank=base if not k else jnp.concatenate([base,B.features(xy,patches,k,kind,params)],axis=1)
 Rk=bank.shape[1];phi,lam,_=E.modes(L,4*Rk);phi=jnp.asarray(phi);lam=jnp.asarray(lam);A=phi.T@bank
 # Physical coefficients preserved, rank removal only in initial fit.
 _,sv,_=B.fit(bank,jnp.zeros((bank.shape[0],1)));rank=int(jnp.sum(sv>sv[0]*1e-10))
 def residual(c,prev,nu,bank,phi,A,lam):
  u=bank@c;adv,_=E.spatial(u,L)
  return (A@c-prev+.005*(phi.T@adv+nu*lam*(A@c)))/(1+.005*nu*lam)
 def evalJ(c,args):
  return residual(c,*args),jax.jacfwd(residual)(c,*args)
 lm=Q.HF.make_fused_lm(evalJ,Rk,0,80,np.inf,1e-6,1e-10,'chol',False)
 @jax.jit
 def solve(u0,nu,bank,phi,A,lam):
  from jax.scipy.ndimage import map_coordinates
  z=map_coordinates(u0,(xy*4095).T,order=1,mode='nearest')
  c,_,_=B.fit(bank,z[:,None]);c=c[:,0]
  def step(carry,i):
   c,prev,lam0=carry;p=A@c
   new,rn,it,reason,gn,rej,lam1=lm(c,(p,nu,bank,phi,A,lam),1e-10*jnp.linalg.norm(z),lam0)
   return (new,c,lam1),(new,rn,it,reason,gn,rej)
  _,v=jax.lax.scan(step,(c,c,jnp.float64(1e-6)),jnp.arange(50))
  return jnp.concatenate([c[None],v[0][9::10]]),v[1:]
 decode=make_decoder(r,k,kind)
 for i,ph in enumerate(dev[:8]):
  u0=jnp.asarray(E.initial(4095,ph));jax.block_until_ready(u0)
  def query():
   cs,stats=solve(u0,ph[4],bank,phi,A,lam);field=decode(cs[-1],params,Vc);return cs,stats,field
  v,coldsec=timer(query);E.burn(.4);v,warmsec=timer(query)
  cs,stats,field=v;pred=np.asarray((bank@cs.T).T).reshape(6,L-1,L-1)
  ref=np.load(verified('reference01',f'dev{i}_fd512_dt0.000625.npz'))['out'].reshape(6,511,511)[:,3::4,3::4]
  error=np.linalg.norm((pred-ref).reshape(6,-1),axis=1)/np.linalg.norm(ref[0])
  dense_ref=jnp.asarray(np.load(verified('dense01',f'dev{i}_final4096.npz'))['field'])
  final_error=float(jnp.linalg.norm(field-dense_ref)/jnp.linalg.norm(u0))
  # Save full final field for first two cases, coefficients reconstruct every remaining return.
  np.savez_compressed(o/f'{name}_{i}.npz',coefficients=cs,residual=stats[0],iterations=stats[1],reason=stats[2],stationarity=stats[3],rejected=stats[4],**({'final4096':field} if i<2 else {}))
  rep['rows'].append(dict(method=name,case=i,rank=rank,columns=Rk,error=error,cold_seconds=coldsec,warm_seconds=warmsec,final4096_error=final_error,output_shape=list(field.shape),output_norm=float(jnp.linalg.norm(field)),max_stationarity=float(jnp.max(stats[3])),reasons=np.asarray(stats[2])))
  save(o/'result.json',rep);print(name,i,error[1:].max(),warmsec,flush=True)
rep['complete']=True;save(o/'result.json',rep)
