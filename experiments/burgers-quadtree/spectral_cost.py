"""Array-input spectral comparator, same resident dense input/output contract."""
from campaign import *
from owned_inputs import verified
sys.path.insert(0,str(T/'jcp-fom2d'));import psjax2d as PJ

def make(N=64,dt=.00125,restriction=512):
 xy=jnp.asarray(E.coords(restriction,True));kv=jnp.arange(1,N,dtype=jnp.float64)
 eig=-jnp.pi**2*(kv[:,None]**2+kv[None,:]**2)
 @jax.jit
 def query(u0,nu):
  from jax.scipy.ndimage import map_coordinates
  initial=map_coordinates(u0,(xy*4095).T,order=1,mode='nearest').reshape(restriction-1,restriction-1)
  b=P.analysis(initial,restriction)[:N-1,:N-1];co=P.etdrk4_coefs(nu*eig,dt,32)
  def step(v):
   nv,_=P.nonlinear(v,N,2);a=co['E2']*v+co['Q']*nv;na,_=P.nonlinear(a,N,2)
   b=co['E2']*v+co['Q']*na;nb,_=P.nonlinear(b,N,2);c=co['E2']*a+co['Q']*(2*nb-nv);nc,_=P.nonlinear(c,N,2)
   return co['E']*v+co['f1']*nv+2*co['f2']*(na+nb)+co['f3']*nc
  def interval(v,_):
   v=jax.lax.fori_loop(0,round(.05/dt),lambda i,v:step(v),v);return v,v
  final,cs=jax.lax.scan(interval,b,None,length=5)
  return jnp.concatenate([b[None],cs]),PJ.synth_mesh(final,4095)
 return query

def run_panel(o):
 _,dev=draw();rows=[]
 for N,dt,restriction in [(64,.00125,512),(128,.000625,1024)]:
  query=make(N,dt,restriction)
  for i,ph in enumerate(dev[:8]):
   initial=jnp.asarray(E.initial(4095,ph));jax.block_until_ready(initial)
   v,cold=timer(lambda:query(initial,ph[4]));times=[];errs=[];dense_errors=[]
   fine=np.load(verified('reference01',f'dev{i}_fd512_dt0.000625.npz'))['out'].reshape(6,511,511)
   fine_coeff=jax.vmap(lambda u:P.analysis(u,512))(jnp.asarray(fine[1:]))
   fine_dense=[PJ.synth_mesh(c,4095) for c in fine_coeff]
   for rep in range(5):
    E.burn(.4);v,sec=timer(lambda:query(initial,ph[4]));times.append(sec)
    cs,field=v;pred=np.asarray(jax.vmap(lambda c:PJ.synth_mesh(c,256))(cs))
    ref=np.load(verified('reference01',f'dev{i}_fd512_dt0.000625.npz'))['out'].reshape(6,511,511)
    ref=np.pad(ref,((0,0),(1,1),(1,1)))[:,::2,::2]
    dense_errors.append([float(jnp.linalg.norm(PJ.synth_mesh(cs[t+1],4095)-fine_dense[t])/jnp.linalg.norm(initial)) for t in range(5)])
    errs.append((np.linalg.norm((pred-ref).reshape(6,-1),axis=1)/np.linalg.norm(ref[0])).tolist())
    np.savez_compressed(o/f'spectral{N}_case{i}_rep{rep}.npz',coefficients=cs,output_norm=float(jnp.linalg.norm(field)))
   if i==0:np.savez_compressed(o/f'spectral{N}_final4096.npz',field=field)
   rows.append(dict(N=N,dt=dt,restriction=restriction,case=i,cold_seconds=cold,seconds=times,error_per_repetition=errs,dense4096_errors=dense_errors,output_shape=list(field.shape)))
   save(o/'spectral.json',dict(meta=meta(),rows=rows));print('spectral',N,i,max(errs[-1][1:]),np.median(times),flush=True)
 return rows

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);run_panel(out)
