"""Classical sine-mode/time tuning with separate nonlinear integration check."""
from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
def solver(N,dt,over):
 lam=C.eigenvalues(N).reshape(N-1,N-1)
 @jax.jit
 def run(b,nu):
  co=P.etdrk4_coefs(-nu*lam,dt,32)
  def nonlin(v):
   v=jnp.pad(v,((0,N*(over-1)),(0,N*(over-1))));f,_=P.nonlinear(v,N*over,2);return f[:N-1,:N-1]
  def step(v):
   nv=nonlin(v);a=co['E2']*v+co['Q']*nv;na=nonlin(a);b=co['E2']*v+co['Q']*na;nb=nonlin(b);c=co['E2']*a+co['Q']*(2*nb-nv);nc=nonlin(c)
   return co['E']*v+co['f1']*nv+2*co['f2']*(na+nb)+co['f3']*nc
  def block(v,_):
   v=jax.lax.fori_loop(0,round(.025/dt),lambda i,x:step(x),v);return v,v
  _,states=jax.lax.scan(block,b,None,length=10);return jnp.concatenate([b[None],states])
 return run
_,dev=draw();rows=[]
for N in (24,32,48):
 for dt,over in [(h,1) for h in (.025,.0125,.00625)]+[(.0125,2)]:
  f=solver(N,dt,over)
  for i,ph in enumerate(dev[:32]):
   d=np.load(verified('c1data',f'check{128+i}.npz'));truth=jnp.asarray(d['coefficients']);norm=float(np.load(verified('c1data',f'case{128+i}.npz'))['initial_norm']);b0=truth[0,:N-1,:N-1]
   states,sec=timer(lambda:f(b0,ph[4]));pred=jnp.pad(states,((0,0),(0,256-N),(0,256-N)));err=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/(2*norm)
   np.savez_compressed(o/f'N{N}_dt{dt}_over{over}_case{i}.npz',coefficients=states,error=err)
   rows.append(dict(N=N,dt=dt,over=over,case=i,error=err,seconds=sec,finite=bool(np.isfinite(states).all()),evolved=float(err[[2,4,6,8,10]].max())))
  save(o/'result.json',dict(meta=meta(),rows=rows));print(N,dt,over,max(x['evolved'] for x in rows if x['N']==N and x['dt']==dt and x['over']==over),flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='classical mode/time screen with distinct overintegration check; coefficient inputs, not complete timing'))
