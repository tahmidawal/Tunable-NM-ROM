"""Tune qualified family-B local operators without mistaking conservative costs for a frontier."""
from campaign import *
from owned_inputs import verified
import continuous as C,family_b as FB,quadtree as QT,argparse,itertools
from jax.scipy.ndimage import map_coordinates
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
params=json.loads(verified('c1familyb','parameters.json').read_text())['dev'][:8];xy=jnp.asarray(E.coords(512,True));inputs=[];orders=[];truths=[];norms=[];rows=[];solvers={dt:C.flux_trajectory(dt) for dt in (.025,.0125,.00625)}
for case,ph in enumerate(params):
 u=jnp.asarray(FB.initial(4095,ph));orders.append(tuple(sorted(QT.initial_order(u)[:2])));inputs.append(P.analysis(map_coordinates(u,(xy*4095).T,order=1,mode='nearest').reshape(511,511),512));d=np.load(verified('c1familyb',f'case{32+case}.npz'));truths.append(jnp.asarray(d['coefficients']));norms.append(float(d['initial_norm']))
for split in list(itertools.combinations(range(4),2))+[(0,1,2,3)]:
 cases=list(range(8)) if len(split)==4 else [i for i,x in enumerate(orders) if x==split]
 if not cases:continue
 name='uniform1152' if len(split)==4 else 'tree768';d=np.load(verified('c1bcomplete','split'+''.join(map(str,split))+'_basis.npz'));W=jnp.asarray(d['basis']);e=jnp.asarray(d['eigenvalues'])
 for q in (64,96,128,192,256):
  G,Hq=C.flux_tables(W,128,q)
  for case in cases:
   a0=W.T@inputs[case][:127,:127].ravel()/2;truth=truths[case];norm=norms[case]
   for dt in (.025,.0125,.00625):
    states,sec=timer(lambda:solvers[dt](a0,params[case]['nu'],G,Hq,e));pred=jnp.pad((2*(states@W.T)).reshape(11,127,127),((0,0),(0,128),(0,128)));err=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/(2*norm);np.savez_compressed(o/f'{name}_q{q}_dt{dt}_case{case}.npz',states=states,error=err);rows.append(dict(name=name,case=case,q=q,dt=dt,error=err,evolved=float(err[[2,4,6,8,10]].max()),seconds=sec,finite=bool(np.isfinite(states).all())))
  save(o/'result.json',dict(meta=meta(),rows=rows));print(name,split,q,flush=True)
def classical(N,dt,over):
 lam=C.eigenvalues(N).reshape(N-1,N-1)
 @jax.jit
 def run(b,nu):
  co=P.etdrk4_coefs(-nu*lam,dt,32)
  def nl(v):return P.nonlinear(jnp.pad(v,((0,N*(over-1)),(0,N*(over-1)))),N*over,2)[0][:N-1,:N-1]
  def step(v):
   nv=nl(v);a=co['E2']*v+co['Q']*nv;na=nl(a);b=co['E2']*v+co['Q']*na;nb=nl(b);c=co['E2']*a+co['Q']*(2*nb-nv);nc=nl(c);return co['E']*v+co['f1']*nv+2*co['f2']*(na+nb)+co['f3']*nc
  def block(v,_):
   v=jax.lax.fori_loop(0,round(.05/dt),lambda i,x:step(x),v);return v,v
  _,s=jax.lax.scan(block,b,None,length=5);return jnp.concatenate([b[None],s])
 return run
classrows=[]
for N in (32,48,64):
 for dt,over in [(.01,1),(.005,1),(.01,2)]:
  f=classical(N,dt,over)
  for case,ph in enumerate(params):
   states,sec=timer(lambda:f(inputs[case][:N-1,:N-1],ph['nu']));pred=jnp.pad(states,((0,0),(0,256-N),(0,256-N)));err=np.asarray(jnp.linalg.norm((pred-truths[case][::2]).reshape(6,-1),axis=1))/(2*norms[case]);np.savez_compressed(o/f'classical{N}_dt{dt}_over{over}_case{case}.npz',coefficients=states,error=err);classrows.append(dict(N=N,case=case,dt=dt,over=over,error=err,evolved=float(err[1:].max()),seconds=sec))
  save(o/'classical.json',dict(rows=classrows));print('classical',N,dt,over,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,classical=classrows,complete=True,contract='same-input familyB quadrature/time tuning, all failures retained; updated classical timestep controls; prepared-input screen, no cost claim'))
