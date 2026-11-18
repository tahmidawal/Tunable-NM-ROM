"""Family B complete bounded-tree cache, finer-space qualification and output costs."""
from campaign import *
from owned_inputs import verified
from spectral_cost import PJ,make
import continuous as C,specialized as S,local_bank as B,family_b as FB,quadtree as QT,pickle,argparse,itertools
from jax.scipy.ndimage import map_coordinates
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
params=json.loads(verified('c1familyb','parameters.json').read_text())['dev'][:8];p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1bbank','network.pkl'),'rb')));V=jnp.asarray(np.load(verified('c1bbank','pod.npz'))['basis'])[:,:128];configs=list(itertools.combinations(range(4),2))+[(0,1,2,3)];models={};resolutions=[];setup=time.perf_counter();points=jnp.asarray(E.coords(128,True));blocks={}
for cell in set(QT.leaves([])+QT.leaves(range(4))):
 F=jnp.concatenate([B.features(points,[cell.patch()],32,'poly'),S.features(p,points,[cell.patch()],32)],axis=1);blocks[cell]=jax.vmap(lambda f:P.analysis(f.reshape(127,127),128).ravel()/2,in_axes=1,out_axes=1)(F)
for split in configs:
 A=jnp.concatenate([V]+[blocks[cell] for cell in QT.leaves(split)],axis=1);Q,s,_=jnp.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10;W,e,_=C.diffusion_basis(Q[:,keep],128);G,H=C.flux_tables(W,128,256);jax.block_until_ready((W,e,G,H));models[split]=(W,e,G,H);np.savez_compressed(o/('split'+''.join(map(str,split))+'_basis.npz'),basis=W,eigenvalues=e)
setup_seconds=time.perf_counter()-setup
# All exposed configurations re-evaluated with identical physical columns atN256.
Vfine=jnp.pad(V.reshape(127,127,128),((0,128),(0,128),(0,0))).reshape(255**2,128);points=jnp.asarray(E.coords(256,True));fineblocks={}
for cell in blocks:
 F=jnp.concatenate([B.features(points,[cell.patch()],32,'poly'),S.features(p,points,[cell.patch()],32)],axis=1);fineblocks[cell]=jax.vmap(lambda f:P.analysis(f.reshape(255,255),256).ravel()/2,in_axes=1,out_axes=1)(F)
orders=[QT.initial_order(FB.initial(4095,ph)) for ph in params]
for split in configs:
 cases=[i for i,order in enumerate(orders) if len(split)==4 or tuple(sorted(order[:2]))==split]
 if not cases:continue
 A=jnp.concatenate([Vfine]+[fineblocks[cell] for cell in QT.leaves(split)],axis=1);Q,s,_=jnp.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10;W,e,_=C.diffusion_basis(Q[:,keep],256);G,H=C.flux_tables(W,256,384);oldW,olde,oldG,oldH=models[split]
 for dt in (.00625,.003125):
  f=C.flux_trajectory(dt)
  for case in cases:
   d=np.load(verified('c1familyb',f'case{32+case}.npz'));truth=jnp.asarray(d['coefficients']);norm=float(d['initial_norm']);states,sec=timer(lambda:f(W.T@truth[0].ravel()/2,params[case]['nu'],G,H,e));pred=(2*(states@W.T)).reshape(11,255,255)
   old=f(oldW.T@truth[0,:127,:127].ravel()/2,params[case]['nu'],oldG,oldH,olde);oldpred=jnp.pad((2*(old@oldW.T)).reshape(11,127,127),((0,0),(0,128),(0,128)))
   err=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/(2*norm);delta=np.asarray(jnp.linalg.norm((pred-oldpred).reshape(11,-1),axis=1))/(2*norm)
   np.savez_compressed(o/f'resolved_split{"".join(map(str,split))}_dt{dt}_case{case}.npz',coefficients=pred,error=err)
   resolutions.append(dict(case=case,split=split,dt=dt,error=err,feature_change=delta,seconds=sec))
 save(o/'resolution.json',dict(rows=resolutions));print('resolved',split,flush=True)
xy=jnp.asarray(E.coords(512,True));x=jnp.arange(4096,dtype=jnp.float64)*256/4095;coords=jnp.stack(jnp.meshgrid(x,x,indexing='ij')).reshape(2,-1);f=C.flux_trajectory(.00625)
@jax.jit
def restrict(u):return P.analysis(map_coordinates(u,(xy*4095).T,order=1,mode='nearest').reshape(511,511),512)
@jax.jit
def extend(c):return map_coordinates(PJ.synth_mesh(c,256),coords,order=1,mode='nearest').reshape(4096,4096)
@jax.jit
def solve(u,nu,W,e,G,H):
 b=restrict(u)[:127,:127];states=f(W.T@b.ravel()/2,nu,G,H,e);coeff=(2*(states@W.T)).reshape(11,127,127);return coeff,extend(coeff[-1])
def tree(u,nu):return solve(u,nu,*models[tuple(sorted(QT.initial_order(u)[:2]))])
def uniform(u,nu):return solve(u,nu,*models[(0,1,2,3)])
# Separate classical integration control at twice nonlinear evaluation resolution.
def over_solver(N,dt):
 lam=C.eigenvalues(N).reshape(N-1,N-1)
 @jax.jit
 def run(b,nu):
  co=P.etdrk4_coefs(-nu*lam,dt,32)
  def nl(v):return P.nonlinear(jnp.pad(v,((0,N),(0,N))),2*N,2)[0][:N-1,:N-1]
  def step(v):
   nv=nl(v);a=co['E2']*v+co['Q']*nv;na=nl(a);b=co['E2']*v+co['Q']*na;nb=nl(b);c=co['E2']*a+co['Q']*(2*nb-nv);nc=nl(c);return co['E']*v+co['f1']*nv+2*co['f2']*(na+nb)+co['f3']*nc
  def block(v,_):
   v=jax.lax.fori_loop(0,round(.025/dt),lambda i,x:step(x),v);return v,v
  _,s=jax.lax.scan(block,b,None,length=10);return jnp.concatenate([b[None],s])
 return run
queries={'tree768':tree,'uniform1152':uniform};settings={}
for n in (24,32,48,64):
 for dt in (.0125,.00625):
  name=f'spectral{n}_dt{dt}';queries[name]=make(n,dt,512,'bilinear256');settings[name]=(n,dt)
controls={(n,dt):(C.full_trajectory(n,dt/2),over_solver(n,dt)) for n,dt in settings.values()};rows=[];rng=np.random.default_rng(2026111814)
for i,ph in enumerate(params):
 u=jnp.asarray(FB.initial(4095,ph));jax.block_until_ready(u);norm=float(jnp.linalg.norm(u));normphys=float(np.load(verified('c1familyb',f'case{32+i}.npz'))['initial_norm']);times={name:[[],[]] for name in queries};cold={};outputs={};checks={}
 for name,query in queries.items():outputs[name],cold[name]=timer(lambda:query(u,ph['nu']))
 for block in range(2):
  for rep in range(5):
   for name in rng.permutation(list(queries)):
    E.burn(.4);outputs[name],sec=timer(lambda:queries[name](u,ph['nu']));times[name][block].append(sec);np.savez_compressed(o/f'{name}_case{i}_block{block}_rep{rep}.npz',coefficients=outputs[name][0],final_norm=float(jnp.linalg.norm(outputs[name][1])))
 initial=restrict(u)
 for name,(n,dt) in settings.items():
  half,over=controls[n,dt];half=half(initial[:n-1,:n-1],ph['nu']);over=over(initial[:n-1,:n-1],ph['nu']);base=outputs[name][0];checks[name]=dict(time_change=np.asarray(jnp.linalg.norm((half[::2]-base).reshape(6,-1),axis=1))/(2*normphys),integration_change=np.asarray(jnp.linalg.norm((over[::2]-base).reshape(6,-1),axis=1))/(2*normphys));np.savez_compressed(o/f'{name}_controls_case{i}.npz',half=half,over=over)
 def fd(L,dt):
  u=np.load(verified('c1familyb',f'dev{i}_fd{L}_dt{dt}.npz'))['out'].reshape(6,L-1,L-1);return jax.vmap(lambda u:P.analysis(u,L))(jnp.asarray(u[1:]))
 bc=fd(512,.0003125);space=[fd(L,.0003125) for L in (128,256)];timec=[fd(512,h) for h in (.00125,.000625)];ps=np.load(verified('c1familyb',f'check{32+i}.npz'))['coefficients'];errors={name:[] for name in queries};extension={name:[] for name in queries};refchecks=[]
 for t in range(5):
  ref=PJ.synth_mesh(bc[t],4095);sp=[PJ.synth_mesh(c[t],4095) for c in space];tm=[PJ.synth_mesh(c[t],4095) for c in timec];ds=[float(jnp.linalg.norm(sp[0]-sp[1]))/norm,float(jnp.linalg.norm(sp[1]-ref))/norm];dts=[float(jnp.linalg.norm(tm[0]-tm[1]))/norm,float(jnp.linalg.norm(tm[1]-ref))/norm];mid=float(jnp.linalg.norm(PJ.synth_mesh(space[1][t],2047)-PJ.synth_mesh(bc[t],2047)))/np.linalg.norm(FB.initial(2047,ph));cross=float(jnp.linalg.norm(PJ.synth_mesh(jnp.asarray(ps[2*(t+1)]),4095)-ref))/norm;refchecks.append(dict(space_deltas=ds,time_deltas=dts,score_grid_change=abs(mid-ds[-1]),cross_difference=cross))
  for name,(coeff,field) in outputs.items():
   c=coeff[2*(t+1)] if name in ('tree768','uniform1152') else coeff[t+1];dense=extend(c);errors[name].append(float(jnp.linalg.norm(dense-ref))/norm);extension[name].append(float(jnp.linalg.norm(dense-PJ.synth_mesh(c,4095)))/norm)
 if i==0:np.savez_compressed(o/'final4096_case0.npz',reference=ref,tree768=outputs['tree768'][1],uniform1152=outputs['uniform1152'][1],spectral48=outputs['spectral48_dt0.00625'][1])
 rows.append(dict(case=i,times=times,cold_query_seconds=cold,errors=errors,extension_errors=extension,reference_checks=refchecks,classical_checks=checks,output_shape=[4096,4096]));save(o/'result.json',dict(meta=meta(),rows=rows,resolution=resolutions,setup_seconds=setup_seconds));print('cost',i,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,resolution=resolutions,setup_seconds=setup_seconds,complete=True,contract='familyB development only; all seven possible policy caches prepared; resident4096 common input to actual4096 output; all online selection restriction dynamics decode charged; separate finer physical-space and reference ladders'))
