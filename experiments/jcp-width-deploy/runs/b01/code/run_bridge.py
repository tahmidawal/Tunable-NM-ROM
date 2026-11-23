"""Registered matrix-free nonnegative cubature fidelity and site-economy audit."""
from pathlib import Path
import os,json,pickle,time,subprocess,argparse,hashlib
os.environ.setdefault('JAX_ENABLE_X64','true')
import numpy as np
import scipy.optimize
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import core as C
from core import e
import sep_common as sc
from run import save_json,npz,sha
H=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
 assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
 n,R,M,dt=256,512,8192,.005
 spec=json.loads((H/'configs/bridge.json').read_text());assert spec['M']==M and spec['R']==R
 save_json(out/'environment.json',dict(job_id=os.environ['SLURM_JOB_ID'],commit=os.environ['SOURCE_COMMIT'],backend=jax.default_backend(),x64=True,matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip(),config=spec,source_sha256={str(x.relative_to(H)):sha(x) for x in [H/'run_bridge.py',H/'core.py',H/'inputs/bridge-development.npz',H/'inputs/frozen.pkl',H/'inputs/rotation_R512.npz']}))
 ckpt=pickle.load((H/'inputs/frozen.pkl').open('rb'));params=jax.tree_util.tree_map(jnp.asarray,ckpt['params']);T=jnp.asarray(np.load(H/'inputs/rotation_R512.npz')['T'][:,:R]);del ckpt
 f=jax.jit(lambda p,x,t:sc.features(p,x)@t);xy=e.coords(n)
 G=jax.block_until_ready(jnp.concatenate([f(params,jnp.asarray(xy[i:i+2048]),T) for i in range(0,len(xy),2048)]))
 modes,lam=C.mode_order(n,M);mo=jnp.asarray(modes);la=jnp.asarray(lam)
 A=jax.block_until_ready(jax.jit(lambda g,mode:C.project(g,mode,n))(G,mo))
 gram,ev,transform=C.physical_tangent(G,n);assert transform.shape[1]==R
 data=np.load(H/'inputs/bridge-development.npz');coeff=jnp.asarray(data['coeff'])
 physical=np.concatenate((e.params_draw(7090702,4),e.params_draw(911702,2)))
 assert np.allclose(physical,data['physical'],rtol=0,atol=1e-15)
 fields=jax.block_until_ready(jax.vmap(lambda cc:cc@G.T)(coeff))
 fit_pairs=[(c,t) for c in [0,1,2] for t in [0,10,20,30,40,50]]
 X=jnp.stack([fields[c,t] for c,t in fit_pairs]).T
 X=X/(jnp.linalg.norm(X,axis=0)/n)[None,:]
 Y=jax.block_until_ready(C.project(X,mo,n));den=jnp.sum(Y*Y)
 @jax.jit
 def objective(w,ix,x,y,modes,denominator):
  rho=jnp.zeros((n-1)**2,dtype=jnp.float64).at[ix].set(w)
  delta=C.project(x*rho[:,None],modes,n)-y
  return .5*jnp.sum(delta*delta)/denominator
 valuegrad=jax.jit(jax.value_and_grad(objective))
 @jax.jit
 def moments(x,rho,modes):return C.project(x*rho[:,None],modes,n)
 @jax.jit
 def evaluate(c,old,nu,dc,g,a,modes,lam,rho):
  u=g@c;du=g@dc
  adv,dadv=jax.jvp(lambda v:e.spatial(v,n)[0],(u,),(du,))
  scale=1/(1+dt*nu*lam)
  linear=a@(c-old)+dt*nu*lam*(a@c)
  jl=a@dc+dt*nu*lam*(a@dc)
  rd=scale*(linear+dt*C.project(adv,modes,n));rs=scale*(linear+dt*C.project(adv*rho,modes,n))
  jd=scale*(jl+dt*C.project(dadv,modes,n));js=scale*(jl+dt*C.project(dadv*rho,modes,n))
  return rd,rs,jd,js
 @jax.jit
 def cold(g,ix,w,u):
  B=g[ix]*jnp.sqrt(w)[:,None]/n;b=u[ix]*jnp.sqrt(w)/n
  q,r=jnp.linalg.qr(B,mode='reduced');c=jax.scipy.linalg.solve_triangular(r,q.T@b,lower=False)
  return c,jnp.linalg.svd(r,compute_uv=False)
 ii,jj=np.meshgrid(np.arange(1,n),np.arange(1,n),indexing='ij')
 supports={'checkerboard':((ii+jj)%2==0),'three_quarters':~((ii%2==0)&(jj%2==0))}
 report=dict(scope='development-only discrete upwind quadrature qualification; not a runtime experiment',fit_pairs=fit_pairs,heldout_cases=[3,4,5],rules={},admitted_rules=[],fresh_data_used=False)
 for name,mask in supports.items():
  start=time.monotonic();ix=np.flatnonzero(mask.ravel());jx=jnp.asarray(ix);w0=np.full(len(ix),len(mask.ravel())/len(ix))
  trace=[]
  def fun(w):
   loss,grad=jax.block_until_ready(valuegrad(jnp.asarray(w),jx,X,Y,mo,den));trace.append([float(loss),float(np.linalg.norm(np.asarray(grad),np.inf))]);return float(loss),np.asarray(grad)
  result=scipy.optimize.minimize(fun,w0,jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(ix),options={'maxiter':500,'maxls':30,'ftol':1e-14,'gtol':1e-10,'maxcor':10})
  w=np.asarray(result.x);loss,grad=fun(w);pg=np.where((w<=0)&(grad>0),0,grad);pgmax=float(np.linalg.norm(pg,np.inf))
  rho=np.zeros((n-1)**2);rho[ix]=w;jrho=jnp.asarray(rho)
  closure=mask.copy();closure[:-1]|=mask[1:];closure[1:]|=mask[:-1];closure[:,:-1]|=mask[:,1:];closure[:,1:]|=mask[:,:-1]
  sites=int(closure.sum());sitefraction=sites/((n-1)**2)
  fiterr=np.linalg.norm(np.asarray(moments(X,jrho,mo)-Y),axis=0)/np.linalg.norm(np.asarray(Y),axis=0)
  moment_records=[]
  for case in range(6):
   xx=fields[case].T;target=jax.block_until_ready(C.project(xx,mo,n));actual=jax.block_until_ready(moments(xx,jrho,mo));delta=np.linalg.norm(np.asarray(actual-target),axis=0);denom=np.linalg.norm(np.asarray(target),axis=0)
   for ti in range(51):
    moment_records.append(dict(case=case,time_index=ti,fit_snapshot=(case,ti) in fit_pairs,absolute_error=float(delta[ti]),reference_moment_norm=float(denom[ti]),relative_error=float(delta[ti]/max(denom[ti],1e-300))))
  held=[v for v in moment_records if not v['fit_snapshot']]
  probes=[];cold_records=[]
  for case in [3,4,5]:
   u0=jnp.asarray(e.initial(n,physical[case])[1:-1,1:-1].ravel());norm0=float(jnp.linalg.norm(u0)/n)
   cs,sv=cold(G,jx,jnp.asarray(w),u0);cs,sv=jax.block_until_ready((cs,sv))
   initialgap=float(jnp.linalg.norm(G@(cs-coeff[case,0]))/(n*norm0));rank=int(np.sum(np.asarray(sv)**2>1e-12*float(sv[0])**2))
   cold_records.append(dict(case=case,rank=rank,squared_singular_ratio=float((sv[-1]/sv[0])**2),field_difference_initial_norm=initialgap,coefficient_difference_norm=float(jnp.linalg.norm(cs-coeff[case,0]))))
   for ti in [1,10,25,50]:
    for seed in [1701,1702,1703]:
     z=np.random.default_rng(seed).normal(size=R);z/=np.linalg.norm(z);dc=jnp.asarray(transform@z)
     rd,rs,jd,js=jax.block_until_ready(evaluate(coeff[case,ti],coeff[case,ti-1],float(physical[case,-1]),dc,G,A,mo,la,jrho))
     da=float(jnp.linalg.norm(js-jd));ja=float(jnp.linalg.norm(jd));ra=float(jnp.linalg.norm(rs-rd))
     probes.append(dict(case=case,time_index=ti,seed=seed,residual_absolute=ra,residual_initial_norm=ra/norm0,dense_residual_norm=float(jnp.linalg.norm(rd)),action_absolute=da,dense_action_norm=ja,action_relative=da/max(ja,1e-300),direction_mass_norm=float(jnp.linalg.norm(G@dc)/n)))
  gates=dict(nonnegative_finite=bool(np.isfinite(w).all() and np.all(w>=0)),nnls_projected_gradient=pgmax<=1e-8,heldout_moments=max(v['relative_error'] for v in held)<=1e-2,residual_fidelity=max(v['residual_initial_norm'] for v in probes)<=1e-5,action_fidelity=max(v['action_relative'] for v in probes)<=1e-3,cold_start=all(v['rank']==512 and v['field_difference_initial_norm']<=1e-4 for v in cold_records),site_economy=sitefraction<=.8)
  fidelity=all(v for k,v in gates.items() if k!='site_economy');admit=all(gates.values())
  rec=dict(fit=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),function_calls=int(result.nfev),objective=loss,projected_gradient_inf=pgmax,training_moment_max=float(fiterr.max()),trace=trace),sample_count=int(len(ix)),actual_positive_weights=int(np.sum(w>0)),dense_interior_sites=(n-1)**2,unique_stencil_interior_sites=sites,stencil_fraction=sitefraction,neighbor_policy='both possible upwind directions; boundary values prescribed zero',gates=gates,fidelity_passed=fidelity,admitted=admit,heldout_moment_max=max(v['relative_error'] for v in held),probe_residual_max=max(v['residual_initial_norm'] for v in probes),probe_action_max=max(v['action_relative'] for v in probes),cold_start_max=max(v['field_difference_initial_norm'] for v in cold_records),moment_records=moment_records,probes=probes,cold_start=cold_records,seconds=time.monotonic()-start)
  report['rules'][name]=rec
  if admit:report['admitted_rules'].append(name)
  npz(out/f'{name}-rule.npz',indices=ix,weights=w/n**2,dimensionless_weights=w,closure=np.flatnonzero(closure.ravel()),fit_error=fiterr)
  save_json(out/'summary.json',report)
  print(name,json.dumps({k:rec[k] for k in ['sample_count','unique_stencil_interior_sites','heldout_moment_max','probe_residual_max','probe_action_max','cold_start_max','gates','seconds']}),flush=True)
 report['further_rollout_gate_open']=bool(report['admitted_rules']);report['completed']=True;save_json(out/'summary.json',report)
 print('completed=True',flush=True)
if __name__=='__main__':main()
