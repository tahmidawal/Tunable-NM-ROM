"""Focused GPU/NumPy replay of failed stationarity instrumentation, no timing claim."""
import argparse,json,pickle,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import engines as e
import accuracy_paths as ap
import audit_accuracy as aa

p=argparse.ArgumentParser();p.add_argument('record',type=Path);a=p.parse_args();begin=time.perf_counter();run=a.record/'archive';out=run/'out';d=json.loads((out/'result.json').read_text());L=256;case=4
params=jax.tree_util.tree_map(jnp.asarray,pickle.loads((run/'in/checkpoint.pkl').read_bytes())['params']);pn=jax.tree_util.tree_map(np.asarray,params)
op=dict(np.load(out/f'operators_frozen_L{L}.npz'));info=next(v for v in d['mesh_setup'] if v['model']=='frozen' and v['intervals']==L)
G=e.sc.SeparableDecoder(params,16,512).feat_at(e.coords(L),chunk=8192)
Zc=jnp.asarray(op['candidate_Z']);R=jnp.asarray(op['cold_R']);Hrot=e.sc.head(params,Zc)@R.T
cold=(jnp.asarray(op['cold_xy']),jnp.asarray(op['cold_w']),jnp.asarray(op['cold_Q']),R,Hrot,jnp.sum(Hrot*Hrot,1))
data=(G,jnp.asarray(op['A']),jnp.asarray(op['lam']),jnp.asarray(op['G5']),jnp.asarray(op['Pq']),jnp.empty((0,)),jnp.empty((0,)),Zc)
phys=np.array(d['physical_cases'][case]);nu=float(phys[-1]);u0=e.initial(L,phys);saved=dict(np.load(out/f'L256_frozen_stationary_case{case}_rep0.npz'))
q=ap.make_rom(params,L,.005,info['trust_radius'],**d['config']['strict']);v=jax.tree_util.tree_map(np.asarray,q(jnp.asarray(u0),nu,data,cold))
post,icpost=jax.tree_util.tree_map(np.asarray,ap.make_diagnostics(params,L,.005)(jnp.asarray(v[7]),jnp.asarray(u0),nu,data,cold))
# NumPy analytic gradients at exactly the reproduced states, plus saved states separately.
def cpu(states):
 h,Jh=aa.head_jac(pn,states[0]);r=op['cold_R']@h-op['cold_Q'].T@(aa.ai.sample_field(u0,op['cold_xy'],L)*op['cold_w']);initial=aa.stationarity(r,op['cold_R']@Jh);prev=op['A']@h;gn=[];rns=[]
 for z in states[1:]:
  h,Jh=aa.head_jac(pn,z);us=np.einsum('msr,r->ms',op['G5'],h);dus=np.einsum('msr,rk->msk',op['G5'],Jh)
  c,xp,xm,yp,ym=us.T;dc,dxp,dxm,dyp,dym=dus.transpose(1,0,2);diff=np.where(c>0,2*c-xm-ym,xp+yp-2*c);ddiff=np.where((c>0)[:,None],2*dc-dxm-dym,dxp+dyp-2*dc)
  adv=L*c*diff;dadv=L*(dc*diff[:,None]+c[:,None]*ddiff);ah=op['A']@h;dah=op['A']@Jh;scale=1+.005*nu*op['lam']
  r=(ah-prev+.005*(op['Pq'].T@adv+nu*op['lam']*ah))/scale;J=(dah+.005*(op['Pq'].T@dadv+nu*op['lam'][:,None]*dah))/scale[:,None]
  gn.append(aa.stationarity(r,J));rns.append(float(np.linalg.norm(r)));prev=ah
 return np.array(gn),initial,np.array(rns)
cn,iccn,rns=cpu(v[7]);sn,icsn,srns=cpu(saved['internal_latents']);gap=abs(post-v[8]);ix=int(np.argmax(gap))
res=dict(scope='focused replay of failed N256 frozen_stationary fresh-development case4; local GPU, not benchmark timing',backend=jax.default_backend(),gpu=jax.devices()[0].device_kind,x64=jax.config.jax_enable_x64,
 elapsed_seconds=time.perf_counter()-begin,source_job=d['job_id'],case=case,intervals=L,original_orphan_artifact=f'L256_frozen_stationary_case{case}_rep0.npz',
 field_relative_delta_from_failed_artifact=float(np.linalg.norm(v[0]-saved['fields'])/np.linalg.norm(saved['fields'])),latent_max_delta_from_failed_artifact=float(np.max(abs(v[7]-saved['internal_latents']))),
 charged_vs_posthoc_weak_max_delta=float(np.max(gap)),charged_vs_posthoc_initial_delta=float(abs(v[9]-icpost)),charged_vs_cpu_weak_max_delta=float(np.max(abs(v[8]-cn))),posthoc_vs_cpu_weak_max_delta=float(np.max(abs(post-cn))),
 charged_vs_cpu_initial_delta=float(abs(v[9]-iccn)),posthoc_vs_cpu_initial_delta=float(abs(icpost-iccn)),charged_residual_vs_cpu_max_delta=float(np.max(abs(v[2]-rns))),
 maximum_gap_step=ix+1,charged_gradient_at_max_gap=float(v[8][ix]),posthoc_gradient_at_max_gap=float(post[ix]),cpu_gradient_at_max_gap=float(cn[ix]),
 reproduced_stationarity=cn.tolist(),charged_stationarity=v[8].tolist(),posthoc_stationarity=post.tolist(),saved_state_cpu_stationarity=sn.tolist(),saved_initial_cpu_stationarity=icsn,
 initial=dict(charged=float(v[9]),posthoc=float(icpost),cpu=iccn),residual_norms=rns.tolist(),saved_residual_norms=srns.tolist(),iterations=v[1].tolist(),reasons=v[3].tolist(),ic_iterations=int(v[5]),ic_reason=int(v[6]))
(a.record/'STATIONARITY-DIAGNOSTIC.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if not isinstance(v,list)}),flush=True)
assert res['field_relative_delta_from_failed_artifact']<1e-8
