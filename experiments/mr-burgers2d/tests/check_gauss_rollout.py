"""Bounded parity smoke for fixed physical cold fitting inside the full query."""
from pathlib import Path
import signal,sys,time,json
signal.alarm(55)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engines as e
import cold_fit as c
import numpy as np
import jax
import jax.numpy as jnp
begin=time.perf_counter();root=Path(__file__).resolve().parents[3]
params,Z,_=e.sc.load_pkl(root/'experiments/separable-decoder/runs/b2dtensor/n512/out/sep_b2d_tensor_n512_ckpt.pkl')
L=24;data,info=e.build_rom(params,Z,L,64,256,candidate_cap=256,fit_states=4)
cold,ci=e.build_gauss_cold(params,data[7]);xy,w,Q,R,Hrot,Hnorm=cold
phys=e.params_draw(7090702,4)[0];u=jnp.asarray(e.initial(L,phys))
fun,parts=e.make_gauss_rom(params,L,.01,info['trust_radius'],ic_budget=180,return_parts=True)
f,it,rn,reasons,z,icit,icreason=jax.tree_util.tree_map(np.asarray,fun(u,float(phys[4]),data,cold))
zi,ii,ir,scale=parts['initialize'](u,data,cold)
Zp,_,_,_=parts['evolve'](zi,float(phys[4]),scale,data)
fp=np.asarray(parts['decode'](Zp,data));rollout_parity=float(np.linalg.norm(fp-f)/np.linalg.norm(f))
assert rollout_parity<1e-10
# Independent cold03 code path returns both actual initial field and latent fit.
cf=c.make_fit(params,L,180,starts=1)
cfield,cz,*_=jax.tree_util.tree_map(np.asarray,cf(u,data[0],xy,w,Q,R,jnp.zeros((1,1)),data[7],Hrot,Hnorm))
latent_parity=float(np.max(np.abs(cz-z[0])));initial_parity=float(np.max(np.abs(cfield-f[0])))
assert latent_parity<1e-10 and initial_parity<1e-10
interp=float(np.max(np.abs(np.asarray(e.sample_field(u,xy,L)-c.sample_field(u,xy,L)))))
assert interp==0. and np.isfinite(f).all() and f.shape==(6,L+1,L+1)
assert np.max(np.abs(f[:,[0,-1],:]))==np.max(np.abs(f[:,:,[0,-1]]))==0.
result=dict(backend=jax.default_backend(),x64=jax.config.jax_enable_x64,elapsed_seconds=time.perf_counter()-begin,
    rollout_staged_fused_relative=rollout_parity,cold03_latent_max_difference=latent_parity,cold03_initial_field_max_difference=initial_parity,
    charged_interpolation_parity=interp,initial_iterations=int(icit),initial_stop_reason=int(icreason),finite_and_zero_boundaries=True)
path=Path(__file__).with_name('GAUSS-ROLLOUT-CHECK.json');path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
