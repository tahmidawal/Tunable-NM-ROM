"""Small actual-checkpoint smoke, under one minute on the local GPU."""
from pathlib import Path
import signal,sys,time,pickle
signal.alarm(55)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engines as e
import numpy as np
import jax
import jax.numpy as jnp
root=Path(__file__).resolve().parents[3]
p=root/'experiments/separable-decoder/runs/b2dtensor/n512/out/sep_b2d_tensor_n512_ckpt.pkl'
params,Z,cfg=e.sc.load_pkl(p)
t=time.perf_counter()
data,info=e.build_rom(params,Z,24,64,256,candidate_cap=256,fit_states=4)
q,parts=e.make_rom(params,24,.01,info['trust_radius'],ic_starts=4,return_parts=True)
physical=e.params_draw(7090702,4)[0]
f,it,rn,reasons,z,icit,icreason=jax.tree_util.tree_map(np.asarray,q(jnp.asarray(e.initial(24,physical)),physical[4],data))
zi,ii,ir,scale=parts['initialize'](jnp.asarray(e.initial(24,physical)),data)
Zp,_,_,_=parts['evolve'](zi,physical[4],scale,data)
fp=np.asarray(parts['decode'](Zp,data))
parity=float(np.linalg.norm(fp-f)/np.linalg.norm(f))
assert parity<1e-8
assert f.shape==(6,25,25) and np.isfinite(f).all()
assert np.max(np.abs(f[:,[0,-1],:]))==0 and np.max(np.abs(f[:,:,[0,-1]]))==0
print(dict(pipeline='actual R64/K16 checkpoint',seconds=time.perf_counter()-t,eq_fit=info['eq_relative_fit'],
    shape=f.shape,ic_starts=4,staged_fused_relative=parity,iterations=it.tolist(),stop_reasons=reasons.tolist(),ic_iterations=int(icit),ic_reason=int(icreason)),flush=True)
