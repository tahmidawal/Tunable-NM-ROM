"""Under-minute alignment/finite-solve smoke for the declared timestep arms."""
from pathlib import Path
import signal,sys,time,json
signal.alarm(55)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engines as e
import numpy as np
import jax
import jax.numpy as jnp
begin=time.perf_counter();root=Path(__file__).resolve().parents[3]
params,Z,_=e.sc.load_pkl(root/'experiments/separable-decoder/runs/b2dtensor/n512/out/sep_b2d_tensor_n512_ckpt.pkl')
L=24;data,info=e.build_rom(params,Z,L,64,256,candidate_cap=256,fit_states=4)
cold,_=e.build_gauss_cold(params,data[7]);phys=e.params_draw(7090702,4)[2];u=jnp.asarray(e.initial(L,phys))
rows=[];first=None
for dt in [.005,.01,.025,.05]:
    q=e.make_gauss_rom(params,L,dt,info['trust_radius'],stall=.01,ic_budget=180)
    f,it,rn,reasons,z,icit,icreason=jax.tree_util.tree_map(np.asarray,q(u,float(phys[4]),data,cold))
    assert f.shape==(6,L+1,L+1) and len(it)==round(.25/dt)
    assert np.isfinite(f).all() and np.isfinite(rn).all()
    assert np.max(np.abs(f[:,[0,-1],:]))==np.max(np.abs(f[:,:,[0,-1]]))==0.
    if first is None:first=f[0].copy()
    diff=float(np.max(np.abs(f[0]-first)));assert diff<1e-12
    assert np.linalg.norm(f[-1]-f[0])>1e-6
    rows.append(dict(dt=dt,time_steps=len(it),outputs=len(f),initial_max_difference=diff,
        stop_reasons=reasons.tolist(),initial_reason=int(icreason),initial_attempts=int(icit)))
q,_=e.make_fom(L,.05)
f,it,rn=jax.tree_util.tree_map(np.asarray,q(u,float(phys[4]),1e-11,1e-9))
assert f.shape==(6,L+1,L+1) and np.isfinite(f).all() and np.max(rn)<2e-11
result=dict(backend=jax.default_backend(),x64=jax.config.jax_enable_x64,
    elapsed_seconds=time.perf_counter()-begin,rom=rows,fom_largest_step_max_residual=float(np.max(rn)),
    fom_largest_step_iterations=it.tolist())
Path(__file__).with_name('TIMESTEP-CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
