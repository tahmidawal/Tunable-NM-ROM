"""Synthetic GPU-only dynamic-argument equivalence; no TRAIN/model inputs or timing claim."""
import json,hashlib
from pathlib import Path
import numpy as np
from accuracy_v2.device_blocks import prepare,theta_once

def check(directory):
    out=Path(directory);out.mkdir(exist_ok=False)
    import jax,jax.numpy as jnp
    assert jax.default_backend()=='gpu' and jax.config.x64_enabled and str(jax.config.jax_default_matmul_precision)=='highest'
    rng=np.random.default_rng(482103);n=19;block=7
    arrays={'x':rng.normal(size=(n,2)),'w':rng.uniform(.1,1,size=n),'target':rng.normal(size=(n,2)),'target_gradient':rng.normal(size=(n,2,2))}
    def field(theta,x):
        return jnp.tanh(x@theta[:8].reshape(2,4)+theta[8:12])@theta[12:20].reshape(4,2)+theta[20:22]
    def objective(theta,x,w,target,target_gradient):
        value=jax.vmap(field,in_axes=(None,0))(theta,x)
        gradient=jax.vmap(jax.jacfwd(field,argnums=1),in_axes=(None,0))(theta,x)
        return jnp.sum(w[:,None]*(value-target)**2)+.3*jnp.sum(w[:,None,None]*(gradient-target_gradient)**2)
    calc=jax.jit(jax.value_and_grad(objective));cache=prepare(arrays,block,jax,2**20)
    assert cache.ranges==((0,7),(7,14),(14,19)) and cache.names==tuple(arrays)
    rows=[]
    for case,scale in enumerate((.1,.35,-.2)):
        theta=np.asarray(scale*rng.normal(size=22),dtype=np.float64);oldf=0.;oldg=np.zeros_like(theta)
        for i,j in cache.ranges:
            f,g=calc(jnp.asarray(theta),*(a[i:j] for a in arrays.values()));oldf+=float(f);oldg+=np.asarray(g)
        newf=0.;newg=np.zeros_like(theta);device_theta=theta_once(theta,jax)
        for args in cache.arguments:
            f,g=calc(device_theta,*args);newf+=float(f);newg+=np.asarray(g)
        raw=out/f'case-{case}.npz';np.savez(raw,**arrays,theta=theta,original_objective=np.asarray(oldf),cached_objective=np.asarray(newf),original_gradient=oldg,cached_gradient=newg)
        gap=float(np.max(np.abs(newg-oldg)));assert abs(newf-oldf)<=1e-12*max(1,abs(oldf)) and gap<=1e-12*max(1,float(np.max(np.abs(oldg))))
        rows.append({'raw_file':raw.name,'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'theta_sha256':hashlib.sha256(theta.tobytes()).hexdigest(),'objective':oldf,'objective_gap':abs(newf-oldf),'flattened_parameter_gradient_max_gap':gap})
    try:prepare(arrays,block,jax,1)
    except MemoryError:pass
    else:raise AssertionError('preallocation teacher bound did not reject')
    return {'passed':True,'schema':'ns-obstacle-synthetic-device-argument-equivalence-v1','source_sha256':hashlib.sha256(Path(__file__).with_name('device_blocks.py').read_bytes()).hexdigest(),'fixture_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'backend':jax.default_backend(),'x64':jax.config.x64_enabled,'precision':str(jax.config.jax_default_matmul_precision),'devices':[d.device_kind for d in jax.devices()],'point_count':n,'block_ranges':cache.ranges,'logical_teacher_bytes':cache.logical_bytes,'byte_guard_scope':'logical teacher payload only, not compiler or device peak memory','theta_cases':rows,'actual_training_or_model_inputs':False,'production_integrated':False,'speedup_claim':False,'fixture_scope':'three synthetic theta values, exact original reduction/block ordering and final short block; no universal equivalence claim'}
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=check(a.output);(a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
