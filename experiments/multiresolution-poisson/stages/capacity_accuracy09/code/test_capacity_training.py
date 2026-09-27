"""Bounded GPU controls for function/Jacobian preservation and widened training."""
import argparse,copy,json,os
from pathlib import Path
from staged_training import *
from widen_bank import widen


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=Path('in/model.pkl'));a=ap.parse_args();p,z,_=sc.load_pkl(a.checkpoint)
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    parameters=source_params(0,576)[:24];U,coords,_=field_dataset(parameters,32);q,info=widen(p,128,lambda pp:sc.features(pp,coords));G,Q,R,T,C,perp,qr=bank_qr(q,U,coords);assert qr['rank_valid']
    Go=sc.features(p,coords);zz=jnp.asarray(z[:24]);before=sc.head(p,zz)@Go.T;after=sc.head(q,zz)@G.T;parity=relative(after,before)
    jo=Go@jax.jacfwd(lambda v:sc.head(p,v))(zz[0]);jn=G@jax.jacfwd(lambda v:sc.head(q,v))(zz[0]);jp=relative(jn,jo);assert max(parity,jp)<1e-12
    full=jnp.sum((after-U)**2,axis=1);compressed=jnp.sum((sc.head(q,zz)@R.T-T)**2,axis=1)+perp;metric=float(jnp.max(jnp.abs(full-compressed))/(jnp.max(full)+1e-300));assert metric<1e-12
    bad=copy.deepcopy(p);w,b=[np.array(v,copy=True) for v in bad['g'][-1]];w[:,1]=w[:,0];b[1]=b[0];bad['g'][-1]=(w,b)
    try:widen(bad,128,lambda pp:sc.features(pp,coords))
    except AssertionError:pass
    else:raise AssertionError('Deficient affine bank must fail before complement construction')
    cfg=dict(source_batch=4,point_batch=32,orthogonality_weight=1e-4,timing_block_updates=1,log_every_updates=1,training_burn_seconds=.001,warmup_fraction=.02,final_learning_rate_fraction=.01,max_matched_steps=100,steps=2,learning_rate=3e-4,rng_seed=123)
    after_p,free,record=train_phase(q,np.asarray(C),U,coords,cfg,'bank','capacity_smoke_bank');assert record['finite'] and record['updates']==2
    for key in ['h','h_lin','out_scale']:
        for a,b in zip(jax.tree_util.tree_leaves(q[key]),jax.tree_util.tree_leaves(after_p[key])):np.testing.assert_array_equal(a,b)
    print(json.dumps(dict(passed=True,backend=jax.default_backend(),x64=jax.config.jax_enable_x64,precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],field_parity=parity,jacobian_parity=jp,field_vs_qr_relative=metric,rank=qr['rank'],condition=qr['condition_number'],deficient_affine_rejected=True,bank_phase_frozen_head_verified=True),indent=2),flush=True)


if __name__=='__main__':main()
