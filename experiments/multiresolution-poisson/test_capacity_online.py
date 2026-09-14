"""Separate sub-minute widened online-kernel parity smoke, no benchmark claims."""
import argparse,json,os
from pathlib import Path
from staged_training import *
from widen_bank import widen
from speed_core import weak_code_cache
from tuning_core import make_tuning_kernel,tuning_query


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=Path('in/model.pkl'));ap.add_argument('--config',type=Path,default=Path('code/config-capacity-accuracy.json'));a=ap.parse_args();cfg=json.loads(a.config.read_text());p,z,_=sc.load_pkl(a.checkpoint)
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    x=jnp.arange(1,32)/32;xx,yy=jnp.meshgrid(x,x,indexing='ij');coords=jnp.stack((xx.ravel(),yy.ravel()),axis=1);new,_=widen(p,128,lambda pp:sc.features(pp,coords));source=full_source(64,source_params(7090703,6)[0]);answers=[]
    for params in [p,new]:
        ops=assemble(params,z,64,256,300);cache=weak_code_cache(ops,z);kernel=make_tuning_kernel(ops,cfg['online_preset'],cfg['linear_backward_error_limit']);field,row=tuning_query(source,ops,cache,cfg['online_preset'],kernel,cfg);assert row['solver_valid'] and row['stationary'];answers.append((field,row))
    error=relative(answers[1][0],answers[0][0]);assert error<1e-10
    print(json.dumps(dict(passed=True,backend=jax.default_backend(),x64=jax.config.jax_enable_x64,precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],full_field_relative_parity=error,initial_code_indices=[r['selected_training_code_index'] for _,r in answers],stationarities=[r['stationarity'] for _,r in answers],stop_reasons=[r['reason'] for _,r in answers],scope='smoke controls only; no local benchmark claim'),indent=2),flush=True)


if __name__=='__main__':main()
