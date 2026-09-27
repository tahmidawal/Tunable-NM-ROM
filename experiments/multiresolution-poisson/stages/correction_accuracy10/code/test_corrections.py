"""Bounded GPU proof of nested inclusion, elimination and generic-source solves."""
import argparse,json,os
from pathlib import Path
from correction_core import *
from tuning_core import make_tuning_kernel,tuning_query


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=Path('in/head.pkl'));ap.add_argument('--basis',type=Path,default=Path('in/basis.npz'));ap.add_argument('--config',type=Path,default=Path('code/config-correction-accuracy.json'));ap.add_argument('--mode',choices=['algebra','solve'],required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());p,z,_=sc.load_pkl(a.checkpoint);basis=dict(np.load(a.basis));directions=basis['coefficient_directions'];np.testing.assert_array_equal(z,basis['training_latents'])
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    assert np.linalg.norm(directions.T@basis['R'].T@basis['R']@directions-np.eye(32))<1e-10
    ops=assemble(p,z,64,256,300);source=full_source(64,source_params(7090703,6)[0]);f=ops['project'](jnp.asarray(source),ops['S'],ops['I'],ops['J'],ops['W']);results=[]
    if a.mode=='algebra':
        for count in [0,8,16,32]:
            e=prepare_correction(ops,z,directions,count,cfg);assert e['info']['linear_rank_valid'];C=e['C'];Q=e['Q'];R=e['R'];B=ops['B'];zz=jnp.asarray(z[0]);h=lambda x:sc.head(ops['params'],x)
            y=lambda x:jax.scipy.linalg.solve_triangular(R,Q.T@(f-B@h(x)),lower=False) if count else jnp.zeros((0,))
            full=lambda v:B@(h(v[:len(zz)])+C@v[len(zz):])-f;eliminated=lambda x:full(jnp.concatenate((x,y(x))));projected=lambda x:e['Bp']@h(x)-(f-Q@(Q.T@f) if count else f)
            vector=jnp.concatenate((zz,y(zz)));rf=full(vector);rr=projected(zz);jf=jax.jacfwd(full)(vector);jr=jax.jacfwd(projected)(zz);je=jax.jacfwd(eliminated)(zz)
            residual=relative(rf,rr);jacobian=relative(je,jr);grad=relative((jf.T@rf)[:len(zz)],jr.T@rr);linear_grad=float(jnp.linalg.norm((jf.T@rf)[len(zz):])/(jnp.linalg.norm(jf)*jnp.linalg.norm(rf)+1e-300))
            inclusion=relative(ops['bank']@(h(zz)+C@jnp.zeros((count,))),ops['bank']@h(zz));inclusion_j=relative(jax.jacfwd(lambda x:h(x)+C@jnp.zeros((count,)))(zz),jax.jacfwd(h)(zz))
            assert max(residual,jacobian,grad,linear_grad,inclusion,inclusion_j)<cfg['smoke_parity_tolerance']
            if count==0:assert relative(rr,B@h(zz)-f)<1e-14 and relative(jr,B@jax.jacfwd(h)(zz))<1e-14
            results.append(dict(count=count,residual_relative=residual,eliminated_jacobian_relative=jacobian,gradient_relative=grad,linear_gradient_scaled=linear_grad,old_manifold_value_relative=inclusion,old_manifold_jacobian_relative=inclusion_j))
    else:
        cache=weak_code_cache(ops,z);legacy=tuning_query(source,ops,cache,cfg['online_preset'],make_tuning_kernel(ops,cfg['online_preset'],cfg['linear_backward_error_limit']),cfg)
        for count in [0,32]:
            e=prepare_correction(ops,z,directions,count,cfg);field,row=correction_query(source,ops,e,cfg);assert row['solver_valid'],row
            if count==0:
                assert relative(field,legacy[0])<cfg['smoke_parity_tolerance'] and row['selected_training_code_index']==legacy[1]['selected_training_code_index']
                assert abs(row['residual']-legacy[1]['residual'])/(legacy[1]['initial_residual']+1e-300)<cfg['smoke_parity_tolerance']
                assert abs(row['stationarity']-legacy[1]['stationarity'])<cfg['smoke_parity_tolerance'] and row['reason']==legacy[1]['reason'] and row['attempts']==legacy[1]['attempts'] and legacy[1]['solver_valid']
            results.append(dict(count=count,solver_valid=row['solver_valid'],full_stationarity=row['stationarity'],reduced_stationarity=row['reduced_stationarity'],rank=row['projected_jacobian_rank'],recovery_backward_error=row['linear_recovery_backward_error'],residual_reconstruction=row['residual_reconstruction_scaled'],reason=row['reason'],legacy_field_relative=relative(field,legacy[0]) if count==0 else None,legacy_objective_scaled_difference=abs(row['residual']-legacy[1]['residual'])/(legacy[1]['initial_residual']+1e-300) if count==0 else None,legacy_gradient_absolute_difference=abs(row['stationarity']-legacy[1]['stationarity']) if count==0 else None,legacy_stops_match=(row['reason']==legacy[1]['reason'] and row['attempts']==legacy[1]['attempts']) if count==0 else None))
    print(json.dumps(dict(passed=True,mode=a.mode,backend=jax.default_backend(),x64=jax.config.jax_enable_x64,precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],results=results,scope='smoke controls, not scientific timing measurements'),indent=2),flush=True)


if __name__=='__main__':main()
