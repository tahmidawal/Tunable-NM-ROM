"""Sub-minute GPU checks of field-metric loss, frozen subtrees and phase clocks."""
from staged_training import *
import json

assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
ck=Path('in/model.pkl') if Path('in/model.pkl').exists() else Path(__file__).resolve().parent/'runs/pilot04/checkpoints/original_relative.pkl'
p,z,_=sc.load_pkl(ck);z=z[:72];draws=source_params(0,576)[:72];U,coords,_=field_dataset(draws,32)
G,Q,R,T,C,perp,info=bank_qr(p,U,coords);denom=jnp.sum(U*U,axis=1)
metric=jnp.mean((jnp.sum((sc.head(p,jnp.asarray(z))@R.T-T)**2,axis=1)+perp)/denom)
direct=jnp.mean(jnp.sum((sc.head(p,jnp.asarray(z))@G.T-U)**2,axis=1)/denom)
assert abs(float(metric-direct))<1e-11
P,pod=normalized_pod(U,64);assert pod['orthogonality_error']<1e-10
cfg=dict(source_batch=4,point_batch=32,orthogonality_weight=1e-4,timing_block_updates=1,log_every_updates=1,training_burn_seconds=.001,warmup_fraction=.02,final_learning_rate_fraction=.01,max_matched_steps=20,steps=2,learning_rate=3e-4,rng_seed=71)
pb,free,bankinfo=train_phase(p,np.asarray(C),U,coords,cfg,'bank','smoke_bank')
for key in ['h','h_lin','out_scale']:assert all(np.array_equal(x,y) for x,y in zip(jax.tree_util.tree_leaves(p[key]),jax.tree_util.tree_leaves(pb[key])))
assert not np.array_equal(np.asarray(p['B']),np.asarray(pb['B']))
Gb,Qb,Rb,Tb,Cb,perpb,_=bank_qr(pb,U,coords)
ph,zh,headinfo=train_phase(pb,z,U,coords,cfg,'head','smoke_head',R=Rb,T=Tb,perpendicular=perpb)
for key in ['B','g','out_scale']:assert all(np.array_equal(x,y) for x,y in zip(jax.tree_util.tree_leaves(pb[key]),jax.tree_util.tree_leaves(ph[key])))
pj,zj,jointinfo=train_phase(ph,zh,U,coords,cfg,'joint','smoke_joint')
pc,zc,controlinfo=train_phase(p,z,U,coords,cfg,'joint','smoke_time_control',matched_seconds=.001)
assert controlinfo['budget_reached'] and controlinfo['updates']%cfg['timing_block_updates']==0
for record in [bankinfo,headinfo,jointinfo]:assert record['finite'] and record['updates']==cfg['steps'] and record['budget_reached'] and record['optimizer_seconds']>0
print(json.dumps(dict(passed=True,backend=jax.default_backend(),x64=jax.config.jax_enable_x64,field_metric_absolute_difference=abs(float(metric-direct)),pod_orthogonality_error=pod['orthogonality_error'],bank_rank=info['rank'],frozen_subtree_checks=True,matched_updates=controlinfo['updates']),indent=2))
# A deliberately dependent feature column must use a truncated diagnostic,
# never an unchecked triangular solve or a false full-rank projection label.
w,b=p['g'][-1];bad={**p,'g':[*p['g'][:-1],(w.at[:,-1].set(w[:,-2]),b.at[-1].set(b[-2]))]}
_,qb,rb,tb,cb,pb,badinfo=bank_qr(bad,U,coords)
assert not badinfo['rank_valid'] and badinfo['rank']==63 and badinfo['projection_kind']=='truncated_SVD_diagnostic_failed_rank' and np.isfinite(cb).all()
from staged_oracles import prepare_head_oracle,head_oracle
ops=assemble(p,z,31,64,3);oraclecfg=dict(budget=3,stationarity_tolerance=1e-6,starts=['mean_training_code','nearest_full_field_training_prediction'])
engine=prepare_head_oracle(ops,Q,R,z,oraclecfg,1e-12);same=np.pad(np.asarray(U[0]).reshape(30,30),1)
candidates,record=head_oracle(same,ops,Q,R,engine,oraclecfg,1e-12)
for field,row in candidates:assert abs(np.linalg.norm(field-same)-row['residual'])<1e-10
print(json.dumps(dict(rank_deficiency_detected=True,head_oracle_full_field_residual_verified=True,head_oracle_candidates=len(candidates)),indent=2))
