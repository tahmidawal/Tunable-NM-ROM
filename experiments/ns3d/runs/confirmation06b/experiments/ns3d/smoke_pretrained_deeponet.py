"""Tiny GPU smoke and independent vector-loss/native-architecture checks."""
import json,pickle,subprocess,types
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from operators import extra_models3d as E,models3d as M,pretrained_deeponet as P,training as T
from operator_adapter import checkpoint
from pilot import write


def main():
    out=Path('experiments/ns3d/checks/deeponet_pretraining_v2');out.mkdir(parents=True,exist_ok=True)
    spec=dict(kind='deeponet3d',width=2,levels=1,pool_bins=2,rank=4,trunk_width=8,periodic=True,coordinate_channels=[-3,-2,-1],output_residual_initial=True)
    rng=np.random.default_rng(202609270);axis=np.arange(4)/4;xyz=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    x=np.concatenate((rng.normal(size=(3,4,4,4,4)),np.broadcast_to(xyz,(3,4,4,4,3))),axis=-1)
    y=rng.normal(size=(3,4,4,4,15));dn=rng.uniform(50,100,size=(3,5))
    cfg=dict(seed=202609270,components_per_output=3,steps=2,wall_seconds=15,batch_size=1,learning_rate=.001,validation_every=1,curve_every=1)
    pcfg=dict(teacher_seed=202609270,teacher_oversampling=4,teacher_power_iterations=2,trunk_steps=2,trunk_learning_rate=.001,trunk_batch_points=32,trunk_wall_seconds=10,
        branch_steps=2,branch_learning_rate=.001,branch_batch_size=1,branch_wall_seconds=10,checkpoint_every=1)
    old=types.ModuleType('operators.old_extra');old.__package__='operators'
    exec(subprocess.check_output(['git','show','9d8eb6a8bbf3585cf5482ddfb0df32bbf4cf9e39:experiments/ns3d/operators/extra_models3d.py']),old.__dict__)
    p=M.init_model(jax.random.PRNGKey(cfg['seed']),spec,7,15);before=np.asarray(old.apply_extra(p,jnp.asarray(x),spec));after=np.asarray(E.apply_extra(p,jnp.asarray(x),spec))
    assert np.array_equal(before,after)
    params,info=P.pretrain(x,y,spec,cfg,pcfg,out/'pretraining',lambda path,obj:write(obj,path),checkpoint,dn)
    model,train=T.train(x,y,x,y,spec,cfg,out/'joint',lambda path,obj:write(obj,path),checkpoint,dn,dn,params)
    with np.load(out/'pretraining/training_teacher.npz') as f:basis=f['spatial_basis'];saved=f['projection_errors'];given=f['denominators']
    snapshots=np.moveaxis(y,-1,1).reshape(45,64);residual=snapshots-(snapshots@basis)@basis.T
    scalar=np.sum(residual**2,axis=1)/np.repeat(dn,3,axis=1).ravel()
    vector=scalar.reshape(3,5,3).sum(axis=-1)
    assert np.max(abs(np.sqrt(scalar)-saved))<1e-12 and np.array_equal(given,np.repeat(dn,3,axis=1).ravel())
    assert abs(np.sqrt(vector).max()-info['teacher_projection_error']['worst'])<1e-12
    with np.load(out/'pretraining/branch_teacher.npz') as f:coeff=f['coefficients'];gram=f['gram']
    teacher=pickle.load((out/'pretraining/trunk_selected.pkl').open('rb'))['params']
    trunk=np.asarray(E.deeponet_trunk(teacher,jnp.asarray(xyz.reshape(64,3)),spec))/2
    delta=rng.normal(size=(3,15,4));direct=np.sum((delta@trunk.T)**2,axis=-1)/given.reshape(3,15)
    metric=np.sum((delta@np.linalg.cholesky(gram))**2,axis=-1)/given.reshape(3,15)
    assert np.max(abs(direct.reshape(3,5,3).sum(axis=-1)-metric.reshape(3,5,3).sum(axis=-1)))<1e-12
    record=dict(passed=True,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),native_output_bitwise=True,
        teacher_denominator_parity=True,vector_metric_max_error=float(np.max(abs(direct-metric))),joint_steps=train['steps_completed'],no_final_data_used=True)
    from audit_deeponet_pretraining import audit
    independent=audit(out/'pretraining',zip(y,dn),64);write(independent,out/'independent_teacher_audit.json');assert independent['passed']
    write(record,out/'audit.json');print(json.dumps(record))


if __name__=='__main__':main()
