"""Tiny frozen-asset panel and exact dense/tensor Galerkin smoke; no final draw."""
import json,pickle,shutil
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import ns3d_fom as F
import ns3d_model as D
import ns3d_rom as R
from operators import models3d as M
from operator_adapter import checkpoint
from pilot import write
from extra03 import file_hash
from frozen_panel import prepare_assets,evaluate


def main():
    root=Path('experiments/ns3d/checks/frozen_panel_v2');root.mkdir(parents=True,exist_ok=True);reuse=root/'reuse';reuse.mkdir(exist_ok=True)
    n=6;k=2;r=8;rng=np.random.default_rng(311);p=D.init(jax.random.PRNGKey(311),k,r,width=8,n_ff=6);G=np.asarray(D.bank(p,D.coords(n),F.geometry(n),n));Q,Rb=np.linalg.qr(G)
    codes=rng.normal(size=(12,k));coeff=np.asarray(D.head(p,jnp.asarray(codes)))+.01*rng.normal(size=(12,r))
    D.checkpoint(reuse/'frozen_bank.pkl',p,codes,dict(k=k,r=r),{},extra=dict(bank=G,qr_Q=Q,qr_R=Rb,train_coefficients=coeff))
    D.checkpoint(reuse/'head.pkl',p,codes,dict(k=k,r=r),dict(training=dict(steps=1,smoke_only=True)))
    np.savez_compressed(reuse/'pod.npz',basis=Q)
    checkpoint(reuse/'operator_statistics.pkl',dict(input_mean=np.zeros((1,4,1,1,1)),input_std=np.ones((1,4,1,1,1)),output_scale=1.))
    specs=[dict(kind='fno3d',width=2,modes=[2,2,2],depth=1,padding=0,output_residual_initial=True),
        dict(kind='unet3d',width=2,levels=1,periodic=True,output_residual_initial=True),
        dict(kind='deeponet3d',width=2,levels=1,pool_bins=2,rank=4,trunk_width=8,periodic=True,coordinate_channels=[-3,-2,-1],output_residual_initial=True),
        dict(kind='transolver3d',width=4,depth=1,heads=1,slices=2,patch=2,reference_grid=2,mlp_ratio=2,periodic=True,coordinate_channels=[-3,-2,-1],coordinate_bounds=[0,1],output_residual_initial=True)]
    for index,spec in enumerate(specs):
        params=M.init_model(jax.random.PRNGKey(index+50),spec,7 if index>=2 else 4,15)
        checkpoint(reuse/spec['kind']/'best.pkl',dict(params=params,spec=spec))
    write(dict(files=[dict(path=str(p.relative_to(reuse)),sha256=file_hash(p)) for p in sorted(reuse.rglob('*')) if p.is_file()],final_cohort_opened=False),reuse/'REUSE.json')
    cfg=dict(n=n,k=k,r=r,selected_head='smoke_head',q_values=[0,2],test_modes=32,rom_dt=.0008,dt=.0008,horizon=.004,rom_budget=4,cold_starts=2,gtol=1e-7,
        operators=specs,operator_projection_variants=[True],fom_dts=[.0008],reference_dt=.0004,fine_reference_n=8,fine_reference_dt=.0004,
        refinement_n=12,refinement_dt=.0002,refinement_budget=1.,dev_seed=202609311,timed_cases=1,repetitions=1,evaluation_cohort='development',
        final_seed_unopened=202609203,physical_accuracy_target=1.)
    out=root/'output';out.mkdir(exist_ok=True);prepare_assets(cfg,reuse,out/'assets')
    oldL,T=R.build_galerkin(Q[:,:4],n);newL=R.dense_galerkin_linear(Q[:,:4],n);parameter=F.parameters(cfg['dev_seed'],1)[0];u0=jnp.asarray(F.initial(n,parameter))
    old=np.asarray(R.make_galerkin_run(.0008,5,1)(u0,parameter[-1],jnp.asarray(Q[:,:4]),jnp.asarray(oldL),jnp.asarray(T)))
    new=np.asarray(R.make_dense_galerkin_run(.0008,5,1,n)(u0,parameter[-1],jnp.asarray(Q[:,:4]),jnp.asarray(newL),F.geometry(n)))
    difference=float(np.linalg.norm(old-new)/np.linalg.norm(old));assert difference<1e-10
    write(dict(passed=True,relative_full_trajectory=difference,linear_max=float(np.max(abs(oldL-newL))),final_cohort_opened=False),root/'galerkin_parity.json')
    evaluate(cfg,out/'assets',out,cfg['dev_seed'],1,dict(checkpoint_replay=dict(passed=True,smoke_only=True)),
        smoke_methods=['nmrom_smoke_head_q0','pod_weak_2','free_bank_galerkin','fno3d_increment_projected'])
    write(dict(passed=True,final_cohort_opened=False,backend=jax.default_backend(),complete_panel=True),root/'smoke.json')


if __name__=='__main__':main()
