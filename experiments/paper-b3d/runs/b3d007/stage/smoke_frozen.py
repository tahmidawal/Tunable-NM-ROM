"""Bounded synthetic check of frozen offline loading and final-draw guards."""
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import core as c
import run


def main():
    os.chdir(Path(__file__).parent);root=Path('checks/frozen-smoke');root.mkdir(parents=True,exist_ok=True)
    offline=root/'offline';offline.mkdir(exist_ok=True)
    n=9;k=2;r=4
    p=c.b3.init_separable_3d(jax.random.PRNGKey(13),k,r,n_ff=4,g_hidden=8,g_layers=2,h_hidden=8,h_layers=2,out_scale=.1)
    xyz=c.b3.grid_coords_3d(n)[c.b3.interior_indices_3d(n)]
    G=np.asarray(c.b3.features(p,jnp.asarray(xyz)));Q,R=np.linalg.qr(G,mode='reduced')
    _,_,_,_,lam=c.b3.test_modes_3d(n,16)
    np.savez_compressed(offline/'bases.npz',bank=Q,bank_R=R,pod=Q,pod_singular=np.ones(r),lam=lam)
    np.savez_compressed(offline/'directions.npz',C=np.eye(r))
    checkpoint=root/'checkpoint.pkl';checkpoint.write_bytes(pickle.dumps(dict(params=jax.tree_util.tree_map(np.asarray,p),Z_tr=np.random.default_rng(14).normal(size=(2,k)),cfg=dict(k=k,r=r))))
    cfg=dict(nodes=n,dt=.005,steps=50,seed=0,train_trajectories=2,validation_rows=[512],train_steps=[0,1,2,5,10,20,35,50],
        test_modes=16,q_ladder=[0],pod_ranks=[2],fit_budget=50,step_budget=50,gradient_tolerance=1e-6,repetitions=1,
        timing_seed=51,fom_controls=[[.01,.5]],refinement_cases=[],diagnostic_starts=2,fit_tile=4)
    run.dump(offline/'result.json',dict(config=cfg,checkpoint_sha256=run.sha(checkpoint),directions=dict(training_states=0,synthetic_fixture=True)))
    cfg['frozen_offline']=dict(directory=str(offline),files={name:run.sha(offline/name) for name in ['bases.npz','directions.npz','result.json']})
    path=root/'config.json';run.dump(path,cfg)
    sys.argv=['run.py','--config',str(path),'--checkpoint',str(checkpoint),'--out',str(root/'out')];run.main()
    result=json.loads((root/'out/result.json').read_text());assert result['complete']
    assert len(result['references'])==1 and result['references'][0]['parameter_seed']==0 and result['references'][0]['parameter_row']==512
    for name in ['bases.npz','directions.npz']:assert (root/'out'/name).read_bytes()==(offline/name).read_bytes()
    # A final configuration with an unregistered checkpoint must fail before the
    # parameter-table generator is reached.
    final=dict(cfg,evaluation_kind='final',evaluation_seed=920399)
    freeze=root/'freeze.json';run.dump(freeze,dict(final_parameter_seed=920399,final_cases=1,checkpoint_sha256=[],offline_artifact_hashes=[]))
    final.update(freeze_manifest=str(freeze),freeze_sha256=run.sha(freeze));run.dump(path,final)
    original=c.b3.draw_param_table
    def forbidden(*a,**kw):raise RuntimeError('final data generator reached before freeze validation')
    c.b3.draw_param_table=forbidden
    try:
        try:run.main()
        except AssertionError:guard=True
        else:raise AssertionError('invalid final freeze accepted')
    finally:c.b3.draw_param_table=original
    run.dump('checks/frozen-offline-smoke.json',dict(passed=True,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),
        frozen_artifacts_byte_identical=True,only_evaluation_reference_generated=True,final_guard_rejected_before_parameter_access=guard,
        scope='synthetic small-grid frozen-loading path; real-checkpoint development replay still required before final evaluation'))
    print('FROZEN OFFLINE SMOKE PASS',flush=True)


if __name__=='__main__':main()
