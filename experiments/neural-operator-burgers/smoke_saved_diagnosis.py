"""Bounded full-query replay using frozen EQ weights, not a new benchmark."""
from pathlib import Path
import json

import numpy as np
import data as d

jax,e=d.gpu_modules()
import jax.numpy as jnp
import accuracy_paths as ap

folder=d.HERE/'runs/refinement02/live-diagnosis'
reference=json.loads((d.HERE/'runs/refinement02/live-reference/index.json').read_text())
report=json.loads((folder/'index.json').read_text());case='burgers-calibration-00002';L=256
checkpoint=d.ROOT/'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
assert d.sha(checkpoint)==report['checkpoint_sha256']
params,Z,_=e.sc.load_pkl(checkpoint);dec=e.sc.SeparableDecoder(params,report['K'],report['R'])
G=dec.feat_at(e.coords(L),chunk=8192);phi,lam,mode_ids=e.modes(L,report['M'])
assert np.array_equal(mode_ids,np.asarray(report['setup']['mode_ids']))
A=jnp.asarray(phi).T@G
pos=np.asarray(report['setup']['eq_indices']);w=np.asarray(report['setup']['eq_weights'])
ij=np.stack(np.unravel_index(pos,(L-1,L-1)),axis=1)+1
offsets=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
G5=dec.feat_at(((ij[:,None,:]+offsets[None,:,:])/L).reshape(-1,2)).reshape(report['m'],5,report['R'])
candidates=jnp.asarray(Z[::max(1,len(Z)//8192)])
assert len(candidates)==report['setup']['cold_candidates']
operators=(G,A,jnp.asarray(lam),G5,jnp.asarray(phi[pos]*w[:,None]),None,None,candidates)
cold,_=e.build_gauss_cold(params,candidates)
row=next(r for r in report['cases'] if r['case_id']==case)
artifact=folder/row['path'];assert d.sha(artifact)==row['sha256']
with np.load(artifact) as arrays:
    supplied=arrays['input'];expected=arrays['online']
nu=next(r['generation_descriptors']['nu'] for r in reference['records'] if r['case_id']==case)
query=ap.make_rom(params,L,report['config']['dt'],report['setup']['trust_radius'],**report['config']['strict'])
result=jax.tree_util.tree_map(np.asarray,query(jnp.asarray(supplied),jnp.asarray(nu),operators,cold))
relative=float(np.linalg.norm(result[0]-expected)/np.linalg.norm(expected))
stationary=bool(result[8].max()<=1e-6 and result[9]<=1e-6)
passed=relative<=1e-8 and stationary
record=dict(passed=passed,scope='one already-opened saved case full-query local smoke; no performance or new accuracy claim',
            case_id=case,intervals=L,relative_field_replay_discrepancy=relative,declared_tolerance=1e-8,
            initial_and_steps_stationary=stationary,backend=jax.default_backend(),jax_version=jax.__version__,
            f64=bool(jax.config.jax_enable_x64),gpu=jax.devices()[0].device_kind,
            matmul_precision=str(jax.config.jax_default_matmul_precision),expected_artifact_sha256=d.sha(artifact),
            checkpoint_sha256=d.sha(checkpoint),source_sha256=d.source_hashes())
d.write_json(d.HERE/'checks/refinement02-query-replay.json',record);print(json.dumps(record),flush=True)
assert passed
