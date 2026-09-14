"""CPU audit of every preserved cold-fit artifact and its campaign lineage.

Full-grid fields are retained at the observation resolution only. Consequently
fine-grid error scalars and QR projection floors are checked for consistency,
not falsely described as independently reconstructed from dense fine fields.
"""
import argparse, collections, hashlib, itertools, json, subprocess
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--pilot',required=True);a=p.parse_args()
run=Path(a.run);pilot=Path(a.pilot);d=json.loads((run/'out/cold_fit.json').read_text());q=json.loads((pilot/'out/pilot.json').read_text())
assert d['commit']==(run/'COMMIT.txt').read_text().strip()
assert str(d['job_id'])==(run/'JOB_ID.txt').read_text().strip()
assert d['complete'] and d['backend']=='gpu' and d['x64'] and d['matmul_precision']=='highest'
assert d['weights_frozen'] and d['checkpoint_sha256']==q['checkpoint_sha256']==q['checkpoint_sha256_after']
assert d['physical_cases']==q['physical_cases'] and d['config']['seed']==q['config']['seed']
root=Path(__file__).resolve().parents[3]
for r in json.loads((run/'PROVENANCE.json').read_text()):
    content=subprocess.check_output(['git','show',r['commit']+':'+r['source']],cwd=root)
    assert hashlib.sha256(content).hexdigest()==r['sha256'],r['source']
for line in (run/'COLLECT.sha256').read_text().splitlines():
    h,file=line.split(maxsplit=1);assert hashlib.sha256((run/file).read_bytes()).hexdigest()==h,file
stderr='\n'.join(x.read_text() for x in (run/'logs').glob('*.err'))
assert not stderr.strip(),stderr
stdout='\n'.join(x.read_text() for x in (run/'logs').glob('*.out'))
assert 'jax_backend=gpu' in stdout and 'ALL-DONE' in stdout
meshes=list(map(int,d['config']['meshes'].split(',')));obs=min(meshes)
expected=set(itertools.product(meshes,['edge_gram','edge_qr','fixed_midpoint','fixed_gauss','full_qr'],[60,180],[1,4],range(d['config']['cases'])))
actual=[(r['intervals'],r['rule'],r['budget'],r['starts'],r['case']) for r in d['rows']]
assert len(actual)==len(expected) and set(actual)==expected
rng=np.random.default_rng(d['config']['seed']);n=d['config']['cases']
physical=np.stack([rng.uniform(.15,.85,n),rng.uniform(.15,.85,n),rng.uniform(.05,.2,n),rng.uniform(.5,2,n),np.exp(rng.uniform(np.log(.01),np.log(.1),n))],axis=1)
assert np.allclose(physical,d['physical_cases'],rtol=2e-15,atol=0)
x=np.arange(obs+1)/obs;xx,yy=np.meshgrid(x,x,indexing='ij');truth=[]
for cx,cy,w,amp,nu in physical:
    u=amp*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*w*w));u[[0,-1],:]=0.;u[:,[0,-1]]=0.;truth.append(u)
max_error_delta=max_truth_delta=max_pilot_delta=0.;floor_groups=collections.defaultdict(list)
for r in d['rows']:
    z=np.load(run/'out'/r['artifact']);f,t=z['field'],z['truth']
    assert f.shape==t.shape==(obs+1,obs+1) and z['latent'].shape==(d['mesh_setup'][0]['latent_K'],)
    assert np.isfinite(f).all() and np.isfinite(z['latent']).all()
    assert np.max(np.abs(f[[0,-1],:]))==np.max(np.abs(f[:,[0,-1]]))==0.
    td=float(np.max(np.abs(t-truth[r['case']])));max_truth_delta=max(max_truth_delta,td);assert td<1e-14
    error=float(np.linalg.norm(f-t)/np.linalg.norm(t));delta=abs(error-r['relative_common_grid_error'])
    max_error_delta=max(max_error_delta,delta);assert delta<1e-13
    if r['intervals']==obs:assert abs(error-r['relative_full_grid_error'])<1e-13
    assert 0<=r['unrestricted_bank_floor']<=r['relative_full_grid_error']+1e-10
    floor_groups[r['intervals'],r['case']].append(r['unrestricted_bank_floor'])
    assert 0<=r['iterations']<=r['budget'] and r['stop_reason'] in range(4) and r['seconds']>0
    if r['stop_reason']==0:assert r['iterations']==r['budget']
    if r['rule']=='edge_gram' and r['budget']==60:
        matches=[s for s in q['invocations'] if s['method']=='rom' and s['solver_intervals']==r['intervals'] and s['case']==r['case'] and s['ic_starts']==r['starts'] and s['rep']==0 and s['dt']==.005 and s['stall']==.001]
        assert len(matches)==1
        pf=np.load(pilot/'out'/matches[0]['observation_artifact'])['fields'][0]
        pd=float(np.max(np.abs(f-pf)));max_pilot_delta=max(max_pilot_delta,pd);assert pd<1e-8
assert all(max(v)==min(v) for v in floor_groups.values())
assert all(s['qr_relative_reconstruction']<1e-11 for s in d['mesh_setup'])
result=dict(source_sha256=hashlib.sha256((run/'out/cold_fit.json').read_bytes()).hexdigest(),output_checksums_verified=True,closed_logs_clean=True,
    source_hashes_against_commits_verified=True,checkpoint_and_physical_cases_match_pilot=True,declared_fits_verified=len(actual),
    common_grid_error_recompute_max_difference=max_error_delta,analytic_truth_max_difference=max_truth_delta,
    original_pilot_cold_field_max_difference=max_pilot_delta,stop_reason_counts=dict(collections.Counter(r['stop_reason'] for r in d['rows'])),
    full_grid_fields_independently_recomputed_only_at_intervals=obs,
    limitation='Fine-grid full-field errors and bank floors are same-invocation scalar diagnostics; fine fields were restricted before archival, so their full norms are not independently reconstructed here. Every retained common-grid field is independently checked. A configured stall is not stationarity; no corrected-initializer rollout was run.')
(run/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
