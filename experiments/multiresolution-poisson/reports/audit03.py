"""Kernel artifact/provenance, CPU field metrics and normal-system audit."""
import argparse
from pathlib import Path
import hashlib
import io
import json
import math
import subprocess
import tarfile
import numpy as np
from scipy.fft import dstn

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
run=a.run.resolve();tree=Path(__file__).resolve().parents[3]
d=json.loads((run/'result.json').read_text());sub=json.loads((run/'submission.json').read_text())
clean=json.loads((run/'cleanup.json').read_text());cfg=d['config']
sha=lambda b:hashlib.sha256(b).hexdigest()
archive=run/'verified-cluster.tar.gz'
if not archive.exists():
    manifest=json.loads((run/'ARCHIVE.json').read_text())
    with archive.open('wb') as stream:
        for part in manifest['ordered_parts']:
            payload=(run/part['name']).read_bytes();assert sha(payload)==part['sha256'];stream.write(payload)
assert sha(archive.read_bytes())==clean['archive_sha256']
assert clean['remote_deleted_and_absence_checked'] and d['complete']
assert d['provenance']['commit']==sub['source_commit'] and d['provenance']['job_id']==sub['job_id']
assert d['provenance']['backend']=='gpu' and d['provenance']['x64'] and d['provenance']['matmul_precision']=='highest'
files={};fields={};reference={};oracles={}
with tarfile.open(archive,'r|gz') as tar:
    for item in tar:
        if not item.isfile():continue
        name=item.name.removeprefix('cluster/')
        if name.startswith('code/') or name in ('in/model.pkl','in/ORIGIN.json','out/pilot/result.json'):
            files[name]=tar.extractfile(item).read()
        elif name.startswith('out/pilot/') and name.endswith('.npz'):
            payload=tar.extractfile(item).read()
            arrays=np.load(io.BytesIO(payload))
            if 'field' in arrays:
                field=arrays['field'];fields[sha(np.ascontiguousarray(field).tobytes())]=field
            elif 'observation' in arrays:
                case=int(Path(name).stem.split('case')[1]);reference[case]={key:arrays[key] for key in arrays.files}
            else:oracles[Path(name).stem]={key:arrays[key] for key in arrays.files}
assert files['out/pilot/result.json']==(run/'result.json').read_bytes()
pathmap={'sep_common.py':'experiments/separable-decoder/sep_common.py',
    'ctol_tol.py':'experiments/cost-to-tolerance/ctol_tol.py',
    'ms_parametric.py':'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py'}
for staged,expected in sub['source_hashes'].items():
    content=files[staged];assert sha(content)==expected
    name=Path(staged).name;path=pathmap.get(name,'experiments/multiresolution-poisson/'+name)
    assert content==subprocess.check_output(['git','show',sub['source_commit']+':'+path],cwd=tree)
origin=json.loads(files['in/ORIGIN.json']);ck=files['in/model.pkl']
assert sha(ck)==d['checkpoint_sha256']==origin['checkpoint_sha256']
assert ck==subprocess.check_output(['git','show',sub['source_commit']+':'+origin['checkpoint_path']],cwd=tree)
log=(run/(sub['job_id']+'.out')).read_text()+(run/(sub['job_id']+'.err')).read_text()
assert 'jax_backend=gpu' in log and 'ALL-DONE' in log and (run/'EXIT_CODE').read_text().strip()=='0'
for bad in ['cuInit','captured constant','large constant','OUT_OF_MEMORY','No space left','Traceback']:assert bad not in log,bad
# Independent host regeneration of same-grid source and discrete DST reference.
truth={}
for n in cfg['intervals']:
    x=np.arange(1,n)/n
    lam1=4*n*n*np.sin(np.pi*np.arange(1,n)/(2*n))**2
    for case,(cx,cy,width,amp) in enumerate(d['cohort']['parameters']):
        f=amp*np.exp(-((x[:,None]-cx)**2+(x[None,:]-cy)**2)/(2*width*width))
        truth[(n,case)]=np.pad(dstn(dstn(f,type=1,norm='ortho')/(lam1[:,None]+lam1[None,:]),type=1,norm='ortho'),1)
rel=lambda x,y:float(np.linalg.norm(x-y)/np.linalg.norm(y))
errors=[];keys=set();components=0
for row in d['rows']:
    key=(row['intervals'],row['case'],row['arm'],row['requested_modes'],row['tau'],row['repetition'])
    assert key not in keys;keys.add(key)
    names=['input_seconds','fused_device_seconds','output_seconds'] if row['arm'] in ('rom_fused','rom_gj') else ['input_seconds','projection_init_seconds','solver_seconds','output_seconds']
    assert abs(row['total_seconds']-sum(row[k] for k in names))<1e-10;components+=1
    field=fields[row['field_sha256']];n=row['intervals'];case=row['case'];factor=n//cfg['observation_intervals']
    obs=field[::factor,::factor];fine=reference[case]['observation'];coarser=reference[case]['coarser_observation']
    delta=rel(coarser,fine);error=rel(obs,fine);adjusted=(error+delta)/(1-delta)
    diffs=[abs(error-row['physical_error']),abs(delta-row['reference_delta']),abs(adjusted-row['conservative_physical_error']),abs(rel(field,truth[(n,case)])-row['same_grid_error'])]
    errors+=diffs
    assert max(diffs)<1e-10
    if row['arm']=='rom_gj':
        assert row['fallback_count']>=0 and row['fallback_count']<=row['attempts']
        assert row['max_linear_backward_error']<=cfg['linear_backward_limit']
    if row['arm'].startswith('rom'):
        normalized=row['gradient_norm']/(row['jacobian_frobenius']*row['residual']+1e-300)
        assert abs(normalized-row['stationarity'])<1e-10
        assert row['stationary']==(row['stationarity']<=cfg['stationarity_tolerance'])
        assert row['solver_valid']==(row['finite'] and row['parity_passed'] and (row['tau_reached'] or row['stationary']))
expected=len(cfg['intervals'])*cfg['cohort_count']*cfg['repetitions']*(2+3*len(cfg['taus'])*len(cfg['requested_modes_ladder']))
assert len(keys)==expected
trace_errors=[];normal_count=0;cond_max=0.;backward_max=0.;cpu_difference_max=0.;min_eig=math.inf
for item in d['trajectories']:
    saved=oracles[Path(item['artifact']).stem]
    assert len(saved['A'])==item['attempts']
    assert int(saved['fallback'].sum())==item['fallback_count']
    np.testing.assert_array_equal(saved['final_z'],item['latent'])
    for i,(aa,bb,xx) in enumerate(zip(saved['A'],saved['b'],saved['step'])):
        normal_count+=1
        condition=float(np.linalg.cond(aa));eig=float(np.linalg.eigvalsh(aa).min())
        backward=float(np.linalg.norm(aa@xx-bb)/(np.linalg.norm(aa)*np.linalg.norm(xx)+np.linalg.norm(bb)+1e-300))
        forward=rel(xx,np.linalg.solve(aa,bb))
        assert eig>0 and np.allclose(aa,aa.T,rtol=1e-13,atol=1e-20)
        assert abs(backward-item['backward_errors'][i])<1e-12
        assert abs(backward-saved['backward'][i])<1e-12
        assert abs(condition/item['condition_numbers'][i]-1)<1e-8
        trace_errors.append(abs(backward-item['backward_errors'][i]))
        cond_max=max(cond_max,condition);backward_max=max(backward_max,backward)
        cpu_difference_max=max(cpu_difference_max,forward);min_eig=min(min_eig,eig)
    assert item['replay_latent_relative']<=cfg['agreement_limits']['latent_relative']
    arm='rom_gj' if item['linear_solver']=='gj' else 'rom_modular'
    matching=[r for r in d['rows'] if r['arm']==arm and all(r[k]==item[k] for k in ('intervals','case','requested_modes','tau'))]
    for row in matching:
        assert rel(saved['final_z'],row['latent'])<=cfg['agreement_limits']['latent_relative']
        if arm=='rom_gj':assert item['fallback_count']==row['fallback_count']
for row in d['synthetic_controls']:
    assert row['known_solution_relative_error']<=row['forward_limit']
    assert row['backward_error']<=cfg['linear_backward_limit']
report=dict(passed=True,job_id=sub['job_id'],source_commit=sub['source_commit'],archive_sha256=clean['archive_sha256'],
    row_count=len(keys),distinct_preserved_field_hashes=len(fields),same_invocation_components_checked=components,
    maximum_independent_cpu_metric_difference=max(errors),normal_systems_checked=normal_count,
    normal_system_condition_max=cond_max,normal_system_backward_max=backward_max,
    normal_system_cpu_solve_difference_max=cpu_difference_max,normal_system_minimum_eigenvalue=min_eig,
    maximum_cpu_recorded_backward_difference=max(trace_errors),
    source_checkpoint_and_result_match_archive_and_git=True,all_reference_differences_recomputed=True,
    all_original_fusion_parity_passed=all(g['original_fusion']['passed'] for g in d['parity']),
    all_specialized_numerical_agreement_passed=all(g['passed'] for g in d['parity']),
    specialized_field_relative_max=max(g['field_relative'] for g in d['parity']),
    specialized_latent_relative_max=max(g['latent_relative'] for g in d['parity']),
    specialized_objective_initial_scaled_difference_max=max(g['objective_scaled_difference'] for g in d['parity']),
    original_fusion_counters_all_match=all(g['original_fusion']['counters_match'] for g in d['parity']),
    replay_counters_all_match=all(g['replay_counter_agreement'] for g in d['trajectories']),
    specialized_timed_fallbacks=sum(r.get('fallback_count',0) for r in d['rows']),remote_deleted=True,
    reference_limit='Fine observation artifacts audited; no rigorous continuum bound established')
(run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
