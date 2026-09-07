"""CPU archive audit of factorial source draws, endpoints and all field metrics."""
import argparse
from pathlib import Path
import hashlib
import io
import json
import math
import pickle
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
        if name.startswith('code/') or name.endswith('.pkl') or name in ('in/ORIGIN.json','out/pilot/result.json'):
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
def sample(seed,count):
    rng=np.random.default_rng(seed)
    return np.column_stack((rng.uniform(.15,.85,count),rng.uniform(.15,.85,count),
        np.exp(rng.uniform(np.log(.02),np.log(.1),count)),rng.uniform(.5,2.,count)))
original=sample(cfg['original_draw_seed'],cfg['original_draw_count'])
np.testing.assert_array_equal(original,d['training']['original_full_draw_parameters'])
union=np.concatenate((original[:cfg['original_training_prefix_count']],sample(cfg['additional_training_seed'],cfg['additional_training_count'])))
np.testing.assert_array_equal(union,d['training']['union_parameters'])
draws=np.concatenate((sample(cfg['existing_development_seed'],cfg['existing_development_count']),sample(cfg['additional_development_seed'],cfg['additional_development_count'])))
np.testing.assert_array_equal(draws,d['cohort']['parameters'])
assert sha(np.ascontiguousarray(union).tobytes())==d['training']['union_sha256']
assert sha(np.ascontiguousarray(draws).tobytes())==d['cohort']['parameter_sha256']
den=np.asarray(d['training']['data']['mean_squares'])
assert np.all(den>0)
normalizer=d['training']['shared_global_normalizer']
assert abs(normalizer/den[:cfg['original_training_prefix_count']].mean()-1)<1e-12
training_norm_differences=[]
train_n=cfg['training_nodes_per_axis']-1;x=np.arange(1,train_n)/train_n
lam1=4*train_n**2*np.sin(np.pi*np.arange(1,train_n)/(2*train_n))**2
for i in (0,127,511,512,1023,len(union)-1):
    cx,cy,w,amp=union[i]
    f=amp*np.exp(-((x[:,None]-cx)**2+(x[None,:]-cy)**2)/(2*w*w))
    u=dstn(dstn(f,type=1,norm='ortho')/(lam1[:,None]+lam1[None,:]),type=1,norm='ortho')
    error=abs(np.mean(u*u)/den[i]-1);assert error<1e-10;training_norm_differences.append(error)
for coverage in ('original','expanded'):
    aa=[r for r in d['training']['arms'] if r['tag'].startswith(coverage)]
    assert len(aa)==2 and len({r['initial_codes_sha256'] for r in aa})==1
assert len({r['initial_weights_sha256'] for r in d['training']['arms']})==1
endpoints=[]
for checkpoint in d['checkpoints']:
    name='in/model.pkl' if checkpoint['model']=='original_frozen' else 'out/pilot/'+checkpoint['path']
    payload=files[name];assert sha(payload)==checkpoint['sha256']
    saved=pickle.loads(payload);assert saved['Z_tr'].shape==(checkpoint['training_count'],16)
    if checkpoint['model']!='original_frozen':
        record=next(r for r in d['training']['arms'] if r['tag']==checkpoint['model'])
        assert record==saved['cfg']['continuation']
        assert checkpoint['steps_done']==record['steps_done']
        valid=record['finite'] and not record['time_capped'] and record['steps_done']==cfg['training_steps_each']
        assert checkpoint['valid_endpoint']==record['valid_endpoint']==valid
        assert record['elapsed_seconds_including_compile']>=record['compile_seconds']+record['optimizer_loop_seconds']
        endpoints.append(dict(model=checkpoint['model'],steps=record['steps_done'],valid=valid))
errors=[];keys=set();components=0;cache={}
def metrics(field_hash,n,case):
    key=(field_hash,n,case)
    if key not in cache:
        field=fields[field_hash];fine=reference[case]['observation'];coarser=reference[case]['coarser_observation']
        delta=rel(coarser,fine);error=rel(field[::n//cfg['observation_intervals'],::n//cfg['observation_intervals']],fine)
        cache[key]=dict(physical_error=error,reference_delta=delta,conservative_physical_error=(error+delta)/(1-delta),same_grid_error=rel(field,truth[(n,case)]))
    return cache[key]
for row in d['rows']:
    key=(row['intervals'],row['case'],row['model'],row['arm'],row['requested_modes'],row['tau'],row['repetition'])
    assert key not in keys;keys.add(key)
    names=['input_seconds','fused_device_seconds','output_seconds'] if row['arm'] in ('rom_fused','rom_gj') else ['input_seconds','projection_init_seconds','solver_seconds','output_seconds']
    assert abs(row['total_seconds']-sum(row[k] for k in names))<1e-10;components+=1
    computed=metrics(row['field_sha256'],row['intervals'],row['case'])
    diffs=[abs(value-row[key]) for key,value in computed.items()]
    errors+=diffs
    assert max(diffs)<1e-10
    if row['model'] is not None:
        assert row['stationary']==(row['stationarity']<=cfg['stationarity_tolerance'])
        assert row['solver_valid']==(row['finite'] and row['parity_passed'] and (row['tau_reached'] or row['stationary']))
    if row['arm']=='rom_gj':
        assert row['fallback_count']>=0 and row['fallback_count']<=row['attempts']
        if row['solver_valid']:assert row['max_linear_backward_error']<=cfg['linear_backward_limit']
expected=len(cfg['intervals'])*len(draws)*cfg['repetitions']*(2+3*len(cfg['taus'])*len(cfg['arms']))
assert len(keys)==expected
oracle_errors=[]
for item in d['oracles']:
    n=item['intervals'];case=item['case']
    for name in ('bank_metrics','best_head_metrics'):
        computed=metrics(item[name]['field_sha256'],n,case)
        oracle_errors.extend(abs(value-item[name][key]) for key,value in computed.items())
    oracle_errors.extend((abs(item['bank_metrics']['same_grid_error']-item['full_bank_same_grid_error']),abs(item['best_head_metrics']['same_grid_error']-item['best_same_grid_error'])))
    assert item['full_bank_same_grid_error']<=item['best_same_grid_error']+1e-12
    assert max(x['qr_identity_absolute'] for x in item['rows'])<1e-10
assert max(oracle_errors)<1e-10
report=dict(passed=True,job_id=sub['job_id'],source_commit=sub['source_commit'],archive_sha256=clean['archive_sha256'],
    row_count=len(keys),distinct_preserved_field_hashes=len(fields),same_invocation_components_checked=components,
    maximum_independent_cpu_metric_difference=max(errors),maximum_oracle_cpu_error_difference=max(oracle_errors),
    exact_source_draws_and_training_prefix_verified=True,training_denominator_cpu_sample_relative_difference_max=max(training_norm_differences),
    shared_initialization_across_objectives_verified=True,training_endpoints=endpoints,
    all_training_endpoints_complete=all(x['valid'] for x in endpoints),trained_checkpoint_hashes_and_metadata_verified=True,
    source_checkpoint_and_result_match_archive_and_git=True,all_reference_differences_recomputed=True,
    all_original_fusion_parity_passed=all(g['original_fusion']['passed'] for g in d['parity']),
    specialized_agreement_failures=sum(not g['passed'] for g in d['parity']),
    specialized_field_relative_max=max(g['field_relative'] for g in d['parity']),
    specialized_latent_relative_max=max(g['latent_relative'] for g in d['parity']),
    specialized_timed_fallbacks=sum(r.get('fallback_count',0) for r in d['rows']),remote_deleted=True,
    reference_limit='Fine observation artifacts audited; no rigorous continuum bound established',
    oracle_limit='All starts saved, but stationary local fits do not prove global minima')
(run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
