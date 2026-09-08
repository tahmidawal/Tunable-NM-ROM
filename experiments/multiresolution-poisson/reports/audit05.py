"""CPU audit of saved source/solution/cache bytes and initialization diagnostics."""
import argparse,hashlib,io,json,pickle,subprocess,tarfile
from pathlib import Path
import numpy as np
from scipy.fft import dstn
from scipy.special import expit

ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve()
tree=Path(__file__).resolve().parents[3];d=json.loads((run/'result.json').read_text());cfg=d['config']
sub=json.loads((run/'submission.json').read_text());clean=json.loads((run/'cleanup.json').read_text())
sha=lambda x:hashlib.sha256(x).hexdigest()
array_sha=lambda x:sha(np.ascontiguousarray(x).tobytes())
archive=run/'verified-cluster.tar.gz'
if not archive.exists():
    manifest=json.loads((run/'ARCHIVE.json').read_text())
    with archive.open('wb') as stream:
        for part in manifest['ordered_parts']:
            payload=(run/part['name']).read_bytes();assert sha(payload)==part['sha256'];stream.write(payload)
assert sha(archive.read_bytes())==clean['archive_sha256']
assert d['complete'] and clean['remote_deleted_and_absence_checked']
assert d['provenance']['commit']==sub['source_commit'] and d['provenance']['job_id']==sub['job_id']
assert d['provenance']['backend']=='gpu' and d['provenance']['x64'] and d['provenance']['matmul_precision']=='highest'
files={};fields={};refs={};caches={}
with tarfile.open(archive,'r|gz') as tar:
    for entry in tar:
        if not entry.isfile():continue
        name=entry.name.removeprefix('cluster/');stream=tar.extractfile(entry)
        if name.startswith('code/') or name.startswith('in/') or name=='out/pilot/result.json':files[name]=stream.read()
        elif name.startswith('out/pilot/') and name.endswith('.npz'):
            with np.load(io.BytesIO(stream.read())) as z:arrays={k:z[k] for k in z.files}
            if 'field' in arrays:
                field=arrays['field'];fields[array_sha(field)]=field
            elif 'observation' in arrays:refs[int(Path(name).stem.split('case')[1])]=arrays
            else:
                tag,n=Path(name).stem.removeprefix('cache_').rsplit('_',1);caches[tag,int(n)]=arrays
assert files['out/pilot/result.json']==(run/'result.json').read_bytes()
pathmap={'sep_common.py':'experiments/separable-decoder/sep_common.py','ctol_tol.py':'experiments/cost-to-tolerance/ctol_tol.py',
    'ms_parametric.py':'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py'}
for staged,digest in sub['source_hashes'].items():
    payload=files[staged];assert sha(payload)==digest
    name=Path(staged).name;path=pathmap.get(name,'experiments/multiresolution-poisson/'+name)
    assert payload==subprocess.check_output(['git','show',sub['source_commit']+':'+path],cwd=tree)
origin=json.loads(files['in/ORIGIN.json']);models={}
for tag,name,path in [('original_frozen','model.pkl',origin['checkpoint_path']),('original_relative','selected.pkl',origin['selected_checkpoint_path'])]:
    payload=files['in/'+name];assert sha(payload)==cfg['checkpoint_sha256'][tag]
    assert payload==subprocess.check_output(['git','show',sub['source_commit']+':'+path],cwd=tree)
    models[tag]=pickle.loads(payload)
log=(run/(sub['job_id']+'.out')).read_text()+(run/(sub['job_id']+'.err')).read_text()
assert 'jax_backend=gpu' in log and 'ALL-DONE' in log and (run/'EXIT_CODE').read_text().strip()=='0'
for bad in ['cuInit','captured constant','large constant','OUT_OF_MEMORY','No space left','Traceback']:assert bad not in log,bad
draws=np.asarray(d['cohort']['parameters']);assert array_sha(draws)==cfg['parameter_sha256']==d['cohort']['parameter_sha256']
previous=json.loads((run.parent/'pilot04/result.json').read_text())
np.testing.assert_array_equal(draws,previous['cohort']['parameters'])
def sample(seed,count):
    rng=np.random.default_rng(seed)
    return np.column_stack((rng.uniform(.15,.85,count),rng.uniform(.15,.85,count),
        np.exp(rng.uniform(np.log(.02),np.log(.1),count)),rng.uniform(.5,2.,count)))
independent=np.concatenate((sample(cfg['existing_development_seed'],cfg['existing_development_count']),sample(cfg['fresh_development_seed'],cfg['fresh_development_count'])))
np.testing.assert_allclose(independent,draws,rtol=5e-14,atol=1e-15)
def mlp(layers,x):
    for w,b in layers[:-1]:v=x@w+b;x=v*expit(v)
    return x@layers[-1][0]+layers[-1][1]
def head(p,z):
    assert 'hB' not in p
    return mlp(p['h'],z)+z@p['h_lin']
def head_jac(p,z):
    x=z.copy();jac=np.eye(len(z))
    for w,b in p['h'][:-1]:
        v=x@w+b;s=expit(v);jac=(jac@w)*(s+v*s*(1-s));x=v*s
    return jac@p['h'][-1][0]+p['h_lin']
relative=lambda x,y:float(np.linalg.norm(x-y)/(np.linalg.norm(y)+1e-300))
banks={};source={};truth={};fm={};lookup={};cache_errors=[];operator_errors=[]
for setup in d['setup']:
    tag,n=setup['model'],setup['intervals'];cache=caches[tag,n];p=models[tag]['params'];codes=models[tag]['Z_tr']
    np.testing.assert_array_equal(cache['codes'],codes)
    assert array_sha(cache['B'])==setup['operator_sha256'] and array_sha(cache['predictions'])==setup['cache']['prediction_sha256']
    assert array_sha(codes)==setup['cache']['training_codes_sha256']
    error=relative(head(p,codes)@cache['B'].T,cache['predictions']);cache_errors.append(error);assert error<1e-11
    x=np.arange(1,n)/n;xx,yy=np.meshgrid(x,x,indexing='ij');coords=np.column_stack((xx.ravel(),yy.ravel()))
    pieces=[]
    for start in range(0,len(coords),16384):
        xy=coords[start:start+16384];ang=2*np.pi*(xy@p['B']);ff=np.concatenate((np.sin(ang),np.cos(ang)),axis=-1)
        bc=16*xy[:,0]*(1-xy[:,0])*xy[:,1]*(1-xy[:,1])
        pieces.append(p['out_scale']*bc[:,None]*mlp(p['g'],ff))
    bank=np.concatenate(pieces);banks[tag,n]=bank
    modes=np.asarray(setup['mode_indices'])-1;ii,jj=modes.T;maxmode=max(modes.ravel())+1
    S=np.sqrt(2/n)*np.sin(np.pi*np.outer(np.arange(1,n),np.arange(1,maxmode+1))/n)
    projected=np.einsum('xa,xyr,yb->abr',S,bank.reshape(n-1,n-1,-1),S,optimize=True)[ii,jj]
    error=relative(projected,cache['B']);operator_errors.append(error);assert error<1e-10
    lam1=4*n*n*np.sin(np.pi*np.arange(1,n)/(2*n))**2;lam=lam1[:,None]+lam1[None,:]
    for case,(cx,cy,width,amp) in enumerate(draws):
        f=amp*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*width*width));source[n,case]=np.pad(f,1)
        truth[n,case]=np.pad(dstn(dstn(f,type=1,norm='ortho')/lam,type=1,norm='ortho'),1)
        target=dstn(f,type=1,norm='ortho')[ii,jj]/lam[ii,jj];fm[tag,n,case]=target
        distances=np.sum((cache['predictions']-target[None,:])**2,axis=1)
        index=int(np.argmin(distances));closest=distances[index];distances[index]=np.inf
        lookup[tag,n,case]=(index,float(np.sqrt(closest)),float(distances.min()-closest))
keys=set();errors=[];field_errors=[];residual_errors=[];gradient_errors=[];lookup_errors=[];native_source={};metric_cache={};reconstruction_cache={}
rows=d['rows']+d['stationary_rows']
assert len(d['rows'])==cfg['expected_timed_invocations'] and len(d['stationary_rows'])==cfg['expected_stationary_invocations']
for row in rows:
    n,case,tag=row['intervals'],row['case'],row['model'];digest=row['field_sha256'];field=fields[digest]
    key=(row['panel'],n,case,tag,row['arm'],row['projection'],row['initialization'],row['repetition']);assert key not in keys;keys.add(key)
    assert field.shape==(n+1,n+1) and field.dtype==np.float64
    assert np.count_nonzero(field[[0,-1]])==0 and np.count_nonzero(field[:,[0,-1]])==0
    previous_hash=native_source.setdefault((n,case),row['source_sha256']);assert previous_hash==row['source_sha256']
    mk=digest,n,case
    if mk not in metric_cache:
        fine=refs[case]['observation'];delta=relative(refs[case]['coarser_observation'],fine)
        error=relative(field[::n//cfg['observation_intervals'],::n//cfg['observation_intervals']],fine)
        metric_cache[mk]=dict(physical_error=error,same_grid_error=relative(field,truth[n,case]),reference_delta=delta,conservative_physical_error=(error+delta)/(1-delta))
    differences=[abs(value-row[name]) for name,value in metric_cache[mk].items()];errors.extend(differences);assert max(differences)<1e-10
    parts=['input_seconds','fused_device_seconds','output_seconds'] if tag else ['input_seconds','projection_init_seconds','solver_seconds','output_seconds']
    assert abs(sum(row[k] for k in parts)-row['total_seconds'])<1e-10
    if tag:
        p=models[tag]['params'];z=np.asarray(row['latent']);z0=np.asarray(row['initial_latent']);cache=caches[tag,n];target=fm[tag,n,case]
        expected_start=models[tag]['Z_tr'].mean(0)
        if row['initialization']=='nearest_cached_scaled_weak_prediction':
            index,dist,gap=lookup[tag,n,case];assert index==row['selected_training_code_index']
            expected_start=cache['codes'][index];lookup_errors.append(abs(dist-row['cache_distance'])/(dist+1e-300))
            assert lookup_errors[-1]<1e-10
            assert abs(gap-row['cache_squared_distance_gap'])/(dist*dist+1e-300)<1e-10
            assert row['cache_near_tie']==(row['cache_relative_squared_gap']<=row['cache_near_tie_relative_threshold'])
        np.testing.assert_allclose(z0,expected_start,rtol=1e-13,atol=1e-14)
        r=cache['B']@head(p,z)-target;r0=cache['B']@head(p,z0)-target
        initial=float(np.linalg.norm(r0));final=float(np.linalg.norm(r))
        residual_errors.extend([abs(initial-row['initial_residual'])/(initial+1e-300),abs(final-row['residual'])/(initial+1e-300)])
        assert max(residual_errors[-2:])<1e-10
        assert row['absolute_tau_threshold']==row['tau']*row['initial_residual']
        jac=cache['B']@head_jac(p,z).T
        gradient=float(np.linalg.norm(jac.T@r)/(np.linalg.norm(jac)*final+1e-300))
        gradient_errors.append(abs(gradient-row['stationarity']));assert gradient_errors[-1]<1e-8
        if (tag,n,digest) not in reconstruction_cache:
            reconstructed=np.pad((banks[tag,n]@head(p,z)).reshape(n-1,n-1),1)
            reconstruction_cache[tag,n,digest]=relative(reconstructed,field)
        field_errors.append(reconstruction_cache[tag,n,digest]);assert field_errors[-1]<1e-9
        assert row['stationary']==(row['stationarity']<=cfg['stationarity_tolerance'])
        assert row['solver_valid']==(row['finite'] and row['parity_passed'] and row['max_linear_backward_error']<=cfg['linear_backward_error_limit'] and (row['tau_reached'] or row['stationary']))
        assert 0<=row['fallback_count']<=row['attempts']
report=dict(passed=True,job_id=sub['job_id'],source_commit=sub['source_commit'],archive_sha256=clean['archive_sha256'],
    primary_invocations=len(d['rows']),stationary_invocations=len(d['stationary_rows']),distinct_preserved_field_hashes=len(fields),
    maximum_independent_cpu_metric_difference=max(errors),maximum_decoder_field_relative_difference=max(field_errors),
    maximum_cpu_cache_prediction_relative_difference=max(cache_errors),maximum_cpu_bank_operator_relative_difference=max(operator_errors),
    maximum_initial_scaled_residual_difference=max(residual_errors),maximum_stationarity_absolute_difference=max(gradient_errors),
    maximum_nearest_distance_relative_difference=max(lookup_errors),
    parameter_regeneration_absolute_difference=float(np.max(np.abs(independent-draws))),all_declared_draws_match_pilot04_exactly=True,
    source_checkpoint_and_result_match_archive_and_git=True,recorded_source_hashes_stable=True,training_only_cache_and_all_selected_codes_verified=True,
    projection_gate_failures=sum(not x['passed'] for x in d['projection_parity']),lookup_index_mismatches=sum(not x['lookup_index_matches'] for x in d['coefficient_checks']),
    near_tie_invocations=sum(bool(r.get('cache_near_tie')) for r in rows),timed_fallbacks=sum(r.get('fallback_count',0) for r in d['rows']),
    stationary_fallbacks=sum(r.get('fallback_count',0) for r in d['stationary_rows']),remote_deleted=True,
    timing_boundary='Full host source through returned host field, stop reason/counters/latents, guard diagnostics and initialization metadata. Independent stationarity recomputation, physical-error and parity scoring occur after the timer, using that same invocation.',
    reference_limit='Empirical nested observation differences audited; no rigorous continuum bound',
    source_regeneration_note='Exact persisted parameter/hash identity across pilots; independent CPU exp regeneration is checked numerically, not claimed bitwise identical across architectures.')
(run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
