"""Independent NumPy/SciPy audit of every saved iterative-panel invocation."""
import argparse,hashlib,json,pickle,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.fft import dstn
from scipy.special import expit

sha=lambda b:hashlib.sha256(b).hexdigest()
array_sha=lambda a:sha(np.ascontiguousarray(a).tobytes())
relative=lambda a,b:float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-300))


def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):digest.update(block)
    return digest.hexdigest()


def mlp(layers,x):
    for w,b in layers[:-1]:v=x@w+b;x=v*expit(v)
    return x@layers[-1][0]+layers[-1][1]


def head(p,z):return mlp(p['h'],z)+z@p['h_lin']


def head_jac(p,z):
    x=z.copy();jac=np.eye(len(z))
    for w,b in p['h'][:-1]:
        v=x@w+b;s=expit(v);jac=(jac@w)*(s+v*s*(1-s));x=v*s
    return jac@p['h'][-1][0]+p['h_lin']


def bank(p,n):
    x=np.arange(1,n)/n;xx,yy=np.meshgrid(x,x,indexing='ij');coords=np.column_stack((xx.ravel(),yy.ravel()))
    parts=[]
    for start in range(0,len(coords),16384):
        xy=coords[start:start+16384];ang=2*np.pi*(xy@p['B']);ff=np.concatenate((np.sin(ang),np.cos(ang)),axis=-1)
        bc=16*xy[:,0]*(1-xy[:,0])*xy[:,1]*(1-xy[:,1]);parts.append(p['out_scale']*bc[:,None]*mlp(p['g'],ff))
    return np.concatenate(parts)


def source(n,param):
    cx,cy,width,amp=param;x=np.arange(1,n)/n
    return np.pad(amp*np.exp(-((x[:,None]-cx)**2+(x[None,:]-cy)**2)/(2*width*width)),1)


def eigenvalues(n):
    lam=4*n*n*np.sin(np.pi*np.arange(1,n)/(2*n))**2
    return lam[:,None]+lam[None,:]


def truth(n,param):
    f=source(n,param)[1:-1,1:-1]
    return np.pad(dstn(dstn(f,type=1,norm='ortho')/eigenvalues(n),type=1,norm='ortho'),1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve()
    root=Path(__file__).resolve().parents[3];local=run/'cluster';out=local/'out/pilot'
    d=json.loads((out/'result.json').read_text());cfg=d['config'];sub=json.loads((run/'submission.json').read_text());clean=json.loads((run/'cleanup.json').read_text())
    assert d['complete'] and clean['remote_deleted_and_absence_checked']
    assert file_sha(run/'verified-cluster.tar.gz')==clean['archive_sha256']
    assert d['provenance']['commit']==sub['source_commit'] and d['provenance']['job_id']==sub['job_id']
    assert d['provenance']['backend']=='gpu' and d['provenance']['x64'] and d['provenance']['matmul_precision']=='highest'
    for manifest in ['PULL.sha256','MANIFEST.sha256','RESULTS.sha256']:
        subprocess.run(['sha256sum','-c',manifest,'--quiet'],cwd=local,check=True)
    pathmap={'sep_common.py':'experiments/separable-decoder/sep_common.py','ctol_tol.py':'experiments/cost-to-tolerance/ctol_tol.py','ms_parametric.py':'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py'}
    for staged,digest in sub['source_hashes'].items():
        payload=(local/staged).read_bytes();assert sha(payload)==digest;name=Path(staged).name
        path=pathmap.get(name,'experiments/multiresolution-poisson/'+name)
        assert payload==subprocess.check_output(['git','show',sub['source_commit']+':'+path],cwd=root)
    ck=(local/'in/model.pkl').read_bytes();assert sha(ck)==cfg['checkpoint_sha256']
    assert ck==subprocess.check_output(['git','show',sub['source_commit']+':'+cfg['checkpoint']],cwd=root)
    model=pickle.loads(ck);p=model['params'];codes=model['Z_tr'];assert all(a.dtype==np.float64 for a in [codes,p['B'],p['h_lin']])
    log='\n'.join(q.read_text() for q in (local/'logs').glob('*'))
    assert 'jax_backend=gpu' in log and 'ITERATIVE POISSON COMPLETE' in log and (local/'EXIT_CODE').read_text().strip()=='0'
    for bad in ['cuInit','captured constant','large constant','OUT_OF_MEMORY','No space left','Traceback','Resource exhausted']:assert bad not in log,bad
    draws=np.asarray(d['cohort']['parameters']);assert array_sha(draws)==cfg['parameter_sha256']==d['cohort']['parameter_sha256']
    previous=json.loads((run.parent/'pilot05/result.json').read_text());np.testing.assert_array_equal(draws,previous['cohort']['parameters'])
    def sample(seed,count):
        rng=np.random.default_rng(seed)
        return np.column_stack((rng.uniform(.15,.85,count),rng.uniform(.15,.85,count),np.exp(rng.uniform(np.log(.02),np.log(.1),count)),rng.uniform(.5,2.,count)))
    regenerated=np.concatenate((sample(cfg['existing_development_seed'],cfg['existing_development_count']),sample(cfg['fresh_development_seed'],cfg['fresh_development_count'])))
    np.testing.assert_allclose(draws,regenerated,atol=1e-15,rtol=5e-14)
    assert len(d['rows'])==cfg['expected_timed_invocations']
    rows=defaultdict(list);keys=set();field_hashes=set();metrics=[];ref_errors=[];bank_errors=[];decoded_errors=[];cache_errors=[];gradient_errors=[];residual_errors=[];cg_residual_errors=[];parity=[];ref_deltas=[]
    for row in d['rows']:
        key=row['intervals'],row['case'],row['method'],row['repetition'];assert key not in keys;keys.add(key);rows[row['intervals'],row['case']].append(row)
        assert abs(sum(row[k] for k in ['input_seconds','fused_device_seconds','output_seconds'])-row['total_seconds'])<1e-10
        assert all(row[k]>0 for k in ['total_seconds','fused_device_seconds','input_seconds','output_seconds'])
    methods=['nmrom']+[f'cg_{t:.0e}' for t in cfg['cg_tolerances']]+['dst']
    for n in cfg['intervals']:
        print('audit bank',n,flush=True);g=bank(p,n);cache=dict(np.load(out/f'cache_n{n}.npz'));B=cache['B']
        setup=next(s for s in d['setup'] if s['intervals']==n);modes=np.asarray(setup['mode_indices'])-1;ii,jj=modes.T
        assert array_sha(B)==setup['operator_sha256'];np.testing.assert_array_equal(cache['codes'],codes)
        maxmode=int(max(ii.max(),jj.max()))+1;S=np.sqrt(2/n)*np.sin(np.pi*np.outer(np.arange(1,n),np.arange(1,maxmode+1))/n)
        native=np.einsum('xa,xyr,yb->abr',S,g.reshape(n-1,n-1,-1),S,optimize=True)[ii,jj]
        bank_errors.append(relative(native,B));assert bank_errors[-1]<1e-10
        cache_errors.append(relative(head(p,codes)@B.T,cache['predictions']));assert cache_errors[-1]<1e-11
        assert array_sha(cache['predictions'])==setup['cache']['prediction_sha256'];lam=eigenvalues(n)
        for case,param in enumerate(draws):
            expectedkeys={(n,case,m,r) for m in methods for r in range(cfg['repetitions'])};assert expectedkeys.issubset(keys)
            refs=dict(np.load(out/'references'/f'n{n}_case{case}.npz'))
            for nr,key in zip(cfg['reference_intervals'],['coarse','fine']):
                exact=truth(nr,param)[::nr//n,::nr//n];ref_errors.append(relative(exact,refs[key]));assert ref_errors[-1]<1e-12
            fine=refs['fine'];delta=relative(refs['coarse'],fine);ref_deltas.append(delta)
            f=source(n,param);same=truth(n,param);target=dstn(f[1:-1,1:-1],type=1,norm='ortho')[ii,jj]/lam[ii,jj]
            distances=np.sum((cache['predictions']-target[None,:])**2,axis=1);nearest=int(np.argmin(distances));source_hashes={r['source_sha256'] for r in rows[n,case]};assert len(source_hashes)==1
            fieldcache={};metriccache={}
            for row in rows[n,case]:
                digest=row['field_sha256'];field_hashes.add(digest)
                if digest not in fieldcache:
                    field=np.load(out/'fields'/(digest+'.npz'))['field'];assert array_sha(field)==digest
                    assert field.shape==(n+1,n+1) and field.dtype==np.float64 and np.isfinite(field).all()
                    assert np.count_nonzero(field[[0,-1]])==0 and np.count_nonzero(field[:,[0,-1]])==0
                    fieldcache[digest]=field;error=relative(field,fine);stride=n//cfg['observation_intervals']
                    metriccache[digest]=dict(physical_error=error,same_grid_error=relative(field,same),reference_delta=delta,
                        conservative_physical_error=(error+delta)/(1-delta),common_observation_error=relative(field[::stride,::stride],fine[::stride,::stride]))
                field=fieldcache[digest]
                diffs=[abs(row[k]-v) for k,v in metriccache[digest].items()];metrics.extend(diffs);assert max(diffs)<1e-10
                assert row['finite']
                if row['method']=='nmrom':
                    z=np.asarray(row['latent']);z0=np.asarray(row['initial_latent']);assert row['selected_training_code_index']==nearest
                    np.testing.assert_array_equal(z0,codes[nearest]);r=B@head(p,z)-target;r0=B@head(p,z0)-target
                    initial=np.linalg.norm(r0);final=np.linalg.norm(r);J=B@head_jac(p,z).T
                    gradient=np.linalg.norm(J.T@r)/(np.linalg.norm(J)*final+1e-300)
                    residual_errors.extend([abs(initial-row['initial_residual'])/(initial+1e-300),abs(final-row['residual'])/(initial+1e-300)]);assert max(residual_errors[-2:])<1e-10
                    gradient_errors.append(abs(gradient-row['stationarity']));assert gradient_errors[-1]<1e-8
                    assert row['stationary']==(row['stationarity']<=cfg['stationarity_tolerance'])
                    valid=row['finite'] and row['max_linear_backward_error']<=cfg['linear_backward_error_limit'] and (row['stationary'] or row['reason']==2)
                    assert row['solver_valid']==valid and 0<=row['fallback_count']<=row['attempts']
                    if row['repetition']==0:
                        decoded=np.pad((g@head(p,z)).reshape(n-1,n-1),1);decoded_errors.append(relative(decoded,field));assert decoded_errors[-1]<1e-9
                        if n==256:
                            old=next(r for r in previous['rows'] if r['model']=='original_relative' and r['projection']==cfg['projection'] and r['initialization']==cfg['initialization'] and r['intervals']==n and r['case']==case and r['repetition']==0)
                            oldpath=run.parent/'pilot05/cluster/out/pilot/fields'/(old['field_sha256']+'.npz')
                            oldfield=np.load(oldpath)['field'];parity.append(relative(field,oldfield));assert parity[-1]<1e-8
                elif row['method'].startswith('cg_'):
                    interior=field[1:-1,1:-1]
                    applied=n*n*(4*interior-field[:-2,1:-1]-field[2:,1:-1]-field[1:-1,:-2]-field[1:-1,2:])
                    rr=relative(applied,f[1:-1,1:-1]);cg_residual_errors.append(abs(rr-row['true_relative_residual']));assert cg_residual_errors[-1]<1e-9
                    tol=row['cg_tolerance'];assert row['cg_converged']==(row['true_relative_residual']<=tol*(1+1e-6)+1e-12)
                    assert row['solver_valid']==row['cg_converged'] and 0<=row['iterations']<=cfg['cg_maxiter']
                else:assert relative(field,same)<1e-11
            print('audit case',n,case,flush=True)
        del g
    assert len(field_hashes)==len(list((out/'fields').glob('*.npz')))
    report=dict(passed=True,job_id=sub['job_id'],source_commit=sub['source_commit'],archive_sha256=clean['archive_sha256'],timed_invocations=len(d['rows']),distinct_full_fields=len(field_hashes),
        maximum_metric_disagreement=max(metrics),maximum_reference_disagreement=max(ref_errors),maximum_cpu_bank_operator_relative=max(bank_errors),maximum_cpu_cache_prediction_relative=max(cache_errors),
        maximum_decoder_field_relative=max(decoded_errors),maximum_initial_scaled_residual_difference=max(residual_errors),maximum_stationarity_absolute_difference=max(gradient_errors),
        maximum_true_cg_residual_absolute_difference=max(cg_residual_errors),maximum_original_pilot05_field_relative_difference=max(parity),maximum_empirical_reference_delta=max(ref_deltas),
        parameter_regeneration_absolute_difference=float(np.max(np.abs(draws-regenerated))),all_frozen_development_parameters_match_prior_archive=True,
        all_source_checkpoint_hashes_match_git=True,all_timing_components_sum=True,all_fields_hashed_and_independently_scored=True,all_requested_cases_methods_repetitions_present=True,
        invalid_invocations=sum(not r['solver_valid'] for r in d['rows']),nonstationary_rom_invocations=sum(r['method']=='nmrom' and not r['stationary'] for r in d['rows']),
        cg_nonconverged_invocations=sum(r['method'].startswith('cg_') and not r['cg_converged'] for r in d['rows']),remote_deleted=True,
        limitation='Empirical finite-difference refinement is not a rigorous continuum bound. Source regeneration checked numerically; ARM/x86 exp is not bitwise identical.')
    (run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');(run/'result.json').write_bytes((out/'result.json').read_bytes());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
