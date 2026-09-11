"""Independent CPU audit of full fields, operators, optimality and selection."""
from audit_iterative import *
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tuning_summary import summarize,select


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve();root=Path(__file__).resolve().parents[3]
    local=run/'cluster';out=local/'out/pilot';d=json.loads((out/'result.json').read_text());cfg=d['config'];sub=json.loads((run/'submission.json').read_text());clean=json.loads((run/'cleanup.json').read_text())
    assert d['complete'] and clean['remote_deleted_and_absence_checked'] and file_sha(run/'verified-cluster.tar.gz')==clean['archive_sha256']
    assert d['provenance']['commit']==sub['source_commit'] and d['provenance']['job_id']==sub['job_id']
    assert d['provenance']['backend']=='gpu' and d['provenance']['x64'] and d['provenance']['matmul_precision']=='highest'
    for manifest in ['PULL.sha256','MANIFEST.sha256','RESULTS.sha256']:subprocess.run(['sha256sum','-c',manifest,'--quiet'],cwd=local,check=True)
    pathmap={'sep_common.py':'experiments/separable-decoder/sep_common.py','ctol_tol.py':'experiments/cost-to-tolerance/ctol_tol.py','ms_parametric.py':'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py'}
    for staged,digest in sub['source_hashes'].items():
        payload=(local/staged).read_bytes();assert sha(payload)==digest;name=Path(staged).name;path=pathmap.get(name,'experiments/multiresolution-poisson/'+name)
        assert payload==subprocess.check_output(['git','show',sub['source_commit']+':'+path],cwd=root)
    ck=(local/'in/model.pkl').read_bytes();assert sha(ck)==cfg['checkpoint_sha256'];assert ck==subprocess.check_output(['git','show',sub['source_commit']+':'+cfg['checkpoint']],cwd=root)
    model=pickle.loads(ck);p=model['params'];codes=model['Z_tr'];assert all(a.dtype==np.float64 for a in [codes,p['B'],p['h_lin']]);k=codes.shape[1]
    log='\n'.join(q.read_text() for q in (local/'logs').glob('*'));assert 'jax_backend=gpu' in log and 'ONLINE TUNING COMPLETE' in log and (local/'EXIT_CODE').read_text().strip()=='0'
    for bad in ['cuInit','captured constant','large constant','OUT_OF_MEMORY','No space left','Traceback','Resource exhausted']:assert bad not in log,bad
    draws=np.asarray(d['cohort']['parameters']);assert array_sha(draws)==cfg['parameter_sha256']==d['cohort']['parameter_sha256'];previous=json.loads((run.parent/'iterative_cg06/result.json').read_text());np.testing.assert_array_equal(draws,previous['cohort']['parameters'])
    def sample(seed,count):
        rng=np.random.default_rng(seed);return np.column_stack((rng.uniform(.15,.85,count),rng.uniform(.15,.85,count),np.exp(rng.uniform(np.log(.02),np.log(.1),count)),rng.uniform(.5,2.,count)))
    regenerated=np.concatenate((sample(cfg['existing_development_seed'],cfg['existing_development_count']),sample(cfg['fresh_development_seed'],cfg['fresh_development_count'])));np.testing.assert_allclose(draws,regenerated,atol=1e-15,rtol=5e-14)
    presets={p['id']:p for p in cfg['presets']};screen_rows=[r for r in d['rows'] if r['phase']=='screen'];screen=summarize(screen_rows,cfg);selected=select(screen,list(presets));assert selected==d['selection']['selections'] and screen==d['selection']['screen_methods']
    assert file_sha(out/'SELECTION.json')==d['selection_sha256'];assert json.loads((out/'SELECTION.json').read_text())==d['selection']
    shortlist=['nmrom_baseline']
    for name in selected.values():
        if name is not None and name not in shortlist:shortlist.append(name)
    assert shortlist==d['selection']['confirmed_preset_ids'];assert d['selection']['presets']=={x:presets[x] for x in shortlist}
    first_confirmation=next(i for i,r in enumerate(d['rows']) if r['phase']=='confirmation');assert all(r['phase']=='screen' for r in d['rows'][:first_confirmation])
    keys=set();groups=defaultdict(list)
    for r in d['rows']:
        key=(r['phase'],r['intervals'],r['case'],r['method'],r['repetition']);assert key not in keys;keys.add(key);groups[r['intervals'],r['case']].append(r)
        assert abs(r['total_seconds']-sum(r[x] for x in ['input_seconds','fused_device_seconds','output_seconds']))<1e-10
        assert all(r[x]>0 for x in ['total_seconds','input_seconds','fused_device_seconds','output_seconds'])
    expected=set()
    controls=[f'cg_{t:.0e}' for t in cfg['cg_tolerances']]+['dst']
    for phase,meshes,ids,reps in [('screen',[cfg['screen_intervals']],list(presets),cfg['screen_repetitions']),('confirmation',cfg['intervals'],shortlist,cfg['repetitions'])]:
        for n in meshes:
            for c in range(len(draws)):
                for m in ids+controls:
                    for r in range(reps):expected.add((phase,n,c,m,r))
    assert keys==expected
    metrics=[];refs_errors=[];bank_errors=[];cache_errors=[];decode_errors=[];residual_errors=[];gradient_errors=[];cg_errors=[];baseline_parity=[];allhash=set();audited_starts=0
    for n in sorted(set(cfg['intervals']+[cfg['screen_intervals']])):
        print('audit bank',n,flush=True);g=bank(p,n);lam=eigenvalues(n);setups=[s for s in d['setup'] if s['intervals']==n];operators={}
        for setup in setups:
            M=setup['requested_modes'];phase=setup['phase'];cache=dict(np.load(out/f'cache_{phase}_n{n}_M{M}.npz'));B=cache['B'];ii,jj=(np.asarray(setup['mode_indices'])-1).T
            assert setup['retained_modes']==len(ii)>=4*k and array_sha(B)==setup['operator_sha256'];np.testing.assert_array_equal(cache['codes'],codes)
            expectedii,expectedjj=np.nonzero(lam<=np.sort(lam.ravel())[M-1]);np.testing.assert_array_equal(ii,expectedii);np.testing.assert_array_equal(jj,expectedjj)
            maxmode=int(max(ii.max(),jj.max()))+1;S=np.sqrt(2/n)*np.sin(np.pi*np.outer(np.arange(1,n),np.arange(1,maxmode+1))/n)
            cpuB=np.einsum('xa,xyr,yb->abr',S,g.reshape(n-1,n-1,-1),S,optimize=True)[ii,jj];bank_errors.append(relative(cpuB,B));assert bank_errors[-1]<1e-10
            cache_errors.append(relative(head(p,codes)@B.T,cache['predictions']));assert cache_errors[-1]<1e-11;assert array_sha(cache['predictions'])==setup['cache']['prediction_sha256']
            operators[phase,M]=(setup,cache,ii,jj)
        for case,param in enumerate(draws):
            refs=dict(np.load(out/'references'/f'n{n}_case{case}.npz'))
            for nr,key in zip(cfg['reference_intervals'],['coarse','fine']):
                exact=truth(nr,param)[::nr//n,::nr//n];refs_errors.append(relative(exact,refs[key]));assert refs_errors[-1]<1e-12
            fine=refs['fine'];delta=relative(refs['coarse'],fine);f=source(n,param);same=truth(n,param);fourier=dstn(f[1:-1,1:-1],type=1,norm='ortho')/lam;fieldcache={};fieldmetrics={};decoded=set()
            for row in groups[n,case]:
                digest=row['field_sha256'];allhash.add(digest)
                if digest not in fieldcache:
                    field=np.load(out/'fields'/(digest+'.npz'))['field'];assert array_sha(field)==digest and field.shape==(n+1,n+1) and field.dtype==np.float64 and np.isfinite(field).all();assert not np.count_nonzero(field[[0,-1]]) and not np.count_nonzero(field[:,[0,-1]])
                    fieldcache[digest]=field;error=relative(field,fine);stride=n//cfg['observation_intervals'];fieldmetrics[digest]=dict(physical_error=error,same_grid_error=relative(field,same),reference_delta=delta,conservative_physical_error=(error+delta)/(1-delta),common_observation_error=relative(field[::stride,::stride],fine[::stride,::stride]))
                field=fieldcache[digest];diffs=[abs(row[key]-value) for key,value in fieldmetrics[digest].items()];metrics.extend(diffs);assert max(diffs)<1e-10 and row['finite']
                if row['method'] in presets:
                    preset=presets[row['method']];assert row['preset']==preset;setup,cache,ii,jj=operators[row['phase'],preset['requested_modes']];B=cache['B'];target=fourier[ii,jj];distances=np.sum((cache['predictions']-target)**2,axis=1);nearest=np.argsort(distances)[:preset['starts']]
                    assert row['retained_modes']==setup['retained_modes'] and row['operator_sha256']==setup['operator_sha256']
                    assert [s['selected_training_code_index'] for s in row['starts']]==nearest.tolist();assert row['selected_start']==int(np.argmin([s['residual'] for s in row['starts']]))
                    for start in row['starts']:
                        audited_starts+=1;latent=np.asarray(start['latent']);initial=np.asarray(start['initial_latent']);np.testing.assert_array_equal(initial,codes[start['selected_training_code_index']])
                        residual=B@head(p,latent)-target;r0=B@head(p,initial)-target;v=np.linalg.norm(residual);v0=np.linalg.norm(r0);J=B@head_jac(p,latent).T;station=np.linalg.norm(J.T@residual)/(np.linalg.norm(J)*v+1e-300)
                        residual_errors.extend([abs(v-start['residual'])/(v0+1e-300),abs(v0-start['initial_residual'])/(v0+1e-300)]);assert max(residual_errors[-2:])<1e-10
                        gradient_errors.append(abs(station-start['stationarity']));assert gradient_errors[-1]<1e-8;assert start['stationary']==(start['stationarity']<=cfg['stationarity_tolerance']);assert start['stationary']==(station<=cfg['stationarity_tolerance'])
                        reason=start['reason'];assert reason in [0,1,2,3,5,6];assert 0<=start['attempts']<=preset['budget'] and start['accepted']<=start['attempts'] and start['jacobians']==start['accepted']+1
                        assert start['absolute_tau_threshold']==preset['tau']*start['initial_residual']
                        if reason==0:assert start['attempts']==preset['budget']
                        if reason==2:assert preset['tau']>0 and start['residual']<=start['absolute_tau_threshold']
                        if reason==6:assert preset['stationarity_stop'] is not None and start['stationarity']<=preset['stationarity_stop']*(1+1e-7)+1e-12
                        assert 0<=start['fallback_count']<=start['attempts']
                    winner=row['starts'][row['selected_start']]
                    for key,val in winner.items():assert row[key]==val
                    assert row['total_attempts']==sum(s['attempts'] for s in row['starts']) and row['total_jacobians']==sum(s['jacobians'] for s in row['starts'])
                    assert row['all_start_linear_valid']==all(s['max_linear_backward_error']<=cfg['linear_backward_error_limit'] for s in row['starts'])
                    assert row['solver_valid']==bool(row['finite'] and row['all_start_linear_valid'] and (row['stationary'] or row['reason']==2))
                    if digest not in decoded:
                        reconstruction=np.pad((g@head(p,np.asarray(row['latent']))).reshape(n-1,n-1),1);decode_errors.append(relative(reconstruction,field));assert decode_errors[-1]<1e-9;decoded.add(digest)
                    if row['method']=='nmrom_baseline' and row['repetition']==0:
                        old=next(r for r in previous['rows'] if r['intervals']==n and r['case']==case and r['method']=='nmrom' and r['repetition']==0);oldfield=np.load(run.parent/'iterative_cg06/cluster/out/pilot/fields'/(old['field_sha256']+'.npz'))['field'];baseline_parity.append(relative(field,oldfield));assert baseline_parity[-1]<1e-8
                elif row['method'].startswith('cg_'):
                    interior=field[1:-1,1:-1];applied=n*n*(4*interior-field[:-2,1:-1]-field[2:,1:-1]-field[1:-1,:-2]-field[1:-1,2:]);rr=relative(applied,f[1:-1,1:-1]);cg_errors.append(abs(rr-row['true_relative_residual']));assert cg_errors[-1]<1e-9
                    assert row['cg_converged']==(row['true_relative_residual']<=row['cg_tolerance']*(1+1e-6)+1e-12) and row['solver_valid']==row['cg_converged']
                else:assert relative(field,same)<1e-11
            print('audit case',n,case,flush=True)
        del g
    assert len(allhash)==len(list((out/'fields').glob('*.npz')))
    report=dict(passed=True,audit_script_sha256=file_sha(Path(__file__)),summary_module_sha256=file_sha(Path(__file__).resolve().parents[1]/'tuning_summary.py'),result_sha256=file_sha(out/'result.json'),selection_sha256=file_sha(out/'SELECTION.json'),job_id=sub['job_id'],source_commit=sub['source_commit'],archive_sha256=clean['archive_sha256'],timed_invocations=len(d['rows']),screen_invocations=len(screen_rows),confirmation_invocations=len(d['rows'])-len(screen_rows),distinct_full_fields=len(allhash),audited_start_records=audited_starts,
        maximum_metric_disagreement=max(metrics),maximum_reference_disagreement=max(refs_errors),maximum_cpu_bank_operator_relative=max(bank_errors),maximum_cpu_cache_prediction_relative=max(cache_errors),maximum_decoder_field_relative=max(decode_errors),maximum_initial_scaled_residual_difference=max(residual_errors),maximum_stationarity_absolute_difference=max(gradient_errors),maximum_true_cg_residual_absolute_difference=max(cg_errors),maximum_previous_baseline_field_relative_difference=max(baseline_parity),
        all_fields_hashed_and_independently_scored=True,all_source_checkpoint_hashes_match_git=True,all_timing_components_sum=True,all_expected_invocations_present=True,screen_selection_recomputed=True,shortlist_hash_verified=True,all_multistart_choices_use_weak_residual_only=True,all_stopping_counters_verified=True,all_operators_refit_without_quadrature=True,remote_deleted=True,
        limitation='Development selection and confirmation reuse already-open sources; reference refinement is empirical, not a rigorous continuum bound; intentional truncations are retained as diagnostics.')
    (run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');(run/'result.json').write_bytes((out/'result.json').read_bytes());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
