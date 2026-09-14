"""Independent NumPy/SciPy training, multi-checkpoint and every-field audit."""
from audit_iterative import *


def leaves(x):
    if isinstance(x,dict):
        for key in sorted(x):yield from leaves(x[key])
    elif isinstance(x,(tuple,list)):
        for value in x:yield from leaves(value)
    else:yield np.asarray(x)


def weights_sha(p):
    h=hashlib.sha256()
    for v in leaves(p):h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def same_tree(a,b):
    aa=list(leaves(a));bb=list(leaves(b));assert len(aa)==len(bb)
    for x,y in zip(aa,bb):np.testing.assert_array_equal(x,y)


def sample(seed,count):
    rng=np.random.default_rng(seed);return np.column_stack((rng.uniform(.15,.85,count),rng.uniform(.15,.85,count),np.exp(rng.uniform(np.log(.02),np.log(.1),count)),rng.uniform(.5,2.,count)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve();root=Path(__file__).resolve().parents[3];local=run/'cluster';out=local/'out/pilot'
    d=json.loads((out/'result.json').read_text());cfg=d['config'];sub=json.loads((run/'submission.json').read_text());clean=json.loads((run/'cleanup.json').read_text());assert d['complete'] and clean['remote_deleted_and_absence_checked']
    assert file_sha(run/'verified-cluster.tar.gz')==clean['archive_sha256'] and d['provenance']['commit']==sub['source_commit'] and d['provenance']['job_id']==sub['job_id']
    assert d['provenance']['backend']=='gpu' and d['provenance']['x64'] and d['provenance']['matmul_precision']=='highest'
    for manifest in ['PULL.sha256','MANIFEST.sha256','RESULTS.sha256']:subprocess.run(['sha256sum','-c',manifest,'--quiet'],cwd=local,check=True)
    mapping={'sep_common.py':'experiments/separable-decoder/sep_common.py','ctol_tol.py':'experiments/cost-to-tolerance/ctol_tol.py','ms_parametric.py':'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py'}
    for staged,expected in sub['source_hashes'].items():
        payload=(local/staged).read_bytes();name=Path(staged).name;assert sha(payload)==expected;assert payload==subprocess.check_output(['git','show',sub['source_commit']+':'+mapping.get(name,'experiments/multiresolution-poisson/'+name)],cwd=root)
    initial_bytes=(local/'in/model.pkl').read_bytes();assert sha(initial_bytes)==cfg['checkpoint_sha256'];assert initial_bytes==subprocess.check_output(['git','show',sub['source_commit']+':'+cfg['checkpoint']],cwd=root);initial=pickle.loads(initial_bytes)
    log='\n'.join(p.read_text() for p in (local/'logs').glob('*'));assert 'jax_backend=gpu' in log and log.count('CAPACITY ACCURACY COMPLETE')==2 and (local/'EXIT_CODE').read_text().strip()=='0'
    for bad in ['cuInit','captured constant','large constant','OUT_OF_MEMORY','No space left','Traceback','Resource exhausted']:assert bad not in log,bad
    train=np.asarray(d['training']['parameters']);draws=np.asarray(d['cohort']['parameters']);assert array_sha(train)==d['training']['parameter_sha256'];assert array_sha(draws)==d['cohort']['parameter_sha256']
    train_regenerated=sample(cfg['original_draw_seed'],cfg['original_draw_count'])[:cfg['training_prefix_count']];np.testing.assert_allclose(train,train_regenerated,atol=1e-15,rtol=5e-14)
    old=np.concatenate((sample(cfg['existing_development_seed'],cfg['existing_development_count']),sample(cfg['fresh_development_seed'],cfg['fresh_development_count'])));fresh=sample(cfg['new_development_seed'],cfg['new_development_count']);np.testing.assert_allclose(draws,np.concatenate((old,fresh)),atol=1e-15,rtol=5e-14)
    previous=json.loads((run.parent/'online_tuning07/result.json').read_text());np.testing.assert_array_equal(draws[:len(old)],previous['cohort']['parameters']);assert d['cohort']['groups']==['existing_development']*len(old)+['new_development']*len(fresh)
    baseline_previous=json.loads((run.parent/'staged_accuracy08/result.json').read_text());np.testing.assert_array_equal(draws,baseline_previous['cohort']['parameters'])
    models={};meta={x['id']:x for x in d['checkpoints']}
    for ident,info in meta.items():
        payload=(out/info['path']).read_bytes();assert sha(payload)==info['sha256'];model=pickle.loads(payload);p=model['params'];z=model['Z_tr'];assert weights_sha(p)==info['weights_sha256'] and array_sha(z)==info['codes_sha256']
        assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in leaves((p,z)));rank=128 if ident.startswith('r128') else 64;assert z.shape==(len(train),16) and p['h_lin'].shape==(16,rank) and model['cfg']['r']==rank;models[ident]=(p,z)
    same_tree(models['original_relative'],(initial['params'],initial['Z_tr']))
    metric=[];reference_errors=[];operator_errors=[];cache_errors=[];decode_errors=[];residual_errors=[];gradient_errors=[];cg_errors=[];projection_errors=[];oracle_errors=[];baseline_parity=[];all_fields=set()
    from audit_capacity_training import audit_training
    training_check=audit_training(d,out,models);timing_blocks=training_check['optimizer_blocks_verified'];training_metrics=[training_check['maximum_reconstruction_metric_difference']]
    groups=defaultdict(list);keys=set();control_methods=[f'cg_{t:.0e}' for t in cfg['cg_tolerances']]+['dst'];methods=cfg['trained_model_ids']+control_methods
    for row in d['rows']:
        key=(row['intervals'],row['case'],row['method'],row['repetition']);assert key not in keys;keys.add(key);groups[row['intervals'],row['case']].append(row)
        assert abs(row['total_seconds']-sum(row[k] for k in ['input_seconds','fused_device_seconds','output_seconds']))<1e-10;assert all(row[k]>0 for k in ['total_seconds','input_seconds','fused_device_seconds','output_seconds'])
    assert keys=={(n,c,m,r) for n in cfg['intervals'] for c in range(len(draws)) for m in methods for r in range(cfg['repetitions'])}
    assert len(d['oracles'])==len(cfg['intervals'])*len(draws)*len(cfg['trained_model_ids']);assert len(d['head_oracles'])==len(draws)*len(cfg['trained_model_ids'])
    audited_starts=0;audited_oracle_candidates=0
    for n in cfg['intervals']:
        print('audit banks',n,flush=True);ops={}
        for ident in cfg['trained_model_ids']:
            p,z=models[ident];g=bank(p,n);q,r=np.linalg.qr(g,mode='reduced');sv=np.linalg.svd(r,compute_uv=False);rank=int((sv>sv[0]*max(g.shape)*np.finfo(float).eps).sum());setup=next(s for s in d['setup'] if s['model']==ident and s['intervals']==n);assert setup['rank_valid']==(rank==p['h_lin'].shape[1]) and setup['projection_rank']==rank and setup['rank_valid']
            cache=dict(np.load(out/f'cache_n{n}_{ident}.npz'));B=cache['B'];assert array_sha(B)==setup['operator_sha256'] and array_sha(cache['R'])==setup['R_sha256'];np.testing.assert_array_equal(cache['codes'],z)
            ii,jj=(np.asarray(setup['mode_indices'])-1).T;lam=eigenvalues(n);expected_ij=np.nonzero(lam<=np.sort(lam.ravel())[cfg['requested_modes']-1]);np.testing.assert_array_equal(ii,expected_ij[0]);np.testing.assert_array_equal(jj,expected_ij[1]);assert setup['retained_modes']==len(ii)>=4*z.shape[1]
            maxmode=int(max(ii.max(),jj.max()))+1;S=np.sqrt(2/n)*np.sin(np.pi*np.outer(np.arange(1,n),np.arange(1,maxmode+1))/n);cpuB=np.einsum('xa,xyr,yb->abr',S,g.reshape(n-1,n-1,-1),S,optimize=True)[ii,jj];operator_errors.append(relative(cpuB,B));assert operator_errors[-1]<1e-10
            cache_errors.append(relative(head(p,z)@B.T,cache['predictions']));assert cache_errors[-1]<1e-10;assert relative(cache['R'].T@cache['R'],g.T@g)<1e-10
            ops[ident]=(g,q,r,cache,ii,jj,setup)
        for case,param in enumerate(draws):
            refs=dict(np.load(out/'references'/f'n{n}_case{case}.npz'))
            for nr,key in zip(cfg['reference_intervals'],['coarse','fine']):
                exact=truth(nr,param)[::nr//n,::nr//n];reference_errors.append(relative(exact,refs[key]));assert reference_errors[-1]<1e-12
            fine=refs['fine'];same=truth(n,param);f=source(n,param);delta=relative(refs['coarse'],fine);fourier=dstn(f[1:-1,1:-1],type=1,norm='ortho')/eigenvalues(n);cachefields={};decoded=set()
            def field_load(digest):
                all_fields.add(digest)
                if digest not in cachefields:
                    field=np.load(out/'fields'/(digest+'.npz'))['field'];assert array_sha(field)==digest and field.shape==(n+1,n+1) and field.dtype==np.float64 and np.isfinite(field).all();assert not np.count_nonzero(field[[0,-1]]) and not np.count_nonzero(field[:,[0,-1]]);cachefields[digest]=field
                return cachefields[digest]
            assert len({r['source_sha256'] for r in groups[n,case]})==1
            for row in groups[n,case]:
                field=field_load(row['field_sha256']);error=relative(field,fine);stride=n//cfg['observation_intervals'];values=dict(physical_error=error,same_grid_error=relative(field,same),reference_delta=delta,conservative_physical_error=(error+delta)/(1-delta),common_observation_error=relative(field[::stride,::stride],fine[::stride,::stride]));diff=[abs(row[k]-v) for k,v in values.items()];metric.extend(diff);assert max(diff)<1e-10 and row['finite']
                ident=row['method']
                if ident in models:
                    p,ztrain=models[ident];g,q,r,cache,ii,jj,setup=ops[ident];B=cache['B'];target=fourier[ii,jj];dist=np.sum((cache['predictions']-target)**2,axis=1);nearest=int(np.argmin(dist));assert row['checkpoint_sha256']==meta[ident]['sha256'] and row['operator_sha256']==setup['operator_sha256'] and row['bank_rank_valid']==setup['rank_valid']
                    assert len(row['starts'])==1 and row['selected_start']==0 and row['selected_training_code_index']==nearest;np.testing.assert_array_equal(row['initial_latent'],ztrain[nearest]);latent=np.asarray(row['latent']);initial_z=np.asarray(row['initial_latent']);res=B@head(p,latent)-target;initial_res=B@head(p,initial_z)-target;J=B@head_jac(p,latent).T;gradient=np.linalg.norm(J.T@res)/(np.linalg.norm(J)*np.linalg.norm(res)+1e-300)
                    residual_errors.extend([abs(np.linalg.norm(res)-row['residual'])/(np.linalg.norm(initial_res)+1e-300),abs(np.linalg.norm(initial_res)-row['initial_residual'])/(np.linalg.norm(initial_res)+1e-300)]);assert max(residual_errors[-2:])<1e-10;gradient_errors.append(abs(gradient-row['stationarity']));assert gradient_errors[-1]<1e-8
                    assert row['stationary']==(gradient<=cfg['stationarity_tolerance']);assert row['solver_valid']==bool(row['finite'] and row['all_start_linear_valid'] and (row['stationary'] or row['reason']==2) and row['bank_rank_valid'])
                    assert row['all_start_linear_valid']==(row['max_linear_backward_error']<=cfg['linear_backward_error_limit']);assert row['reason'] in [0,1,3,5,6]
                    if row['reason']==0:assert row['attempts']==cfg['online_preset']['budget']
                    if row['reason']==6:assert row['stationarity']<=cfg['online_preset']['stationarity_stop']*(1+1e-7)+1e-12
                    assert row['jacobians']==row['accepted']+1 and 0<=row['accepted']<=row['attempts']<=cfg['online_preset']['budget'];assert row['total_attempts']==row['attempts'] and row['total_jacobians']==row['jacobians']
                    for key,value in row['starts'][0].items():assert row[key]==value
                    audited_starts+=1
                    if row['field_sha256'] not in decoded:
                        full=np.pad((g@head(p,latent)).reshape(n-1,n-1),1);decode_errors.append(relative(full,field));assert decode_errors[-1]<1e-9;decoded.add(row['field_sha256'])
                    if ident=='original_relative' and row['repetition']==0:
                        oldrow=next(r for r in baseline_previous['rows'] if r['intervals']==n and r['case']==case and r['method']=='original_relative' and r['repetition']==0);oldfield=np.load(run.parent/'staged_accuracy08/cluster/out/pilot/fields'/(oldrow['field_sha256']+'.npz'))['field'];baseline_parity.append(relative(field,oldfield));assert baseline_parity[-1]<1e-8
                elif ident.startswith('cg_'):
                    v=field[1:-1,1:-1];lap=n*n*(4*v-field[:-2,1:-1]-field[2:,1:-1]-field[1:-1,:-2]-field[1:-1,2:]);rr=relative(lap,f[1:-1,1:-1]);cg_errors.append(abs(rr-row['true_relative_residual']));assert cg_errors[-1]<1e-9;assert row['cg_converged']==(row['true_relative_residual']<=row['cg_tolerance']*(1+1e-6)+1e-12) and row['solver_valid']==row['cg_converged']
                else:assert relative(field,same)<1e-11
            for ident in cfg['trained_model_ids']:
                g,q,r,cache,ii,jj,setup=ops[ident];p,ztrain=models[ident];u=same[1:-1,1:-1].ravel();projection=np.pad((q@(q.T@u)).reshape(n-1,n-1),1);o=next(x for x in d['oracles'] if x['model']==ident and x['intervals']==n and x['case']==case);field=field_load(o['field_sha256']);projection_errors.append(relative(projection,field));assert projection_errors[-1]<1e-9;assert abs(relative(field,same)-o['same_grid_relative_error'])<1e-10 and abs(relative(field,fine)-o['physical_error'])<1e-10 and o['rank_valid']==setup['rank_valid']
                if n==cfg['head_oracle']['intervals']:
                    h=next(x for x in d['head_oracles'] if x['model']==ident and x['intervals']==n and x['case']==case);target=q.T@u;perp=np.linalg.norm(u-q@target);pred=head(p,ztrain)@r.T;nearest=int(np.argmin(np.sum((pred-target)**2,axis=1)));assert h['nearest_full_field_training_code_index']==nearest
                    for i,candidate in enumerate(h['candidates']):
                        audited_oracle_candidates+=1;latent=np.asarray(candidate['latent']);initial_latent=ztrain.mean(0) if i==0 else ztrain[nearest];np.testing.assert_allclose(candidate['initial_latent'],initial_latent,atol=1e-15,rtol=1e-14);field=field_load(candidate['field_sha256']);full=np.pad((g@head(p,latent)).reshape(n-1,n-1),1);oracle_errors.append(relative(full,field));assert oracle_errors[-1]<1e-9
                        residual=np.linalg.norm(full-same);initial_residual=np.linalg.norm(g@head(p,initial_latent)-u);rtest=np.concatenate((r@head(p,latent)-target,[perp]));J=np.vstack((r@head_jac(p,latent).T,np.zeros((1,latent.size))));gradient=np.linalg.norm(J.T@rtest)/(np.linalg.norm(J)*np.linalg.norm(rtest)+1e-300)
                        assert abs(residual-candidate['residual'])/(initial_residual+1e-300)<1e-10 and abs(initial_residual-candidate['initial_residual'])/(initial_residual+1e-300)<1e-10;assert abs(gradient-candidate['stationarity'])<1e-8;assert candidate['stationary']==(gradient<=cfg['head_oracle']['stationarity_tolerance'])
                        assert candidate['solver_valid']==bool(candidate['stationary'] and candidate['max_linear_backward_error']<=cfg['linear_backward_error_limit'] and setup['rank_valid']);assert abs(relative(field,same)-candidate['same_grid_relative_error'])<1e-10 and abs(relative(field,fine)-candidate['physical_error'])<1e-10
                        if candidate['reason']==0:assert candidate['attempts']==cfg['head_oracle']['budget']
                        if candidate['reason']==6:assert candidate['stationarity']<=cfg['head_oracle']['stationarity_tolerance']*(1+1e-7)+1e-12
                    valid=[i for i,c in enumerate(h['candidates']) if c['solver_valid']];best=min(valid,key=lambda i:h['candidates'][i]['same_grid_relative_error']) if valid else None;assert h['best_found_stationary_index']==best
            print('audit case',n,case,flush=True)
        del ops
    assert len(all_fields)==len(list((out/'fields').glob('*.npz')))
    report=dict(passed=True,source_commit=sub['source_commit'],job_id=sub['job_id'],archive_sha256=clean['archive_sha256'],result_sha256=file_sha(out/'result.json'),audit_script_sha256=file_sha(Path(__file__)),
        timed_invocations=len(d['rows']),distinct_full_fields=len(all_fields),online_start_records=audited_starts,head_oracle_candidates=audited_oracle_candidates,training_checkpoints=len(models),optimizer_blocks_verified=timing_blocks,
        maximum_metric_disagreement=max(metric),maximum_reference_disagreement=max(reference_errors),maximum_cpu_bank_operator_relative=max(operator_errors),maximum_cpu_cache_prediction_relative=max(cache_errors),maximum_decoder_field_relative=max(decode_errors),maximum_initial_scaled_residual_difference=max(residual_errors),maximum_stationarity_absolute_difference=max(gradient_errors),maximum_true_cg_residual_absolute_difference=max(cg_errors),maximum_bank_projection_field_relative=max(projection_errors),maximum_head_oracle_decoder_relative=max(oracle_errors),maximum_previous_baseline_field_relative_difference=max(baseline_parity),maximum_training_reconstruction_metric_difference=max(training_metrics),function_preserving_widening_verified=training_check['function_preserving_widening_verified'],training_representation=training_check['training_representation'],
        all_training_freezes_verified=True,all_optimizer_time_blocks_and_matching_verified=True,every_phase_independently_time_matched=True,
        all_fields_hashed_and_independently_scored=True,all_source_checkpoint_hashes_match_git=True,all_timing_components_sum=True,all_expected_invocations_present=True,all_rank_checks_pass=True,all_head_oracle_selection_gates_verified=True,remote_deleted=True,
        invalid_online_invocations=sum(not x['solver_valid'] for x in d['rows']),head_cases_without_stationary_fit=sum(x['best_found_stationary_index'] is None for x in d['head_oracles']),
        limitation='All42 development sources were opened before capacity selection; no fresh independent or sealed final confirmation. Nonlinear head fits are best-found stationary diagnostics, not global minima. Reference refinement is empirical.')
    (run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');(run/'result.json').write_bytes((out/'result.json').read_bytes());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
