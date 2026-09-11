"""Independent NumPy/SciPy audit of every final correction-family invocation."""
from audit_capacity_accuracy import *
from scipy.linalg import solve_triangular


def basis_provenance(run,out,models,d):
    cfg=d['config'];parent=run.parent/'capacity_accuracy09';previous=json.loads((parent/'result.json').read_text());assert file_sha(parent/'result.json')==cfg['basis_parent_result_sha256'];tc=previous['config'];basis=dict(np.load(out/'basis.npz'));p,z=models['r128_joint'];np.testing.assert_array_equal(basis['training_latents'],z)
    draws=sample(tc['original_draw_seed'],tc['original_draw_count'])[:tc['training_prefix_count']];np.testing.assert_allclose(draws,previous['training']['parameters'],atol=1e-15,rtol=5e-14)
    n=255;g=bank(p,n);R=basis['R'];gram=relative(R.T@R,g.T@g);assert gram<1e-11
    q=solve_triangular(R.T,g.T,lower=True).T;U=np.stack([truth(n,param)[1:-1,1:-1].ravel() for param in draws]);norms=np.linalg.norm(U,axis=1);np.testing.assert_allclose(norms,basis['training_norms'],rtol=1e-12)
    E=(U@q-head(p,z)@R.T)/norms[:,None];C=basis['coefficient_directions'];V=R@C;np.testing.assert_allclose(V,basis['physical_metric_directions'],atol=1e-12,rtol=1e-11)
    orth=float(np.linalg.norm(V.T@V-np.eye(32)));assert orth<1e-10
    singular=np.linalg.svd(E,compute_uv=False);np.testing.assert_allclose(singular,basis['singular_values'],atol=1e-11,rtol=1e-10)
    eigerr=relative(E.T@E@V,V*singular[:32]**2);assert eigerr<1e-9
    return dict(passed=True,training_snapshots=len(draws),training_intervals=n,training_parameters_from_seed=True,no_development_snapshots_used=True,saved_optimized_training_codes_verified=True,training_codes_independently_stationary=False,metric_gram_relative=gram,physical_basis_orthogonality=orth,leading_residual_singular_subspace_relative=eigerr,rank=32)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve();root=Path(__file__).resolve().parents[3];local=run/'cluster';out=local/'out/pilot'
    d=json.loads((out/'result.json').read_text());cfg=d['config'];sub=json.loads((run/'submission.json').read_text());clean=json.loads((run/'cleanup.json').read_text());assert d['complete'] and clean['remote_deleted_and_absence_checked']
    assert file_sha(run/'verified-cluster.tar.gz')==clean['archive_sha256'];assert d['provenance']['commit']==sub['source_commit'] and d['provenance']['job_id']==sub['job_id'];assert d['provenance']['backend']=='gpu' and d['provenance']['x64'] and d['provenance']['matmul_precision']=='highest'
    for manifest in ['PULL.sha256','MANIFEST.sha256','RESULTS.sha256']:subprocess.run(['sha256sum','-c',manifest,'--quiet'],cwd=local,check=True)
    mapping={'sep_common.py':'experiments/separable-decoder/sep_common.py','ctol_tol.py':'experiments/cost-to-tolerance/ctol_tol.py','ms_parametric.py':'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py'}
    for staged,expected in sub['source_hashes'].items():
        payload=(local/staged).read_bytes();name=Path(staged).name;assert sha(payload)==expected;assert payload==subprocess.check_output(['git','show',sub['source_commit']+':'+mapping.get(name,'experiments/multiresolution-poisson/'+name)],cwd=root)
    for staged,info in sub['inputs'].items():
        payload=(local/staged).read_bytes();assert sha(payload)==info['sha256'];assert payload==subprocess.check_output(['git','show',sub['source_commit']+':'+info['source_path']],cwd=root)
    assert (out/'basis.npz').read_bytes()==(local/'in/basis.npz').read_bytes();assert file_sha(out/'basis.npz')==cfg['basis_sha256'];basis=dict(np.load(out/'basis.npz'))
    log='\n'.join(p.read_text() for p in (local/'logs').glob('*'));assert 'jax_backend=gpu' in log and log.count('CORRECTION ACCURACY COMPLETE')==2 and (local/'EXIT_CODE').read_text().strip()=='0'
    for bad in ['cuInit','captured constant','large constant','OUT_OF_MEMORY','No space left','Traceback','Resource exhausted']:assert bad not in log,bad
    draws=np.asarray(d['cohort']['parameters']);assert array_sha(draws)==d['cohort']['parameter_sha256'];old=np.concatenate((sample(cfg['existing_development_seed'],cfg['existing_development_count']),sample(cfg['fresh_development_seed'],cfg['fresh_development_count'])));later=sample(cfg['new_development_seed'],cfg['new_development_count']);np.testing.assert_allclose(draws,np.concatenate((old,later)),atol=1e-15,rtol=5e-14);assert array_sha(draws[:len(old)])==cfg['parameter_sha256']
    previous=json.loads((run.parent/'capacity_accuracy09/result.json').read_text());np.testing.assert_array_equal(draws,previous['cohort']['parameters']);assert d['cohort']['groups']==['existing_development']*len(old)+['new_development']*len(later)
    models={};meta={x['id']:x for x in d['checkpoints']}
    for ident,info in meta.items():
        payload=(out/info['path']).read_bytes();assert sha(payload)==info['sha256'];assert payload==(local/'in'/('original.pkl' if ident=='original_relative' else 'head.pkl')).read_bytes();model=pickle.loads(payload);p=model['params'];z=model['Z_tr'];assert array_sha(z)==info['codes_sha256'];assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in leaves((p,z)));rank=64 if ident=='original_relative' else 128;assert z.shape==(512,16) and p['h_lin'].shape==(16,rank) and model['cfg']['r']==rank;models[ident]=(p,z)
    print('audit training-only correction provenance',flush=True);basis_check=basis_provenance(run,out,models,d)
    errors=defaultdict(list)
    def check(name,value,limit):errors[name].append(float(value));assert np.isfinite(value) and value<limit,(name,value,limit)
    groups=defaultdict(list);keys=set();methods=cfg['trained_model_ids']+[f'cg_{t:.0e}' for t in cfg['cg_tolerances']]+['dst'];all_fields=set();starts=0
    for row in d['rows']:
        key=(row['intervals'],row['case'],row['method'],row['repetition']);assert key not in keys;keys.add(key);groups[row['intervals'],row['case']].append(row);assert row['group']==d['cohort']['groups'][row['case']];assert abs(row['total_seconds']-sum(row[k] for k in ['input_seconds','fused_device_seconds','output_seconds']))<1e-10;assert all(row[k]>0 for k in ['total_seconds','input_seconds','fused_device_seconds','output_seconds'])
    assert keys=={(n,c,m,r) for n in cfg['intervals'] for c in range(len(draws)) for m in methods for r in range(cfg['repetitions'])};assert len(d['oracles'])==len(cfg['intervals'])*len(draws)*len(models)
    previous_rows={(r['intervals'],r['case'],r['method']):r for r in previous['rows'] if r['repetition']==0}
    for n in cfg['intervals']:
        print('audit banks/operators',n,flush=True);banks={};ops={}
        for base,(p,z) in models.items():
            g=bank(p,n);q,r=np.linalg.qr(g,mode='reduced');sv=np.linalg.svd(r,compute_uv=False);rank=int((sv>sv[0]*max(g.shape)*np.finfo(float).eps).sum());banks[base]=(g,q,r,sv,rank)
        for ident in cfg['trained_model_ids']:
            base=cfg['method_base_model'][ident];p,z=models[base];g,q,r,sv,rank=banks[base];setup=next(s for s in d['setup'] if s['method']==ident and s['intervals']==n);assert setup['bank_rank']['rank']==rank and setup['bank_rank']['rank_valid']==(rank==g.shape[1]) and setup['bank_rank']['rank_valid'];cache=dict(np.load(out/f'cache_n{n}_{ident}.npz'));B,Bp,C,Q,R=[cache[k] for k in ['B','projected_B','C','Q','R']];count=cfg['correction_counts'][ident];expected_C=basis['coefficient_directions'][:,:count] if count else np.zeros((g.shape[1],0));np.testing.assert_array_equal(C,expected_C);np.testing.assert_array_equal(cache['codes'],z)
            for key,arr in [('operator_sha256',B),('projected_operator_sha256',Bp),('correction_matrix_sha256',C),('weak_correction_sha256',B@C),('Q_sha256',Q),('R_sha256',R)]:
                if key!='weak_correction_sha256':assert array_sha(arr)==setup[key]
            assert array_sha(cache['bank_R'])==setup['bank_rank']['R_sha256'];check('bank_gram',relative(cache['bank_R'].T@cache['bank_R'],g.T@g),1e-10)
            ii,jj=(np.asarray(setup['mode_indices'])-1).T;lam=eigenvalues(n);expected_ij=np.nonzero(lam<=np.sort(lam.ravel())[cfg['requested_modes']-1]);np.testing.assert_array_equal(ii,expected_ij[0]);np.testing.assert_array_equal(jj,expected_ij[1]);assert setup['retained_modes']==len(ii)>4*(16+count)
            maxmode=int(max(ii.max(),jj.max()))+1;S=np.sqrt(2/n)*np.sin(np.pi*np.outer(np.arange(1,n),np.arange(1,maxmode+1))/n)
            # Independent full-bank contraction once per shared checkpoint.
            if ident in ['original_relative','r128_q0']:
                cpuB=np.einsum('xa,xyr,yb->abr',S,g.reshape(n-1,n-1,-1),S,optimize=True)[ii,jj];check('cpu_bank_operator',relative(cpuB,B),1e-10)
            else:np.testing.assert_array_equal(B,ops['r128_q0'][0]['B'])
            L=B@C
            if count:
                check('linear_Q_orthogonality',np.linalg.norm(Q.T@Q-np.eye(count)),1e-10);check('linear_QR',relative(Q@R,L),1e-10);values=np.linalg.svd(L,compute_uv=False);lr=int((values>values[0]*max(L.shape)*np.finfo(float).eps).sum());np.testing.assert_allclose(values,setup['linear_singular_values'],atol=1e-12,rtol=1e-10)
            else:lr=0
            assert setup['linear_rank']==lr==count and setup['linear_rank_valid'];check('projected_operator',relative(B-Q@(Q.T@B),Bp),1e-10);check('cache_prediction',relative(head(p,z)@Bp.T,cache['predictions']),1e-10);ops[ident]=(cache,ii,jj,setup)
        for case,param in enumerate(draws):
            refs=dict(np.load(out/'references'/f'n{n}_case{case}.npz'))
            for nr,key in zip(cfg['reference_intervals'],['coarse','fine']):check('reference',relative(truth(nr,param)[::nr//n,::nr//n],refs[key]),1e-12)
            fine=refs['fine'];same=truth(n,param);f=source(n,param);delta=relative(refs['coarse'],fine);fourier=dstn(f[1:-1,1:-1],type=1,norm='ortho')/eigenvalues(n);cachefields={};decoded=set()
            def field_load(digest):
                all_fields.add(digest)
                if digest not in cachefields:
                    field=np.load(out/'fields'/(digest+'.npz'))['field'];assert array_sha(field)==digest and field.shape==(n+1,n+1) and field.dtype==np.float64 and np.isfinite(field).all();assert not np.count_nonzero(field[[0,-1]]) and not np.count_nonzero(field[:,[0,-1]]);cachefields[digest]=field
                return cachefields[digest]
            assert len({r['source_sha256'] for r in groups[n,case]})==1;assert groups[n,case][0]['source_sha256']==previous_rows[n,case,'dst']['source_sha256']
            for row in groups[n,case]:
                field=field_load(row['field_sha256']);error=relative(field,fine);stride=n//cfg['observation_intervals'];values=dict(physical_error=error,same_grid_error=relative(field,same),reference_delta=delta,conservative_physical_error=(error+delta)/(1-delta),common_observation_error=relative(field[::stride,::stride],fine[::stride,::stride]));check('physical_metrics',max(abs(row[k]-v) for k,v in values.items()),1e-10);assert row['finite'];ident=row['method']
                if ident in ops:
                    base=cfg['method_base_model'][ident];p,ztrain=models[base];g,q,r,sv,bankrank=banks[base];cache,ii,jj,setup=ops[ident];B,Bp,C,Q,R=[cache[k] for k in ['B','projected_B','C','Q','R']];count=C.shape[1];target=fourier[ii,jj];projected_target=target-Q@(Q.T@target);dist=np.sum((cache['predictions']-projected_target)**2,axis=1);nearest=int(np.argmin(dist));assert row['checkpoint_sha256']==meta[base]['sha256'] and row['operator_sha256']==setup['operator_sha256'] and row['projected_operator_sha256']==setup['projected_operator_sha256'] and row['correction_matrix_sha256']==setup['correction_matrix_sha256'] and row['bank_rank_valid'];assert len(row['starts'])==1 and row['selected_start']==0 and row['selected_training_code_index']==nearest;np.testing.assert_array_equal(row['initial_latent'],ztrain[nearest]);z=np.asarray(row['latent']);z0=np.asarray(row['initial_latent']);h=head(p,z);h0=head(p,z0)
                    rhs=Q.T@(target-B@h);y=solve_triangular(R,rhs) if count else np.zeros(0);y0=solve_triangular(R,Q.T@(target-B@h0)) if count else y;np.testing.assert_allclose(y,row['correction_coefficients'],atol=1e-10,rtol=1e-9);np.testing.assert_allclose(y0,row['initial_correction_coefficients'],atol=1e-10,rtol=1e-9);np.testing.assert_array_equal(row['augmented_latent'],np.concatenate((row['latent'],row['correction_coefficients'])));assert row['nominal_latent_dimension']==16+count and row['nonlinear_optimizer_dimension']==16 and row['correction_count']==count
                    res=B@(h+C@y)-target;rp=Bp@h-projected_target;initial=B@(h0+C@y0)-target;initialp=Bp@h0-projected_target;scale=np.linalg.norm(initialp)+1e-300
                    for key,value in [('residual',np.linalg.norm(rp)),('initial_residual',scale),('full_residual',np.linalg.norm(res)),('initial_full_residual',np.linalg.norm(initial))]:check('initial_scaled_residual',abs(row[key]-value)/scale,1e-10)
                    J=B@head_jac(p,z).T;Jp=Bp@head_jac(p,z).T;L=B@C;full_gradient=np.linalg.norm(np.concatenate((J.T@res,L.T@res)))/(np.sqrt(np.linalg.norm(J)**2+np.linalg.norm(L)**2)*np.linalg.norm(res)+1e-300);reduced_gradient=np.linalg.norm(Jp.T@rp)/(np.linalg.norm(Jp)*np.linalg.norm(rp)+1e-300);check('full_stationarity',abs(full_gradient-row['stationarity']),1e-8);check('reduced_stationarity',abs(reduced_gradient-row['reduced_stationarity']),1e-8)
                    assert row['stationary']==(full_gradient<=cfg['stationarity_tolerance']) and row['reduced_stationary']==(reduced_gradient<=cfg['stationarity_tolerance']);jsv=np.linalg.svd(Jp,compute_uv=False);jrank=int((jsv>jsv[0]*max(Jp.shape)*np.finfo(float).eps).sum());np.testing.assert_allclose(jsv,row['projected_jacobian_singular_values'],atol=1e-11,rtol=1e-10);assert row['projected_jacobian_rank']==jrank and row['projected_jacobian_rank_valid']==(jrank==16)
                    recovery=np.linalg.norm(R@y-rhs)/(np.linalg.norm(R)*np.linalg.norm(y)+np.linalg.norm(rhs)+1e-300) if count else 0.;reconstruction=np.linalg.norm(res-rp)/(np.linalg.norm(target)+np.linalg.norm(B@h)+1e-300);check('recovery_backward_difference',abs(recovery-row['linear_recovery_backward_error']),1e-12);check('residual_reconstruction',abs(reconstruction-row['residual_reconstruction_scaled']),1e-12)
                    assert row['all_start_linear_valid']==bool(row['max_linear_backward_error']<=cfg['linear_backward_error_limit'] and row['linear_recovery_backward_error']<=cfg['linear_backward_error_limit']);valid=bool(row['finite'] and row['stationary'] and row['reduced_stationary'] and row['all_start_linear_valid'] and row['projected_jacobian_rank_valid'] and row['residual_reconstruction_scaled']<=cfg['residual_reconstruction_limit'] and row['bank_rank_valid']);assert row['solver_valid']==valid;assert row['reason'] in [0,1,3,5,6]
                    if row['reason']==0:assert row['attempts']==cfg['online_preset']['budget']
                    if row['reason']==6:assert row['reduced_stationarity']<=cfg['online_preset']['stationarity_stop']*(1+1e-7)+1e-12
                    assert row['jacobians']==row['accepted']+1 and 0<=row['accepted']<=row['attempts']<=cfg['online_preset']['budget'];assert row['total_attempts']==row['attempts'] and row['total_jacobians']==row['jacobians'];assert row['absolute_tau_threshold']==0
                    for key,value in row['starts'][0].items():assert row[key]==value
                    starts+=1
                    if row['field_sha256'] not in decoded:check('decoder_field',relative(np.pad((g@(h+C@y)).reshape(n-1,n-1),1),field),1e-9);decoded.add(row['field_sha256'])
                    if count==0 and row['repetition']==0:
                        oldrow=previous_rows[n,case,base];oldfield=np.load(run.parent/'capacity_accuracy09/cluster/out/pilot/fields'/(oldrow['field_sha256']+'.npz'))['field'];check('previous_q0_field_parity',relative(field,oldfield),1e-8);assert row['selected_training_code_index']==oldrow['selected_training_code_index']
                else:
                    v=field[1:-1,1:-1];lap=n*n*(4*v-field[:-2,1:-1]-field[2:,1:-1]-field[1:-1,:-2]-field[1:-1,2:]);rr=relative(lap,f[1:-1,1:-1])
                    if ident.startswith('cg_'):
                        check('true_cg_residual',abs(rr-row['true_relative_residual']),1e-9);assert row['cg_converged']==(row['true_relative_residual']<=row['cg_tolerance']*(1+1e-6)+1e-12) and row['solver_valid']==row['cg_converged']
                    else:check('dst_source_operator',rr,1e-9);check('dst_field',relative(field,same),1e-11)
            for base in models:
                g,q,r,sv,rank=banks[base];u=same[1:-1,1:-1].ravel();projection=np.pad((q@(q.T@u)).reshape(n-1,n-1),1);o=next(x for x in d['oracles'] if x['base_model']==base and x['intervals']==n and x['case']==case);field=field_load(o['field_sha256']);check('bank_projection_field',relative(projection,field),1e-9);check('bank_projection_metrics',max(abs(relative(field,same)-o['same_grid_relative_error']),abs(relative(field,fine)-o['physical_error'])),1e-10);assert o['rank_valid']
            print('audit case',n,case,flush=True)
        del banks,ops
    assert len(all_fields)==len(list((out/'fields').glob('*.npz')))
    report=dict(passed=True,source_commit=sub['source_commit'],job_id=sub['job_id'],archive_sha256=clean['archive_sha256'],result_sha256=file_sha(out/'result.json'),audit_script_sha256=file_sha(Path(__file__)),timed_invocations=len(d['rows']),distinct_full_fields=len(all_fields),online_start_records=starts,checkpoints=len(models),bank_projection_fields=len(d['oracles']),training_only_basis=basis_check,maximum_errors={key:max(values) for key,values in errors.items()},all_fields_hashed_and_independently_scored=True,all_source_checkpoint_basis_hashes_match_git=True,all_timing_components_sum=True,all_expected_invocations_present=True,all_full_and_reduced_gradient_gates_verified=True,all_projected_ranks_and_linear_eliminations_verified=True,all_source_only_initializers_verified=True,remote_deleted=True,invalid_online_invocations=sum(not x['solver_valid'] for x in d['rows']),limitation='All42 cases already-opened development. Added linear capacity and analytic elimination are a combined change. Training codes are saved optimized variables, not newly certified stationary fits. No final-case or minimax claim.')
    (run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');(run/'result.json').write_bytes((out/'result.json').read_bytes());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
