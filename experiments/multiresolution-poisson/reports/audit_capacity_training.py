"""Independent audit of both trained ranks, widening and phase-wise effort match."""
from audit_iterative import *
from audit_staged_accuracy import leaves,weights_sha,same_tree


def features_at(p,coords):
    angle=2*np.pi*(coords@p['B']);ff=np.concatenate((np.sin(angle),np.cos(angle)),axis=1);bc=16*coords[:,0]*(1-coords[:,0])*coords[:,1]*(1-coords[:,1]);return p['out_scale']*bc[:,None]*mlp(p['g'],ff)


def audit_training(d,out,models):
    cfg=d['config'];train=np.asarray(d['training']['parameters']);n=cfg['training_nodes']-1;U=np.array([truth(n,param)[1:-1,1:-1].ravel() for param in train]);denom=np.sum(U*U,axis=1);meta={x['id']:x for x in d['checkpoints']};metrics=[];blocks_verified=0
    orig,oz=models['original_relative'];wide,wz=models['r128_initial'];same_tree(oz,wz)
    for key in ['B','out_scale']:same_tree(orig[key],wide[key])
    for key in ['g','h']:same_tree(orig[key][:-1],wide[key][:-1])
    for a,b in zip(orig['g'][-1],wide['g'][-1]):np.testing.assert_array_equal(a,b[...,:64])
    for a,b in zip(orig['h'][-1],wide['h'][-1]):np.testing.assert_array_equal(a,b[...,:64]);assert not np.count_nonzero(b[...,64:])
    np.testing.assert_array_equal(orig['h_lin'],wide['h_lin'][:,:64]);assert not np.count_nonzero(wide['h_lin'][:,64:])
    wi=d['training']['widening'];complement=np.asarray(wi['complement']);scales=np.asarray(wi['training_grid_new_column_scales']);affine=np.vstack(orig['g'][-1]);assert complement.shape==(129,64) and scales.shape==(64,)
    assert np.linalg.norm(complement.T@complement-np.eye(64))<1e-12 and np.linalg.norm(affine.T@complement)/(np.linalg.norm(affine)+1e-300)<1e-12
    np.testing.assert_allclose(np.vstack(wide['g'][-1])[:,64:],complement*scales,atol=1e-14,rtol=1e-14)
    Gwide=bank(wide,n);norms=np.linalg.norm(Gwide,axis=0);assert np.max(np.abs(norms[64:]/np.median(norms[:64])-1))<1e-10
    par=dict(np.load(out/'widening_parity.npz'));rng=np.random.default_rng(cfg['widening']['parity_seed']);coords=rng.uniform(0.,1.,(cfg['widening']['parity_coordinates'],2));codes=np.concatenate((oz[:16],rng.normal(size=(cfg['widening']['parity_random_codes'],oz.shape[1]))));np.testing.assert_array_equal(coords,par['coordinates']);np.testing.assert_array_equal(codes,par['codes'])
    for p,suffix in [(orig,'before'),(wide,'after')]:
        G=features_at(p,coords);field=head(p,codes)@G.T;jac=np.array([G@head_jac(p,z).T for z in codes]);assert relative(field,par[suffix])<1e-12 and relative(jac,par['jacobian_'+suffix])<1e-12
    assert relative(par['after'],par['before'])<=cfg['widening']['relative_parity_tolerance'] and relative(par['jacobian_after'],par['jacobian_before'])<=cfg['widening']['relative_parity_tolerance'] and max(wi['parity'].values())<=cfg['widening']['relative_parity_tolerance']
    phases={x['tag']:x for x in d['training']['phases']};assert list(phases)==[a+'_'+phase['phase'] for a in ['r128','r64'] for phase in cfg['training_phases']]
    for ident,(p,z) in models.items():
        same_tree(p['out_scale'],orig['out_scale']);G=bank(p,n);err=np.linalg.norm(head(p,z)@G.T-U,axis=1)/np.sqrt(denom);info=meta[ident]['training_metrics'];diff=float(np.max(np.abs(err-info['per_snapshot_relative_l2'])));metrics.append(diff);assert diff<1e-10;assert abs(max(err)-info['worst_relative_l2'])<1e-10 and abs(np.median(err)-info['median_relative_l2'])<1e-10
    def metric_data(p,path,info):
        f=dict(np.load(out/path));G=bank(p,n);q,r=np.linalg.qr(G,mode='reduced');sv=np.linalg.svd(r,compute_uv=False);rank=np.count_nonzero(sv>sv[0]*max(G.shape)*np.finfo(float).eps);assert rank==G.shape[1] and info['rank_valid'] and info['rank']==rank
        target=U@q;perp=np.sum((U-target@q.T)**2,axis=1);optimal=np.linalg.solve(r,target.T).T;assert relative(optimal,f['optimal'])<1e-8 and relative(G.T@G,f['R'].T@f['R'])<1e-10;assert relative(sv,np.asarray(info['singular_values']))<1e-10
        assert relative(f['target'],U@G@np.linalg.inv(f['R']))<1e-8 and np.max(np.abs(perp-f['perpendicular']))/(max(denom)+1e-300)<1e-10
        assert abs(np.sqrt(max(perp/denom))-info['worst_projection_error'])<1e-10
        return f,G
    for arm,start_id in [('r128','r128_initial'),('r64','original_relative')]:
        start=models[start_id];initial,G=metric_data(start[0],arm+'_initial_coefficients.npz',d['training'][arm+'_initial_bank']);free,Gbank=metric_data(models[arm+'_bank'][0],arm+'_bank_free_coefficients.npz',phases[arm+'_bank']['bank_qr'])
        for key in ['h','h_lin','out_scale']:same_tree(models[arm+'_bank'][0][key],start[0][key])
        same_tree(models[arm+'_bank'][1],start[1])
        for key in ['B','g','out_scale']:same_tree(models[arm+'_head'][0][key],models[arm+'_bank'][0][key])
        free_errors=np.linalg.norm(free['learned']@Gbank.T-U,axis=1)/np.sqrt(denom);assert np.max(np.abs(free_errors-phases[arm+'_bank']['free_coefficient_relative_errors']))<1e-10
        hp,hz=models[arm+'_head'];full=np.sum((head(hp,hz)@Gbank.T-U)**2,axis=1);compressed=np.sum((head(hp,hz)@free['R'].T-free['target'])**2,axis=1)+free['perpendicular'];assert np.max(np.abs(full-compressed)/denom)<1e-10
        previous=start_id
        for spec in cfg['training_phases']:
            phase=spec['phase'];tag=arm+'_'+phase;r=phases[tag];options={**cfg['training_common'],**spec};assert r['finite'] and r['budget_reached'];assert r['initial_weights_sha256']==weights_sha(models[previous][0]) and r['final_weights_sha256']==weights_sha(models[tag][0])
            assert r['initial_latent_sha256']==array_sha(initial['optimal'] if phase=='bank' else models[previous][1]);assert r['final_latent_sha256']==array_sha(free['learned'] if phase=='bank' else models[tag][1])
            target=d['training']['matched_phase_optimizer_targets_seconds'][phase];assert target==phases['r128_'+phase]['optimizer_seconds'];elapsed=0.;done=0
            for block in r['optimizer_blocks']:
                assert block['first_update']==done+1 and block['last_update']-done==options['timing_block_updates'] and block['seconds']>0
                progress=done/options['steps'] if arm=='r128' else min(elapsed/target,1.);warm=options['warmup_fraction'];factor=progress/warm if progress<warm else options['final_learning_rate_fraction']+(1-options['final_learning_rate_fraction'])*.5*(1+np.cos(np.pi*(progress-warm)/(1-warm)));rate=options['learning_rate']*max(options['final_learning_rate_fraction'],factor);assert abs(rate-block['learning_rate'])<1e-15
                elapsed+=block['seconds'];done=block['last_update'];blocks_verified+=1
            assert abs(elapsed-r['optimizer_seconds'])<1e-12 and done==r['updates'] and r['sampled_source_exposures']==done*options['source_batch'];assert r['sampled_field_entry_exposures']==(None if phase=='head' else done*options['source_batch']*options['point_batch'])
            if arm=='r128':assert done==options['steps'] and r['matched_optimizer_seconds'] is None
            else:assert r['matched_optimizer_seconds']==target and elapsed-r['optimizer_blocks'][-1]['seconds']<target<=elapsed and abs(elapsed-target-r['control_optimizer_overshoot_seconds'])<1e-12
            previous=tag
    return dict(passed=True,optimizer_blocks_verified=blocks_verified,maximum_reconstruction_metric_difference=max(metrics),function_preserving_widening_verified=True,every_phase_independently_time_matched=True)
