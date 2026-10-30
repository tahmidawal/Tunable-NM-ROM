"""Prospective full-TRAIN fixed-hidden output polish; no executable packet or parent defaults."""
from pathlib import Path
import hashlib,json,time,resource,sys
import numpy as np
from accuracy_v2.spatial_bank import SpatialBank,evaluate_layers
from accuracy_v2.independent_residual_heads import evaluate_heads
from accuracy_v2.fixed_hidden_output_polish import output_features,volume_rows,boundary_rows,qr_append,solve_factor
from accuracy_v2.residual_head_bank import ResidualHeadBank
from accuracy_v2.device_blocks import prepare as prepare_device_blocks,theta_once
from accuracy_v2.train import digest
from accuracy_v2.divergence_objective import block_objective,validate_weight
from accuracy_v2.boundary_traction_objective import block_traction_objective

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def _refine(x,w,target,gradient,teacher_identity,boundary,boundary_record,coefficient_record,parent_bank_path,parent_receipt_path,parent_review_path,warm_model_path,warm_receipt_path,warm_review_path,protocol,directory,deadline):
    out=Path(directory);out.mkdir(exist_ok=False)
    parent=SpatialBank.load(parent_bank_path);receipt=json.loads(Path(parent_receipt_path).read_text());review=json.loads(Path(parent_review_path).read_text())
    if parent.metadata.get('objective_variant')!='supervised_trace_and_boundary_traction_lambda_1':raise ValueError('reviewed traction-objective parent required')
    if coefficient_record.get('passed') is not True or coefficient_record['teacher_identity']!=teacher_identity or boundary_record['volume_teacher_identity']!=teacher_identity:raise ValueError('same actual TRAIN coefficient and boundary teacher closure required')
    if any(digest(a)!=boundary_record['arrays'][k] for k,a in boundary.items()):raise ValueError('boundary teacher bytes changed')
    traction_weight=validate_weight(protocol['traction_weight'])
    bx=boundary['points_physical'];bw=boundary['weights_dimensionless'];bn=boundary['normals_body'];bg=boundary['teacher_gradient_dimensionless'];bs=boundary['singular_values'];nt=len(bw)
    denominator=float(boundary_record['normalization']['global_training_traction_energy'])
    if not np.isfinite(denominator) or denominator<=0:raise ValueError('positive global traction denominator required')
    if parent.rank!=4 or not np.array_equal(parent.transform,np.eye(4)) or receipt['bank_sha256']!=sha(parent_bank_path) or review.get('passed') is not True or review.get('model_sha256')!=sha(parent_bank_path) or review.get('model_receipt_sha256')!=sha(parent_receipt_path):raise ValueError('reviewed unrotated rank4 parent required')
    if parent.metadata['teacher_identity']!=teacher_identity or teacher_identity['split']!='training' or teacher_identity['Re']!=[15+1.25*j for j in range(9)]:raise ValueError('same genuine training teacher required')
    fixed={'method':'fixed_hidden_output_delta_qr_svd','qr_block_points':256,'new_adam_steps':0,'new_lbfgs_steps':0,'supervised_divergence_weight':1.0,'traction_weight':1.0,'head_seed':30102026,'cutoff_policy':'float64_epsilon_times_max_rows_columns'}
    if set(protocol)!=set(fixed)|{'relative_svd_cutoff'} or any(type(protocol[k]) is not type(v) or protocol[k]!=v for k,v in fixed.items()):raise ValueError('exact prospective polish protocol required')
    divergence_weight=validate_weight(protocol['supervised_divergence_weight'])
    x,w,target,gradient=map(lambda a:np.asarray(a,dtype=np.float64),(x,w,target,gradient));n=len(x)
    if x.shape!=(n,2) or w.shape!=(n,) or target.shape!=(n,2,4) or gradient.shape!=(n,2,2,4) or np.min(w)<=0 or any(not np.isfinite(a).all() for a in (x,w,target,gradient)):raise ValueError('finite same-layout teacher required')
    total_rows=7*n+2*nt
    required_cutoff=float(np.finfo(np.float64).eps*max(total_rows,130))
    if type(protocol['relative_svd_cutoff']) is not float or protocol['relative_svd_cutoff']!=required_cutoff:raise ValueError('explicit predeclared dimension/backward-error cutoff required')
    ids={k:digest(v) for k,v in zip(('points','weights','values','gradients'),(x,w,target,gradient))}
    if ids!=teacher_identity['arrays']:raise ValueError('actual teacher bytes changed')
    nv=np.einsum('n,ncr,ncr->r',w,target,target);ng=np.einsum('n,ncdr,ncdr->r',w,gradient,gradient)
    sigma=np.asarray(teacher_identity['singular_values']);importance=np.sqrt(sigma/sigma[0]);importance/=importance.sum()
    for observed,key in ((nv,'value_norms_squared'),(ng,'gradient_norms_squared'),(importance,'mode_weights')):
        if not np.allclose(observed,np.asarray(parent.metadata[key]),rtol=1e-12,atol=0.):raise ValueError('objective normalization replay failed')
    # Reuse the exact frozen norms/weights; reconstruction above only checks them.
    nv=np.asarray(parent.metadata['value_norms_squared'],dtype=np.float64);ng=np.asarray(parent.metadata['gradient_norms_squared'],dtype=np.float64);importance=np.asarray(parent.metadata['mode_weights'],dtype=np.float64)
    config=parent.metadata['config'];prior_adam=int(parent.metadata['actual_applied_adam_steps']);prior_lb=int(parent.metadata['accepted_lbfgs_steps'])
    if review.get('adam_steps_recorded')!=prior_adam or review.get('accepted_lbfgs_steps')!=prior_lb or review.get('selected_theta_sha256')!=receipt['selected_theta_sha256'] or type(review.get('allocated_seconds')) is not int:raise ValueError('reviewed cumulative parent training provenance required')
    warm=ResidualHeadBank.load(warm_model_path)
    warm_receipt=json.loads(Path(warm_receipt_path).read_text());warm_review=json.loads(Path(warm_review_path).read_text())
    if warm.parent_sha256!=sha(parent_bank_path) or warm.parent_bytes!=Path(parent_bank_path).read_bytes():raise ValueError('warm child must embed the exact immutable traction parent')
    if warm_receipt['bank_sha256']!=sha(warm_model_path) or warm_review.get('passed') is not True or warm_review.get('model_sha256')!=sha(warm_model_path) or warm_review.get('model_receipt_sha256')!=sha(warm_receipt_path):raise ValueError('genuine reviewed warm child required')
    if warm_receipt.get('selected_theta_scope')!='child parameters only' or warm_receipt.get('parent_model_sha256')!=sha(parent_bank_path) or warm_review.get('parent_model_sha256')!=sha(parent_bank_path):raise ValueError('explicit reviewed composite child/embedded-parent scope required')
    if warm.metadata['objective_variant'] not in {'independent_residual_heads_same_trace_and_traction','fixed_coordinate_residual_heads_same_trace_and_traction'}:raise ValueError('same child family and objective required')
    if warm.metadata['teacher_identity']!=teacher_identity:raise ValueError('warm teacher identity differs')
    for key in ('value_norms_squared','gradient_norms_squared','mode_weights'):
        if not np.array_equal(np.asarray(warm.metadata[key]),np.asarray(parent.metadata[key])):raise ValueError('warm frozen normalization differs')
    if warm.metadata['refinement']['boundary_teacher']['arrays']!=boundary_record['arrays'] or warm.metadata['refinement']['boundary_teacher']['normalization']['global_training_traction_energy']!=denominator:raise ValueError('warm boundary objective differs')
    warm_theta=np.concatenate([a.ravel() for h in warm.heads for layer in h for a in layer])
    if digest(warm_theta)!=warm_receipt['selected_theta_sha256'] or warm_review.get('selected_theta_sha256')!=digest(warm_theta):raise ValueError('reviewed child-only theta differs')
    prior_adam=int(warm.metadata['actual_applied_adam_steps']);prior_lb=int(warm.metadata['accepted_lbfgs_steps'])
    if warm_review.get('adam_steps_recorded')!=prior_adam or warm_review.get('accepted_lbfgs_steps')!=prior_lb or type(warm_review.get('allocated_seconds')) is not int:raise ValueError('warm child cumulative lineage differs')
    bindings={'schema':'ns-obstacle-independent-residual-head-training-v1','parent_bank_sha256':sha(parent_bank_path),'parent_receipt_sha256':sha(parent_receipt_path),'parent_review_sha256':sha(parent_review_path),'parent_selected_theta_sha256':receipt['selected_theta_sha256'],'teacher_identity':teacher_identity,'config':config,'protocol':protocol,'prior_adam_steps':prior_adam,'prior_lbfgs_steps':prior_lb,'prior_allocation_seconds':review['allocated_seconds'],'warm_restart':True,'curvature_reset':True,'exact_optimizer_continuation_claim':False,'all_intermediate_vectors_retained':False,'retained_vector_audit_scope':'periodic and final checkpoints only; all accepted/trial scalar records and hashes retained','selection':'minimum accepted/initial combined full-training objective; no development/validation selection','objective_changed_from_parent':False,'base_definition':'unchanged mode-weighted normalized value plus gradient error','added_term_definition':'no objective term added; frozen parent plus independent residual capacity with unchanged supervised trace and traction channels','absolute_incompressibility_enforced':False,'causal_comparison_to_equal_additional_base_training':False,'source_sha256':sha(__file__),'parent_full_training_objective':receipt['selected_checkpoint']['objective'],'boundary_teacher':boundary_record,'coefficient_covariance':coefficient_record,'traction_definition':'sigma squared weighted local symmetric dimensionless gradient traction error / fixed global TRAIN teacher traction energy','traction_weight':traction_weight,'selected_theta_scope':'child parameters only; embedded parent hash is separate','optimizer_counts_scope':'historical lineage across parent and child parameter spaces, not exact continuation','serialized_replay_scope':'uncached JAX objective and child gradient from reloaded arrays; independent NumPy live evaluator fixture required before PDE deployment','architecture':{'head_count':4,'layer_widths':[21,64,64,2],'trainable_parameters':22792,'parent_frozen':True},'cache_contract':'explicit dynamic teacher/parent arguments; initial full streamed/cached child-gradient equivalence and final uncached serialized replay; no arbitrary-query cache'}
    bindings.update(schema='ns-obstacle-fixed-hidden-output-polish-v1',warm_restart=False,warm_child_model_sha256=sha(warm_model_path),warm_child_receipt_sha256=sha(warm_receipt_path),warm_child_review_sha256=sha(warm_review_path),warm_child_selected_theta_sha256=digest(warm_theta),prior_allocation_seconds=warm_review['allocated_seconds'],parent_full_training_objective=warm_receipt['selected_checkpoint']['objective'],added_term_definition='none; fixed hidden features and same objective; output-layer affine delta solve',optimizer_counts_scope='no new Adam or L-BFGS; four joint130-variable output solves; historical lineage retained',selection='initial or one jointly polished complete candidate, original full TRAIN objective only',polish_source_sha256=sha(Path(__file__).with_name('fixed_hidden_output_polish.py')),relative_svd_cutoff=required_cutoff,total_rows_per_head=total_rows,curvature_reset=False,retained_vector_audit_scope='initial/final child state, four full small augmented factors/SVDs/deltas; block hashes and counts, not all full design rows',architecture={'head_count':4,'layer_widths':[21,64,64,2],'changed_output_parameters':520,'frozen_hidden_parameters':22272,'parent_frozen':True})
    (out/'CONFIG.json').write_text(json.dumps(bindings,indent=2)+'\n')
    import jax,jax.numpy as jnp,jaxlib
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled or str(jax.config.jax_default_matmul_precision)!='highest':raise RuntimeError('GPU/f64/highest required')
    (out/'RUNTIME.json').write_text(json.dumps({'python':sys.version,'jax':jax.__version__,'jaxlib':jaxlib.__version__,'numpy':np.__version__,'backend':jax.default_backend(),'device_kind':[d.device_kind for d in jax.devices()],'x64':jax.config.x64_enabled,'matmul_precision':str(jax.config.jax_default_matmul_precision)},indent=2)+'\n')
    parent_raw=Path(parent_bank_path).read_bytes()
    parent_theta=np.concatenate([a.ravel() for layer in parent.layers for a in layer])
    if digest(parent_theta)!=receipt['selected_theta_sha256']:raise ValueError('frozen parent theta differs')
    if denominator!=float(parent.metadata['refinement']['boundary_teacher']['normalization']['global_training_traction_energy']) or boundary_record['arrays']!=parent.metadata['refinement']['boundary_teacher']['arrays']:raise ValueError('same frozen parent traction denominator and boundary arrays required')
    heads=warm.heads;shapes=[a.shape for h in heads for layer in h for a in layer];sizes=[int(np.prod(s)) for s in shapes];offsets=np.cumsum([0]+sizes);theta=np.concatenate([a.ravel() for h in heads for layer in h for a in layer])
    if len(theta)!=22792:raise ValueError('fixed head parameter count required')
    parent_runtime_layers=tuple(tuple(jax.device_put(a) for a in layer) for layer in parent.layers)
    parent_calc=jax.jit(lambda layers,xx:evaluate_layers(layers,xx,jnp))
    def unpack(t):
        values=[t[offsets[i]:offsets[i+1]].reshape(shape) for i,shape in enumerate(shapes)]
        layers=list(zip(values[::2],values[1::2]));return tuple(tuple(layers[i*3:(i+1)*3]) for i in range(4))
    def objective(t,xx,ww,yy,gg,pv,pg,nv,ng,importance):
        val,grad=evaluate_heads(unpack(t),xx,jnp);val=pv+val;grad=.1*(pg+grad)
        return block_objective(val,grad,yy,gg,ww,nv,ng,importance,divergence_weight,jnp)
    def boundary_objective(t,xx,ww,nn,gg,pg):
        _,grad=evaluate_heads(unpack(t),xx,jnp)
        return block_traction_objective(.1*(pg+grad),gg,nn,ww,bs,denominator,jnp)
    initial_base_calc=jax.jit(jax.grad(lambda t,xx,ww,yy,gg,pv,pg: objective(t,xx,ww,yy,gg,pv,pg,nv,ng,importance)[1]['base_objective']))
    boundary_calc=jax.jit(jax.value_and_grad(boundary_objective,has_aux=True))
    calc=jax.jit(jax.value_and_grad(objective,has_aux=True));evaluations=0;initiated_evaluations=0;accepted=0;pairs=[];started=time.monotonic();status='started';checkpoints=[];polished_heads=0;polish_complete=False;f=None;g=None;details=None;volume_cache=None;boundary_cache=None;equivalence_passed=False
    def guard(final=False):
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>16*2**30:raise MemoryError('conservative RSS rejection')
        if not final and time.time()>=deadline-90:raise TimeoutError('final replay and retention reserve')
    def full(t,final=False,phase='line_search',cached=True):
        nonlocal evaluations,initiated_evaluations
        initiated_evaluations+=1;eid=initiated_evaluations;tick=time.monotonic();completed_blocks=0;completed_boundary_blocks=0;device_theta=theta_once(t,jax)
        value=0.;derivative=np.zeros_like(t);parts_sum=None;initial_base_derivative=np.zeros_like(t)
        identity={'evaluation_id':eid,'phase':phase,'theta_sha256':digest(t),'accepted_steps_before':accepted,'completed_full_evaluations_before':evaluations,'elapsed_seconds':tick-started,'cached_teacher_and_parent':cached}
        append('EVALUATIONS.jsonl',dict(identity,status='pending'))
        try:
            for i in range(0,n,config['block_points']):
                guard(final);j=min(n,i+config['block_points'])
                if cached:
                    xx,ww,yy,gg,pv,pg=volume_cache.arguments[completed_blocks]
                else:
                    xx=jax.device_put(x[i:j]);ww=jax.device_put(w[i:j]);yy=jax.device_put(target[i:j]);gg=jax.device_put(gradient[i:j]);pv,pg=parent_calc(parent_runtime_layers,xx)
                (loss,parts),d=calc(device_theta,xx,ww,yy,gg,pv,pg,nv,ng,importance)
                value+=float(loss);derivative+=np.asarray(d)
                if parts_sum is None:parts_sum={k:np.zeros_like(np.asarray(v)) for k,v in parts.items()}
                for k,v in parts.items():parts_sum[k]+=np.asarray(v)
                if phase=='initial_parent':initial_base_derivative+=np.asarray(initial_base_calc(device_theta,xx,ww,yy,gg,pv,pg))
                completed_blocks+=1
            old_value=value;old_derivative=derivative.copy();traction_value=0.;traction_derivative=np.zeros_like(t);traction_parts=None
            for i in range(0,nt,config['block_points']):
                guard(final);j=min(nt,i+config['block_points'])
                if cached:
                    xx,ww,nn,gg,pg=boundary_cache.arguments[completed_boundary_blocks]
                else:
                    xx=jax.device_put(bx[i:j]);ww=jax.device_put(bw[i:j]);nn=jax.device_put(bn[i:j]);gg=jax.device_put(bg[i:j]);_,pg=parent_calc(parent_runtime_layers,xx)
                (loss,parts),d=boundary_calc(device_theta,xx,ww,nn,gg,pg)
                traction_value+=float(loss);traction_derivative+=np.asarray(d)
                if traction_parts is None:traction_parts={k:np.zeros_like(np.asarray(v)) for k,v in parts.items()}
                for k,v in parts.items():traction_parts[k]+=np.asarray(v)
                completed_boundary_blocks+=1
            if phase=='initial_parent':
                initial_components={'theta_sha256':digest(t),'definition':'Euclidean norms of full gradients in child-only flattened coordinates; not comparable to historical parent-parameter norms','base_gradient_l2':float(np.linalg.norm(initial_base_derivative)),'trace_gradient_l2':float(np.linalg.norm(old_derivative-initial_base_derivative)),'traction_gradient_l2_unweighted':float(np.linalg.norm(traction_derivative)),'traction_gradient_l2_weighted':float(traction_weight*np.linalg.norm(traction_derivative)),'parent_combined_gradient_l2':float(np.linalg.norm(old_derivative)),'new_combined_gradient_l2':float(np.linalg.norm(old_derivative+traction_weight*traction_derivative))}
                np.savez_compressed(out/'initial-component-gradients.npz',base=initial_base_derivative,trace=old_derivative-initial_base_derivative,traction_unweighted=traction_derivative)
                initial_components['arrays_sha256']=sha(out/'initial-component-gradients.npz')
                (out/'INITIAL-COMPONENT-GRADIENTS.json').write_text(json.dumps(initial_components,indent=2,allow_nan=False)+'\n')
            value=old_value+traction_weight*traction_value;derivative=old_derivative+traction_weight*traction_derivative
            parts_sum['parent_trace_combined_objective']=np.asarray(old_value)
            parts_sum['traction_objective_unweighted']=np.asarray(traction_value)
            parts_sum['traction_objective_weighted']=np.asarray(traction_weight*traction_value)
            parts_sum['combined_objective']=np.asarray(value)
            parts_sum['parent_trace_gradient_l2']=np.asarray(np.linalg.norm(old_derivative))
            parts_sum['traction_gradient_l2_unweighted']=np.asarray(np.linalg.norm(traction_derivative))
            parts_sum['combined_gradient_l2']=np.asarray(np.linalg.norm(derivative))
            parts_sum['parent_traction_gradient_dot_unweighted']=np.asarray(np.dot(old_derivative,traction_derivative))
            for k,v in traction_parts.items():parts_sum['traction_'+k]=v
            if not np.isfinite(value) or not np.isfinite(derivative).all() or any(not np.isfinite(v).all() for v in parts_sum.values()):raise FloatingPointError('nonfinite full objective')
            evaluations+=1
            metrics={k:v.tolist() for k,v in parts_sum.items()};metrics['seconds']=time.monotonic()-tick
            append('EVALUATIONS.jsonl',dict(identity,status='completed',completed_blocks=completed_blocks,completed_boundary_blocks=completed_boundary_blocks,objective=value,gradient_sha256=digest(derivative),gradient_inf=float(np.max(np.abs(derivative))),full_evaluations=evaluations,seconds=metrics['seconds']))
            return value,derivative,metrics
        except BaseException as exc:
            # A partial block sum is not a measured full-rule objective.
            try:
                failure_file=out/f'evaluation-{eid:06d}-failure.json'
                failure_file.write_text(json.dumps({'exception':repr(exc),'evaluation_identity':identity},indent=2)+'\n')
                append('EVALUATIONS.jsonl',dict(identity,status='incomplete',completed_blocks=completed_blocks,completed_boundary_blocks=completed_boundary_blocks,exception_record_file=failure_file.name,exception_record_sha256=sha(failure_file),full_objective_available=False,seconds=time.monotonic()-tick))
            except Exception as retention_error:print('Evaluation failure receipt failed:',repr(retention_error),flush=True)
            raise
    def append(name,row):
        with (out/name).open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush()
    def checkpoint(label):
        stem=f'state-{accepted:05d}-{label}';a={'theta':theta,'gradient':g}
        for i,(s,y,rho) in enumerate(pairs):a['s_'+str(i)]=s;a['y_'+str(i)]=y
        path=out/(stem+'.npz');np.savez_compressed(path,**a)
        row={'iteration':accepted,'label':label,'objective':f,'gradient_inf':float(np.max(np.abs(g))),'theta_sha256':digest(theta),'gradient_sha256':digest(g),'pair_rho':[z[2] for z in pairs],'array_file':path.name,'array_sha256':sha(path),'full_evaluations':evaluations,'metrics':details,'elapsed_seconds':time.monotonic()-started}
        (out/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n');checkpoints.append(row);return row
    def build_cache():
        nonlocal volume_cache,boundary_cache
        # Allocation bound includes teacher and parent value/physical-gradient payloads.
        payload=sum(a.nbytes for a in (x,w,target,gradient,bx,bw,bn,bg))+n*24*8+nt*16*8
        if payload>2**30:raise MemoryError('explicit combined device payload exceeds 1GiB')
        pv=np.empty((n,2,4));pg=np.empty((n,2,2,4));bpg=np.empty((nt,2,2,4))
        for i in range(0,n,config['block_points']):
            guard();j=min(n,i+config['block_points']);v,gp=parent_calc(parent_runtime_layers,jax.device_put(x[i:j]));pv[i:j]=np.asarray(v);pg[i:j]=np.asarray(gp)
        for i in range(0,nt,config['block_points']):
            guard();j=min(nt,i+config['block_points']);_,gp=parent_calc(parent_runtime_layers,jax.device_put(bx[i:j]));bpg[i:j]=np.asarray(gp)
        volume_cache=prepare_device_blocks(dict(x=x,w=w,target=target,gradient=gradient,parent_values=pv,parent_physical_gradient=pg),config['block_points'],jax)
        boundary_cache=prepare_device_blocks(dict(x=bx,w=bw,normal=bn,gradient=bg,parent_physical_gradient=bpg),config['block_points'],jax)
        witness={'parent_sha256':sha(parent_bank_path),'parent_theta_sha256':digest(parent_theta),'teacher_identity':teacher_identity,'boundary_teacher':boundary_record,'parent_volume_values_sha256':digest(pv),'parent_volume_physical_gradient_sha256':digest(pg),'parent_boundary_physical_gradient_sha256':digest(bpg),'volume_ranges':volume_cache.ranges,'boundary_ranges':boundary_cache.ranges,'logical_payload_bytes':payload,'payload_limit_bytes':2**30,'payload_is_not_peak_memory_bound':True,'device_memory_stats':[d.memory_stats() for d in jax.devices()],'host_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        (out/'DEVICE-CACHE.json').write_text(json.dumps(witness,indent=2)+'\n')
    try:
        f,g,details=full(theta,phase='initial_parent',cached=False)
        parent_replay_passed=bool(abs(f-bindings['parent_full_training_objective'])<=1e-10*max(1.,abs(f)))
        (out/'PARENT-BASE-REPLAY.json').write_text(json.dumps({'parent_objective':bindings['parent_full_training_objective'],'initial_base_objective':details['base_objective'],'initial_parent_trace_combined_objective':details['parent_trace_combined_objective'],'initial_combined_objective':f,'metrics':details,'passed':parent_replay_passed,'objective_changed_from_parent':False,'cross_objective_reduction_claim':False},indent=2)+'\n')
        if not parent_replay_passed:raise ValueError('initial unchanged objective does not replay reviewed warm child')
        streamed_f,streamed_g,streamed_details=f,g.copy(),details
        build_cache();cached_f,cached_g,cached_details=full(theta,phase='initial_cached',cached=True)
        np.savez_compressed(out/'initial-streamed-cached-equivalence.npz',theta=theta,streamed_objective=np.asarray(streamed_f),cached_objective=np.asarray(cached_f),streamed_gradient=streamed_g,cached_gradient=cached_g)
        component_equivalence={k:bool(np.allclose(np.asarray(streamed_details[k]),np.asarray(cached_details[k]),rtol=1e-10,atol=1e-10)) for k in streamed_details if k!='seconds'}
        equivalence={'component_equivalence':component_equivalence,'arrays_sha256':sha(out/'initial-streamed-cached-equivalence.npz'),'streamed_metrics':streamed_details,'cached_metrics':cached_details,'objective_gap':abs(streamed_f-cached_f),'gradient_max_abs_gap':float(np.max(np.abs(streamed_g-cached_g))),'tolerance':1e-10,'passed':bool(abs(streamed_f-cached_f)<=1e-10*max(1.,abs(streamed_f)) and np.max(np.abs(streamed_g-cached_g))<=1e-10),'parent_parameters_in_optimizer':False,'child_parameters':len(theta)}
        (out/'INITIAL-CACHE-EQUIVALENCE.json').write_text(json.dumps(equivalence,indent=2)+'\n')
        equivalence['passed']=equivalence['passed'] and all(component_equivalence.values())
        (out/'INITIAL-CACHE-EQUIVALENCE.json').write_text(json.dumps(equivalence,indent=2)+'\n')
        if not equivalence['passed']:raise RuntimeError('initial complete cached objective/child-gradient equivalence failed before updates')
        equivalence_passed=True;f,g,details=cached_f,cached_g,cached_details
        checkpoint('initial')
        initial_theta=theta.copy();initial_f=f;initial_g=g.copy();initial_details=dict(details)
        candidate=theta.copy();factors=[];solutions=[]
        def volume_design(hidden,output,xx,ww,yy,gg,pv,pg,mw,vnorm,gnorm):
            fv,fg=output_features(hidden,xx,jnp);ow=jnp.concatenate((output[0],output[1][None,:]),axis=0)
            current_v=pv+fv@ow;current_g=.1*(pg+jnp.einsum('njd,jc->ncd',fg,ow))
            return volume_rows(fv,.1*fg,current_v-yy,current_g-gg,ww,mw,vnorm,gnorm,divergence_weight,jnp)
        def boundary_design(hidden,output,xx,ww,nn,gg,pg,sigma):
            fv,fg=output_features(hidden,xx,jnp);ow=jnp.concatenate((output[0],output[1][None,:]),axis=0)
            current_g=.1*(pg+jnp.einsum('njd,jc->ncd',fg,ow))
            return boundary_rows(fv,.1*fg,current_g-gg,nn,ww,sigma,denominator,traction_weight,jnp)
        volume_design=jax.jit(volume_design);boundary_design=jax.jit(boundary_design)
        update_factor=jax.jit(lambda r,a,b:qr_append(r,a,b,jnp))
        first_factor=jax.jit(lambda a,b:qr_append(None,a,b,jnp))
        for head in range(4):
            tick=time.monotonic();factor=None;rows_done=0;blocks_done=0
            append('POLISH.jsonl',{'head':head,'status':'pending','rows_expected':total_rows})
            hidden=tuple(tuple(jax.device_put(a) for a in layer) for layer in heads[head][:2]);output=tuple(jax.device_put(a) for a in heads[head][2])
            try:
                for location,count in [('volume',n),('boundary',nt)]:
                    for i in range(0,count,protocol['qr_block_points']):
                        guard();j=min(count,i+protocol['qr_block_points'])
                        if location=='volume':
                            xx=jax.device_put(x[i:j]);pv,pg=parent_calc(parent_runtime_layers,xx)
                            aa,bb=volume_design(hidden,output,xx,jax.device_put(w[i:j]),jax.device_put(target[i:j,:,head]),jax.device_put(gradient[i:j,:,:,head]),pv[:,:,head],pg[:,:,:,head],importance[head],nv[head],ng[head])
                        else:
                            xx=jax.device_put(bx[i:j]);_,pg=parent_calc(parent_runtime_layers,xx)
                            aa,bb=boundary_design(hidden,output,xx,jax.device_put(bw[i:j]),jax.device_put(bn[i:j]),jax.device_put(bg[i:j,:,:,head]),pg[:,:,:,head],bs[head])
                        factor=first_factor(aa,bb) if factor is None else update_factor(factor,aa,bb)
                        host_factor=np.asarray(factor)
                        if not np.isfinite(host_factor).all():
                            np.savez_compressed(out/f'head-{head}-failed-factor.npz',factor=host_factor)
                            raise FloatingPointError('nonfinite QR factor')
                        rows_done+=len(aa);blocks_done+=1
                        append('QR-BLOCKS.jsonl',{'head':head,'location':location,'start':i,'end':j,'rows_processed':rows_done,'blocks_processed':blocks_done,'factor_sha256':digest(host_factor),'elapsed_seconds':time.monotonic()-tick})
                if rows_done!=total_rows:raise ValueError('incomplete full TRAIN affine system')
                solved=solve_factor(host_factor,required_cutoff)
                raw=out/f'head-{head}-qr-svd.npz';np.savez_compressed(raw,**{k:v for k,v in solved.items() if isinstance(v,np.ndarray)})
                stats={k:v for k,v in solved.items() if not isinstance(v,np.ndarray)}
                stats.update(head=head,rows_processed=rows_done,blocks_processed=blocks_done,arrays_sha256=sha(raw),status='completed')
                (out/f'head-{head}-qr-svd.json').write_text(json.dumps(stats,indent=2)+'\n')
                if not np.isfinite(solved['delta']).all():raise FloatingPointError('nonfinite output delta')
                start=head*5698+5568;candidate[start:start+130]+=solved['delta'];factors.append(stats);solutions.append(solved);polished_heads+=1
                append('POLISH.jsonl',stats)
            except BaseException as exc:
                append('POLISH.jsonl',{'head':head,'status':'incomplete','rows_processed':rows_done,'blocks_processed':blocks_done,'factor_progress_after_last_completed_block_only':True,'exception':repr(exc)})
                raise
        hidden_indices=np.ones(len(theta),dtype=bool)
        for head in range(4):hidden_indices[head*5698+5568:head*5698+5698]=False
        if not np.array_equal(candidate[hidden_indices],initial_theta[hidden_indices]):raise ValueError('hidden parameters changed')
        candidate_f,candidate_g,candidate_details=full(candidate,phase='complete_polished_candidate_uncached',cached=False)
        np.savez_compressed(out/'complete-polish-candidate.npz',initial_theta=initial_theta,candidate_theta=candidate,initial_gradient=initial_g,candidate_gradient=candidate_g,initial_objective=np.asarray(initial_f),candidate_objective=np.asarray(candidate_f))
        output_indices=np.concatenate([np.arange(h*5698+5568,h*5698+5698) for h in range(4)])
        factor_gradient=np.concatenate([v['factor_gradient'] for v in solutions]);actual_output_gradient=candidate_g[output_indices]
        factor_initial=sum(v['initial_factor_objective'] for v in solutions);factor_candidate=sum(v['candidate_factor_objective'] for v in solutions)
        eps=np.finfo(np.float64).eps;roundoff_scales=[]
        for v,stats in zip(solutions,factors):
            an=float(np.linalg.norm(v['augmented_factor'][:,:130]));bnorm=float(np.linalg.norm(v['augmented_factor'][:,130]));dn=float(np.linalg.norm(v['delta']));blocks=stats['blocks_processed']
            roundoff_scales.append({'factor_matrix_frobenius':an,'factor_rhs_norm':bnorm,'delta_norm':dn,'blocks':blocks,'candidate_objective_allowance':128*eps*blocks*(an*dn+bnorm)**2,'gradient_allowance':256*eps*blocks*an*(an*dn+bnorm)})
        initial_allowance=128*eps*sum(v['blocks_processed'] for v in factors)*max(1.,abs(initial_f))
        candidate_allowance=max(128*eps,sum(v['candidate_objective_allowance'] for v in roundoff_scales))
        gradient_allowance=max(256*eps,max(v['gradient_allowance'] for v in roundoff_scales))
        objective_cap=1e-8*max(1.,abs(initial_f),abs(candidate_f));gradient_cap=1e-8*max(1.,float(np.max(np.abs(actual_output_gradient))))
        witness_path=out/'affine-original-objective-equivalence.npz'
        np.savez_compressed(witness_path,factor_initial_objective=np.asarray(factor_initial),actual_initial_objective=np.asarray(initial_f),factor_candidate_objective=np.asarray(factor_candidate),actual_candidate_objective=np.asarray(candidate_f),factor_gradient=factor_gradient,actual_output_gradient=actual_output_gradient,output_indices=output_indices)
        assembly={'arrays_sha256':sha(witness_path),'initial_objective_gap':abs(factor_initial-initial_f),'candidate_objective_gap':abs(factor_candidate-candidate_f),'output_gradient_max_gap':float(np.max(np.abs(factor_gradient-actual_output_gradient))),'initial_allowance':initial_allowance,'candidate_allowance':candidate_allowance,'gradient_allowance':gradient_allowance,'objective_cap':objective_cap,'gradient_cap':gradient_cap,'roundoff_scales':roundoff_scales,'tolerance_policy':'conservative dimension/accumulation and cancellation-scale arithmetic allowances with fixed caps; not a proved backward-error guarantee; reject ill-conditioned uncertainty rather than relax cap'}
        assembly['passed']=bool(initial_allowance<=objective_cap and candidate_allowance<=objective_cap and gradient_allowance<=gradient_cap and assembly['initial_objective_gap']<=initial_allowance and assembly['candidate_objective_gap']<=candidate_allowance and assembly['output_gradient_max_gap']<=gradient_allowance)
        (out/'AFFINE-ORIGINAL-OBJECTIVE-EQUIVALENCE.json').write_text(json.dumps(assembly,indent=2)+'\n')
        if not assembly['passed']:raise RuntimeError('full original objective/output-gradient does not certify affine assembly within frozen arithmetic bounds')
        tolerance=100*np.finfo(np.float64).eps*max(1.,abs(initial_f))
        no_worse=bool(candidate_f<=initial_f+tolerance)
        comparison={'initial_objective':initial_f,'candidate_objective':candidate_f,'roundoff_allowance':tolerance,'no_worse':no_worse,'initial_metrics':initial_details,'candidate_metrics':candidate_details,'arrays_sha256':sha(out/'complete-polish-candidate.npz'),'complete_heads':polished_heads,'new_LBFGS_steps':0,'new_Adam_steps':0,'numerical_full_rank_heads':[v['full_column_rank'] for v in factors],'cutoff_limited_delta':True}
        (out/'POLISH-COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n')
        if not no_worse:raise RuntimeError('complete polished objective worse than original beyond roundoff')
        if candidate_f<=initial_f:theta,f,g,details=candidate,candidate_f,candidate_g,candidate_details
        polish_complete=True;status='output_polish_complete_not_physical_admission'
    except TimeoutError:status='job_deadline'
    except BaseException as exc:
        status='failed'
        try:(out/'FAILURE.json').write_text(json.dumps({'exception':repr(exc),'accepted_steps':accepted})+'\n')
        except Exception as error:print('Failure detail retention failed:',repr(error),flush=True)
        raise
    finally:
        retention_errors=[]
        try:
            if f is not None and g is not None:final_checkpoint=checkpoint('final')
        except Exception as error:retention_errors.append(repr(error))
        try:(out/'FINAL.json').write_text(json.dumps({'status':status,'accepted_steps':accepted,'cumulative_adam_steps':prior_adam,'cumulative_lbfgs_steps':prior_lb+accepted,'full_evaluations':evaluations,'initiated_evaluations':initiated_evaluations,'optimizer_elapsed_seconds':time.monotonic()-started,'checkpoints':checkpoints,'retention_errors':retention_errors,'all_intermediate_vectors_retained':False,'warm_restart':False,'exact_optimizer_continuation_claim':False,'validation_seen':False,'polished_heads':polished_heads,'polish_complete':polish_complete},indent=2)+'\n')
        except Exception as error:retention_errors.append(repr(error));print('FINAL retention failed:',repr(error),flush=True)
        if retention_errors and status!='failed':raise RuntimeError('optimizer retention incomplete: '+repr(retention_errors))
    if f is None or g is None:raise RuntimeError('no full-rule checkpoint')
    if not polish_complete:raise RuntimeError('no export of incomplete output polish')
    if not equivalence_passed:raise RuntimeError('no model export before full initial equivalence admission')
    bank=ResidualHeadBank(parent_raw,sha(parent_bank_path),unpack(theta),dict(parent.metadata,refinement=bindings,objective_variant='fixed_hidden_output_polish_same_trace_and_traction',trained=True,new_adam_steps=0,new_lbfgs_steps=accepted,actual_applied_adam_steps=prior_adam,accepted_lbfgs_steps=prior_lb+accepted,selected_checkpoint={'objective':f,'checkpoint':final_checkpoint['array_file']},status=status,head_seed=protocol['head_seed']))
    bank_sha=bank.save(out/'spatial-bank-raw.npz');loaded=ResidualHeadBank.load(out/'spatial-bank-raw.npz');selected=np.concatenate([a.ravel() for h in loaded.heads for layer in h for a in layer]);assert digest(selected)==digest(theta)
    if loaded.parent_sha256!=sha(parent_bank_path) or loaded.parent_bytes!=parent_raw:raise ValueError('serialized frozen parent changed')
    parent_runtime_layers=tuple(tuple(jax.device_put(a) for a in layer) for layer in loaded.parent.layers)
    replay_f,replay_g,replay_details=full(selected,final=True,phase='serialized_selected_uncached',cached=False)
    np.savez_compressed(out/'selected-uncached-full-replay.npz',selected_theta=selected,cached_objective=np.asarray(f),uncached_objective=np.asarray(replay_f),cached_gradient=g,uncached_gradient=replay_g)
    component_replay={k:bool(np.allclose(np.asarray(details[k]),np.asarray(replay_details[k]),rtol=1e-10,atol=1e-10)) for k in details if k!='seconds'}
    replay={'component_replay':component_replay,'raw_replay_arrays_sha256':sha(out/'selected-uncached-full-replay.npz'),'selected_model_sha256':bank_sha,'selected_theta_sha256':digest(selected),'objective':replay_f,'objective_gap':abs(replay_f-f),'gradient_max_gap':float(np.max(np.abs(replay_g-g))),'metrics':replay_details,'passed':bool(abs(replay_f-f)<=1e-10*max(1.,abs(f)) and np.max(np.abs(replay_g-g))<=1e-10)}
    replay['passed']=replay['passed'] and all(component_replay.values())
    (out/'SELECTED-FULL-REPLAY.json').write_text(json.dumps(replay,indent=2)+'\n')
    if not replay['passed']:raise RuntimeError('serialized selected-model full objective replay failed')
    (out/'MODEL-RECEIPT.json').write_text(json.dumps({'schema':'ns-obstacle-supervised-neural-model-v2','bank_file':'spatial-bank-raw.npz','bank_sha256':bank_sha,'selected_checkpoint':bank.metadata['selected_checkpoint'],'selected_theta_sha256':digest(theta),'config_sha256':sha(out/'CONFIG.json'),'runtime_sha256':sha(out/'RUNTIME.json'),'optimizer_status':status,'warm_restart':False,'objective_variant':'fixed_hidden_output_polish_same_trace_and_traction','selection':'minimum initial or complete polished candidate original objective','selected_theta_scope':'child parameters only','parent_model_sha256':sha(parent_bank_path),'warm_child_model_sha256':sha(warm_model_path),'optimizer_counts_scope':'four output affine solves; zero new Adam/LBFGS; prior lineage retained','selected_base_objective':details['base_objective'],'cumulative_adam_steps':prior_adam,'cumulative_lbfgs_steps':prior_lb+accepted,'selected_full_replay_sha256':sha(out/'SELECTED-FULL-REPLAY.json'),'total_full_evaluations_including_serialized_replay':evaluations,'total_training_replay_retention_seconds':time.monotonic()-started,'physical_admission':False},indent=2)+'\n')
    return bank


def refine(x,w,target,gradient,teacher_identity,boundary,boundary_record,coefficient_record,parent_bank_path,parent_receipt_path,parent_review_path,warm_model_path,warm_receipt_path,warm_review_path,protocol,directory,deadline):
    out=Path(directory)
    if out.exists():raise FileExistsError('fresh refinement output required')
    try:return _refine(x,w,target,gradient,teacher_identity,boundary,boundary_record,coefficient_record,parent_bank_path,parent_receipt_path,parent_review_path,warm_model_path,warm_receipt_path,warm_review_path,protocol,directory,deadline)
    except BaseException as exc:
        if out.exists():
            try:(out/'TOP-LEVEL-FAILURE.json').write_text(json.dumps({'exception':repr(exc),'model_export_complete':(out/'MODEL-RECEIPT.json').exists()})+'\n')
            except Exception as error:print('Failure retention failed:',repr(error),flush=True)
            if not (out/'FINAL.json').exists():
                try:(out/'FINAL.json').write_text(json.dumps({'status':'failed','original_exception':repr(exc),'validation_seen':False})+'\n')
                except Exception as error:print('Could not retain FINAL:',repr(error),flush=True)
        raise
