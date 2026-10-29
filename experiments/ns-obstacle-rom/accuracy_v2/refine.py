"""Same-objective warm L-BFGS; periodic state retention, no Adam or exact-resume claim."""
from pathlib import Path
import hashlib,json,time,resource,sys
import numpy as np
from accuracy_v2.spatial_bank import SpatialBank,evaluate_layers
from accuracy_v2.train import digest

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def _refine(x,w,target,gradient,teacher_identity,parent_bank_path,parent_receipt_path,parent_review_path,protocol,directory,deadline):
    out=Path(directory);out.mkdir(exist_ok=False)
    parent=SpatialBank.load(parent_bank_path);receipt=json.loads(Path(parent_receipt_path).read_text());review=json.loads(Path(parent_review_path).read_text())
    if parent.rank!=4 or not np.array_equal(parent.transform,np.eye(4)) or receipt['bank_sha256']!=sha(parent_bank_path) or review.get('passed') is not True or review.get('model_sha256')!=sha(parent_bank_path) or review.get('model_receipt_sha256')!=sha(parent_receipt_path):raise ValueError('reviewed unrotated rank4 parent required')
    if parent.metadata['teacher_identity']!=teacher_identity or teacher_identity['split']!='training' or teacher_identity['Re']!=[15+1.25*j for j in range(9)]:raise ValueError('same genuine training teacher required')
    if protocol!={'max_iterations':10000,'checkpoint_every':250,'history_size':10,'line_search_steps':20,'gradient_tolerance':1e-10,'warm_restart':True,'new_adam_steps':0}:raise ValueError('frozen compact warm-restart protocol required')
    x,w,target,gradient=map(lambda a:np.asarray(a,dtype=np.float64),(x,w,target,gradient));n=len(x)
    if x.shape!=(n,2) or w.shape!=(n,) or target.shape!=(n,2,4) or gradient.shape!=(n,2,2,4) or np.min(w)<=0 or any(not np.isfinite(a).all() for a in (x,w,target,gradient)):raise ValueError('finite same-layout teacher required')
    ids={k:digest(v) for k,v in zip(('points','weights','values','gradients'),(x,w,target,gradient))}
    if ids!=teacher_identity['arrays']:raise ValueError('actual teacher bytes changed')
    nv=np.einsum('n,ncr,ncr->r',w,target,target);ng=np.einsum('n,ncdr,ncdr->r',w,gradient,gradient)
    sigma=np.asarray(teacher_identity['singular_values']);importance=np.sqrt(sigma/sigma[0]);importance/=importance.sum()
    for observed,key in ((nv,'value_norms_squared'),(ng,'gradient_norms_squared'),(importance,'mode_weights')):
        if not np.array_equal(observed,np.asarray(parent.metadata[key])):raise ValueError('objective normalization changed')
    config=parent.metadata['config'];prior_adam=int(parent.metadata['actual_applied_adam_steps']);prior_lb=int(parent.metadata['accepted_lbfgs_steps'])
    if review.get('adam_steps_recorded')!=prior_adam or review.get('accepted_lbfgs_steps')!=prior_lb or review.get('selected_theta_sha256')!=receipt['selected_theta_sha256'] or type(review.get('allocated_seconds')) is not int:raise ValueError('reviewed cumulative parent training provenance required')
    bindings={'schema':'ns-obstacle-compact-warm-LBFGS-v1','parent_bank_sha256':sha(parent_bank_path),'parent_receipt_sha256':sha(parent_receipt_path),'parent_review_sha256':sha(parent_review_path),'parent_selected_theta_sha256':receipt['selected_theta_sha256'],'teacher_identity':teacher_identity,'config':config,'protocol':protocol,'prior_adam_steps':prior_adam,'prior_lbfgs_steps':prior_lb,'prior_allocation_seconds':review['allocated_seconds'],'warm_restart':True,'curvature_reset':True,'exact_optimizer_continuation_claim':False,'all_intermediate_vectors_retained':False,'retained_vector_audit_scope':'periodic and final checkpoints only; all accepted/trial scalar records and hashes retained','selection':'minimum measured full-training objective, no development/validation selection','source_sha256':sha(__file__),'parent_full_training_objective':receipt['selected_checkpoint']['objective']}
    (out/'CONFIG.json').write_text(json.dumps(bindings,indent=2)+'\n')
    import jax,jax.numpy as jnp,jaxlib
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled or str(jax.config.jax_default_matmul_precision)!='highest':raise RuntimeError('GPU/f64/highest required')
    (out/'RUNTIME.json').write_text(json.dumps({'python':sys.version,'jax':jax.__version__,'jaxlib':jaxlib.__version__,'numpy':np.__version__,'backend':jax.default_backend(),'device_kind':[d.device_kind for d in jax.devices()],'x64':jax.config.x64_enabled,'matmul_precision':str(jax.config.jax_default_matmul_precision)},indent=2)+'\n')
    shapes=[a.shape for layer in parent.layers for a in layer];sizes=[int(np.prod(s)) for s in shapes];offsets=np.cumsum([0]+sizes);theta=np.concatenate([a.ravel() for layer in parent.layers for a in layer])
    if digest(theta)!=receipt['selected_theta_sha256']:raise ValueError('warm-start theta differs')
    def unpack(t):
        values=[t[offsets[i]:offsets[i+1]].reshape(s) for i,s in enumerate(shapes)];return list(zip(values[::2],values[1::2]))
    def objective(t,xx,ww,yy,gg,nv,ng,importance):
        val,grad=evaluate_layers(unpack(t),xx,jnp);grad=.1*grad
        ev=jnp.einsum('n,ncr,ncr->r',ww,val-yy,val-yy)/nv;eg=jnp.einsum('n,ncdr,ncdr->r',ww,grad-gg,grad-gg)/ng
        return jnp.sum(importance*(ev+eg)),(ev,eg)
    calc=jax.jit(jax.value_and_grad(objective,has_aux=True));evaluations=0;accepted=0;pairs=[];started=time.monotonic();status='started';checkpoints=[];f=None;g=None;details=None
    def guard(final=False):
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>16*2**30:raise MemoryError('conservative RSS rejection')
        if not final and time.time()>=deadline-90:raise TimeoutError('final replay and retention reserve')
    def full(t,final=False):
        nonlocal evaluations
        value=0.;derivative=np.zeros_like(t);ev=np.zeros(4);eg=np.zeros(4);tick=time.monotonic()
        for i in range(0,n,config['block_points']):
            guard(final);j=min(n,i+config['block_points'])
            (loss,parts),d=calc(jnp.asarray(t),x[i:j],w[i:j],target[i:j],gradient[i:j],nv,ng,importance)
            value+=float(loss);derivative+=np.asarray(d);ev+=np.asarray(parts[0]);eg+=np.asarray(parts[1])
        evaluations+=1
        if not np.isfinite(value) or not np.isfinite(derivative).all():raise FloatingPointError('nonfinite full objective')
        return value,derivative,{'value_relative_squared_by_mode':ev.tolist(),'gradient_relative_squared_by_mode':eg.tolist(),'seconds':time.monotonic()-tick}
    def append(name,row):
        with (out/name).open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush()
    def checkpoint(label):
        stem=f'state-{accepted:05d}-{label}';a={'theta':theta,'gradient':g}
        for i,(s,y,rho) in enumerate(pairs):a['s_'+str(i)]=s;a['y_'+str(i)]=y
        path=out/(stem+'.npz');np.savez_compressed(path,**a)
        row={'iteration':accepted,'label':label,'objective':f,'gradient_inf':float(np.max(np.abs(g))),'theta_sha256':digest(theta),'gradient_sha256':digest(g),'pair_rho':[z[2] for z in pairs],'array_file':path.name,'array_sha256':sha(path),'full_evaluations':evaluations,'metrics':details,'elapsed_seconds':time.monotonic()-started}
        (out/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n');checkpoints.append(row);return row
    try:
        f,g,details=full(theta)
        if abs(f-bindings['parent_full_training_objective'])>1e-10*max(1.,abs(f)):raise ValueError('initial full objective does not replay parent')
        checkpoint('initial')
        for iteration in range(protocol['max_iterations']):
            guard()
            if np.max(np.abs(g))<=protocol['gradient_tolerance']:status='gradient_stationary';break
            q=g.copy();alphas=[]
            for s,y,rho in reversed(pairs):alpha=rho*np.dot(s,q);alphas.append(alpha);q-=alpha*y
            if pairs:q*=np.dot(pairs[-1][0],pairs[-1][1])/np.dot(pairs[-1][1],pairs[-1][1])
            for (s,y,rho),alpha in zip(pairs,reversed(alphas)):q+=s*(alpha-rho*np.dot(y,q))
            direction=-q;slope=float(np.dot(g,direction));reset=False
            if not np.isfinite(slope) or slope>=0:direction=-g;slope=-float(np.dot(g,g));pairs=[];reset=True
            found=False;old_f=f;old_theta=digest(theta);old_gradient=digest(g)
            for attempt in range(protocol['line_search_steps']):
                step=2.**(-attempt);trial=theta+step*direction;nf,dg,nd=full(trial);found=bool(nf<=f+1e-4*step*slope)
                append('LINE-SEARCH.jsonl',{'iteration':iteration,'attempt':attempt,'step':step,'objective':nf,'prior_objective':f,'slope':slope,'accepted':found,'theta_sha256':digest(trial),'gradient_sha256':digest(dg),'direction_sha256':digest(direction),'direction_norm':float(np.linalg.norm(direction)),'gradient_inf':float(np.max(np.abs(dg))),'metrics':nd})
                if found:break
            if not found:status='line_search_stopped';break
            s=trial-theta;y=dg-g;sy=float(np.dot(s,y));ss=float(np.dot(s,s));yy=float(np.dot(y,y));keep=bool(sy>1e-12*np.linalg.norm(s)*np.linalg.norm(y))
            if keep:pairs.append((s,y,1/sy));pairs=pairs[-protocol['history_size']:]
            theta,f,g,details=trial,nf,dg,nd;accepted+=1
            append('ACCEPTED.jsonl',{'iteration':accepted,'prior_theta_sha256':old_theta,'prior_gradient_sha256':old_gradient,'theta_sha256':digest(theta),'gradient_sha256':digest(g),'prior_objective':old_f,'full_post_update_objective':f,'gradient_inf':float(np.max(np.abs(g))),'slope':slope,'step':step,'attempt':attempt,'curvature_sy':sy,'curvature_ss':ss,'curvature_yy':yy,'curvature_kept':keep,'direction_reset':reset,'direction_sha256':digest(direction),'direction_norm':float(np.linalg.norm(direction)),'curvature_threshold':float(1e-12*np.linalg.norm(s)*np.linalg.norm(y)),'history_size':len(pairs),'cumulative_lbfgs_steps':prior_lb+accepted,'metrics':details,'full_evaluations':evaluations,'elapsed_seconds':time.monotonic()-started})
            if accepted%protocol['checkpoint_every']==0:checkpoint('periodic')
        else:status='iteration_limit'
    except TimeoutError:status='job_deadline'
    except BaseException as exc:
        status='failed';(out/'FAILURE.json').write_text(json.dumps({'exception':repr(exc),'accepted_steps':accepted})+'\n');raise
    finally:
        retention_errors=[]
        try:
            if f is not None and g is not None:final_checkpoint=checkpoint('final')
        except Exception as error:retention_errors.append(repr(error))
        try:(out/'FINAL.json').write_text(json.dumps({'status':status,'accepted_steps':accepted,'cumulative_adam_steps':prior_adam,'cumulative_lbfgs_steps':prior_lb+accepted,'full_evaluations':evaluations,'optimizer_elapsed_seconds':time.monotonic()-started,'checkpoints':checkpoints,'retention_errors':retention_errors,'all_intermediate_vectors_retained':False,'warm_restart':True,'exact_optimizer_continuation_claim':False,'validation_seen':False},indent=2)+'\n')
        except Exception as error:retention_errors.append(repr(error));print('FINAL retention failed:',repr(error),flush=True)
        if retention_errors and status!='failed':raise RuntimeError('optimizer retention incomplete: '+repr(retention_errors))
    if f is None or g is None:raise RuntimeError('no full-rule checkpoint')
    bank=SpatialBank([(np.asarray(a),np.asarray(b)) for a,b in unpack(theta)],np.eye(4),dict(parent.metadata,refinement=bindings,trained=True,actual_applied_adam_steps=prior_adam,accepted_lbfgs_steps=prior_lb+accepted,selected_checkpoint={'objective':f,'checkpoint':final_checkpoint['array_file']},status=status))
    bank_sha=bank.save(out/'spatial-bank-raw.npz');loaded=SpatialBank.load(out/'spatial-bank-raw.npz');selected=np.concatenate([a.ravel() for layer in loaded.layers for a in layer]);assert digest(selected)==digest(theta)
    replay_f,replay_g,replay_details=full(selected,final=True)
    replay={'selected_model_sha256':bank_sha,'selected_theta_sha256':digest(selected),'objective':replay_f,'objective_gap':abs(replay_f-f),'gradient_max_gap':float(np.max(np.abs(replay_g-g))),'metrics':replay_details,'passed':bool(abs(replay_f-f)<=1e-10*max(1.,abs(f)) and np.max(np.abs(replay_g-g))<=1e-10)}
    (out/'SELECTED-FULL-REPLAY.json').write_text(json.dumps(replay,indent=2)+'\n')
    if not replay['passed']:raise RuntimeError('serialized selected-model full objective replay failed')
    (out/'MODEL-RECEIPT.json').write_text(json.dumps({'schema':'ns-obstacle-supervised-neural-model-v2','bank_file':'spatial-bank-raw.npz','bank_sha256':bank_sha,'selected_checkpoint':bank.metadata['selected_checkpoint'],'selected_theta_sha256':digest(theta),'config_sha256':sha(out/'CONFIG.json'),'runtime_sha256':sha(out/'RUNTIME.json'),'optimizer_status':status,'warm_restart':True,'cumulative_adam_steps':prior_adam,'cumulative_lbfgs_steps':prior_lb+accepted,'selected_full_replay_sha256':sha(out/'SELECTED-FULL-REPLAY.json'),'total_full_evaluations_including_serialized_replay':evaluations,'total_training_replay_retention_seconds':time.monotonic()-started,'physical_admission':False},indent=2)+'\n')
    return bank


def refine(x,w,target,gradient,teacher_identity,parent_bank_path,parent_receipt_path,parent_review_path,protocol,directory,deadline):
    out=Path(directory)
    if out.exists():raise FileExistsError('fresh refinement output required')
    try:return _refine(x,w,target,gradient,teacher_identity,parent_bank_path,parent_receipt_path,parent_review_path,protocol,directory,deadline)
    except BaseException as exc:
        if out.exists():
            try:(out/'TOP-LEVEL-FAILURE.json').write_text(json.dumps({'exception':repr(exc),'model_export_complete':(out/'MODEL-RECEIPT.json').exists()})+'\n')
            except Exception as error:print('Failure retention failed:',repr(error),flush=True)
            if not (out/'FINAL.json').exists():
                try:(out/'FINAL.json').write_text(json.dumps({'status':'failed','original_exception':repr(exc),'validation_seen':False})+'\n')
                except Exception as error:print('Could not retain FINAL:',repr(error),flush=True)
        raise
