"""Training-only supervised modal values+gradients; deterministic full-rule L-BFGS.

Checkpoint selection uses only the lowest measured post-update full-training objective.
Every full measurement is immutable. L-BFGS curvature is reconstructed from retained
accepted states; no duplicate history arrays or automatic restart/budget reset.
"""
from pathlib import Path
import hashlib,json,time,sys,resource
import numpy as np
from qualification import require_qualification
from accuracy_v2.spatial_bank import initialize,evaluate_layers,SpatialBank

def digest(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()

def _train(x,w,target,gradient,singular_values,qualification,config,directory,deadline,resume=None,teacher_identity=None):
    qualification=require_qualification(qualification)
    if not isinstance(teacher_identity,dict) or teacher_identity.get('split')!='training' or teacher_identity.get('Re')!=[15+1.25*j for j in range(9)] or require_qualification(teacher_identity.get('reference_qualification'))!=qualification:raise ValueError('actual qualified nine-state training identity required')
    for key in ('prepared_manifest_sha256','reference_admission_sha256','volume_manifest_sha256','training_array_sha256'):
        value=teacher_identity.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('missing bound teacher input identity')
    if config['split']!='training' or config['rank']!=4 or config['teacher_definition']!='prepared_training_POD':raise ValueError('training-only rank4 teacher required')
    x,w,target,gradient=map(lambda a:np.asarray(a,dtype=np.float64),(x,w,target,gradient))
    n=len(x)
    if x.shape!=(n,2) or w.shape!=(n,) or target.shape!=(n,2,4) or gradient.shape!=(n,2,2,4) or np.min(w)<=0:raise ValueError('teacher layout')
    if any(not np.isfinite(a).all() for a in (x,w,target,gradient)):raise ValueError('finite teachers required')
    sigma=np.asarray(singular_values,dtype=np.float64)[:4]
    if not np.array_equal(sigma,np.asarray(teacher_identity['singular_values'],dtype=np.float64)):raise ValueError('actual teacher singular values differ')
    if sigma.shape!=(4,) or np.min(sigma)<=0 or np.any(np.diff(sigma)>0):raise ValueError('ordered positive actual singular values required')
    norms_v=np.einsum('n,ncr,ncr->r',w,target,target);norms_g=np.einsum('n,ncdr,ncdr->r',w,gradient,gradient)
    if min(norms_v.min(),norms_g.min())<=0:raise ValueError('null teacher norm')
    importance=np.sqrt(sigma/sigma[0]);importance/=importance.sum()
    arrays_identity={'points':digest(x),'weights':digest(w),'values':digest(target),'gradients':digest(gradient)}
    if arrays_identity!=teacher_identity['arrays']:raise ValueError('actual teacher arrays differ from verified loader')
    from pilot2d import source_digest as inherited_source_digest
    out=Path(directory);out.mkdir(exist_ok=False)
    bindings={'teacher_identity':teacher_identity,'inherited_loader_source_sha256':inherited_source_digest(),'source_files':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('train.py','spatial_bank.py','continuation.py','teacher.py')},'points':digest(x),'weights':digest(w),'values':digest(target),'gradients':digest(gradient),'qualification':qualification,'config':config,'singular_values':sigma.tolist(),'mode_weights':importance.tolist(),'value_norms_squared':norms_v.tolist(),'gradient_norms_squared':norms_g.tolist(),'selection':'minimum measured post-update full-training value+gradient objective; no development/validation selection'}
    (out/'CONFIG.json').write_text(json.dumps(bindings,indent=2)+'\n')
    import jax
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
    import jax.numpy as jnp
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    import jaxlib,scipy
    runtime={'python':sys.version,'jax':jax.__version__,'jaxlib':jaxlib.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'backend':jax.default_backend(),'devices':[{'platform':d.platform,'device_kind':d.device_kind,'id':d.id} for d in jax.devices()],'x64':bool(jax.config.x64_enabled),'matmul_precision':str(jax.config.jax_default_matmul_precision),'device_float_dtype':str(jnp.asarray(1.,dtype=jnp.float64).dtype),'input_dtypes':[str(a.dtype) for a in (x,w,target,gradient)]}
    (out/'RUNTIME.json').write_text(json.dumps(runtime,indent=2)+'\n')
    if runtime['device_float_dtype']!='float64' or not runtime['x64'] or runtime['matmul_precision']!='highest':raise RuntimeError('f64/highest runtime required')
    model=initialize(rank=4,width=config['width'],depth=config['depth'],seed=config['seed']);shapes=[a.shape for layer in model.layers for a in layer];sizes=[int(np.prod(s)) for s in shapes];offsets=np.cumsum([0]+sizes)
    theta=np.concatenate([a.ravel() for layer in model.layers for a in layer]);m=np.zeros_like(theta);v=np.zeros_like(theta)
    def unpack(t):
        values=[t[offsets[i]:offsets[i+1]].reshape(s) for i,s in enumerate(shapes)];return list(zip(values[::2],values[1::2]))
    def objective(t,xx,ww,yy,gg,nv,ng,importance_arg):
        val,grad=evaluate_layers(unpack(t),xx,jnp);grad=.1*grad
        ev=jnp.einsum('n,ncr,ncr->r',ww,val-yy,val-yy)/nv
        eg=jnp.einsum('n,ncdr,ncdr->r',ww,grad-gg,grad-gg)/ng
        return jnp.sum(importance_arg*(ev+eg)),(ev,eg)
    calc=jax.jit(jax.value_and_grad(objective,has_aux=True))
    started=time.monotonic();prior_elapsed=0.;completed_lbfgs=0;retention_errors=[];evaluations=0;serial=0;best=None;history=[];state_ids=[];curvature=[];rng=np.random.default_rng(config['seed']);adam_step=0;phase='initial';last_checkpoint=None
    def guard():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>16*2**30:raise MemoryError('unchanged16GiB process guard')
        if time.time()>=deadline-60:raise TimeoutError('setup-inclusive job deadline; finalization reserved')
    def full(t):
        nonlocal evaluations
        value=0.;derivative=np.zeros_like(t);ev=np.zeros(4);eg=np.zeros(4);tick=time.monotonic()
        for i in range(0,n,config['block_points']):
            guard();j=min(n,i+config['block_points'])
            (loss,parts),grad=calc(jnp.asarray(t),x[i:j],w[i:j],target[i:j],gradient[i:j],norms_v,norms_g,importance)
            value+=float(loss);derivative+=np.asarray(grad);ev+=np.asarray(parts[0]);eg+=np.asarray(parts[1])
        evaluations+=1
        if not np.isfinite(value) or not np.isfinite(derivative).all():raise FloatingPointError('nonfinite full objective')
        return value,derivative,{'value_relative_squared_by_mode':ev.tolist(),'gradient_relative_squared_by_mode':eg.tolist(),'seconds':time.monotonic()-tick}
    def save(t,f,g,details,label):
        nonlocal serial,best,last_checkpoint
        serial+=1;stem=f'checkpoint-{serial:05d}';arrays={'theta':t,'gradient':g}
        if phase in ('initial','adam'):arrays.update(adam_m=m,adam_v=v)
        np.savez_compressed(out/(stem+'.npz'),**arrays)
        record={'phase':phase,'label':label,'full_post_update_objective':f,'gradient_inf':float(np.max(np.abs(g))),'metrics':details,'adam_step':adam_step,'rng_state':rng.bit_generator.state,'curvature_state_pairs':list(curvature),'elapsed_seconds':prior_elapsed+time.monotonic()-started,'full_evaluations':evaluations,'theta_sha256':digest(t),'array_file':stem+'.npz','array_sha256':hashlib.sha256((out/(stem+'.npz')).read_bytes()).hexdigest()}
        (out/(stem+'.json')).write_text(json.dumps(record,indent=2)+'\n');last_checkpoint=stem
        if best is None or f<best['objective']:best={'objective':f,'checkpoint':stem}
        (out/'LATEST.json').write_text(json.dumps({'last':stem,'best':best,'elapsed_seconds':prior_elapsed+time.monotonic()-started})+'\n')
        return stem
    def selected_arrays(stem):
        path=out/(stem+'.npz');record=json.loads((out/(stem+'.json')).read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['array_sha256']:raise ValueError('selected checkpoint SHA differs')
        with np.load(path,allow_pickle=False) as arrays:chosen=arrays['theta'].copy()
        if digest(chosen)!=record['theta_sha256']:raise ValueError('selected parameter identity differs')
        return chosen
    status='started';f=None;g=np.zeros_like(theta);iteration=-1;pairs=[];current=None
    if resume is not None:
        from accuracy_v2.continuation import load
        restored,a,pairs=load(resume['directory'],bindings,resume['receipt_sha256'])
        theta=a['theta'];g=a['gradient'];m=a['adam_m'];v=a['adam_v']
        phase=restored['phase'];adam_step=restored['adam_step'];completed_lbfgs=restored['completed_lbfgs_iterations']
        prior_elapsed=restored['elapsed_seconds'];evaluations=restored['full_evaluations'];serial=restored['serial'];best=restored['best'];curvature=restored['curvature_state_pairs'];current=restored['last_measured_checkpoint'];last_checkpoint=current
        rng.bit_generator.state=restored['rng_state']
        if phase not in ('initial','adam','lbfgs') or adam_step>config['adam_steps'] or completed_lbfgs>config['lbfgs_steps']:raise ValueError('restart phase/count differs')
    try:
        f,g,details=full(theta);current=save(theta,f,g,details,'initial or exact restart')
        if phase!='lbfgs':phase='adam'
        for next_adam_step in range(adam_step+1,config['adam_steps']+1) if phase=='adam' else []:
            guard();idx=rng.integers(0,n,size=config['batch_points'])
            (_, _),grad=calc(jnp.asarray(theta),x[idx],w[idx]*n/len(idx),target[idx],gradient[idx],norms_v,norms_g,importance);grad=np.asarray(grad)
            if not np.isfinite(grad).all():raise FloatingPointError('nonfinite Adam gradient')
            m=.9*m+.1*grad;v=.999*v+.001*grad*grad
            rate=config['adam_learning_rate']*(.1+.9*.5*(1+np.cos(np.pi*next_adam_step/config['adam_steps'])))
            theta=theta-rate*(m/(1-.9**next_adam_step))/(np.sqrt(v/(1-.999**next_adam_step))+1e-8)
            adam_step=next_adam_step
            if adam_step%config['full_check_every']==0 or adam_step==config['adam_steps']:
                f,g,details=full(theta);current=save(theta,f,g,details,'Adam post-update')
        # Deterministic full-rule optimization starts at the best measured Adam iterate.
        if phase!='lbfgs':
            theta=selected_arrays(best['checkpoint'])
            phase='lbfgs';f,g,details=full(theta);current=save(theta,f,g,details,'L-BFGS initial');pairs=[];curvature=[]
        for iteration in range(completed_lbfgs,config['lbfgs_steps']):
            guard()
            if np.max(np.abs(g))<=config['gradient_tolerance']:status='gradient_stationary';break
            q=g.copy();alphas=[]
            for s,y,rho in reversed(pairs):
                alpha=rho*np.dot(s,q);alphas.append(alpha);q-=alpha*y
            if pairs:q*=np.dot(pairs[-1][0],pairs[-1][1])/np.dot(pairs[-1][1],pairs[-1][1])
            for (s,y,rho),alpha in zip(pairs,reversed(alphas)):q+=s*(alpha-rho*np.dot(y,q))
            direction=-q;slope=np.dot(g,direction)
            if not np.isfinite(slope) or slope>=0:direction=-g;slope=-np.dot(g,g);pairs=[];curvature=[]
            accepted=False
            for attempt in range(config['line_search_steps']):
                step=2.**(-attempt);trial=theta+step*direction;nf,ng,nd=full(trial)
                row={'iteration':iteration,'attempt':attempt,'step':step,'objective':nf,'gradient_inf':float(np.max(np.abs(ng))),'accepted':bool(nf<=f+1e-4*step*slope),'seconds':nd['seconds'],'theta_sha256':digest(trial)}
                with (out/'LINE-SEARCH.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                if row['accepted']:accepted=True;break
            if not accepted:status='line_search_stopped';break
            s=trial-theta;y=ng-g;sy=float(np.dot(s,y));previous=current
            theta,f,g=trial,nf,ng
            if sy>1e-12*np.linalg.norm(s)*np.linalg.norm(y):
                pairs.append((s,y,1/sy));curvature.append([previous,f'checkpoint-{serial+1:05d}']);pairs=pairs[-config['history_size']:];curvature=curvature[-config['history_size']:]
            current=save(theta,f,g,nd,'L-BFGS accepted');completed_lbfgs=iteration+1
        else:status='iteration_limit'
    except TimeoutError:status='job_deadline'
    except BaseException as exc:
        status='failed'
        try:(out/'FAILURE.json').write_text(json.dumps({'exception':repr(exc),'last_checkpoint':last_checkpoint})+'\n')
        except Exception as retention_error:retention_errors.append(repr(retention_error))
        raise
    finally:
        try:
            from accuracy_v2.continuation import freeze
            gradient_identity=None if last_checkpoint is None else json.loads((out/(last_checkpoint+'.json')).read_text())['theta_sha256']
            continuation={'theta_sha256':digest(theta),'gradient_theta_sha256':gradient_identity,'gradient_matches_theta':gradient_identity==digest(theta),'phase':phase,'adam_step':adam_step,'completed_lbfgs_iterations':completed_lbfgs,'curvature_state_pairs':curvature,'rng_state':rng.bit_generator.state,'last_measured_checkpoint':last_checkpoint,'best':best,'full_evaluations':evaluations,'serial':serial,'elapsed_seconds':prior_elapsed+time.monotonic()-started}
            freeze(out,continuation,{'theta':theta,'gradient':g,'adam_m':m,'adam_v':v})
        except Exception as retention_error:retention_errors.append(repr(retention_error))
        try:(out/'FINAL.json').write_text(json.dumps({'status':status,'last_checkpoint':last_checkpoint,'best':best,'elapsed_seconds':prior_elapsed+time.monotonic()-started,'full_evaluations':evaluations,'retention_errors':retention_errors,'automatic_resume':False,'validation_seen':False},indent=2)+'\n')
        except Exception as retention_error:
            retention_errors.append(repr(retention_error));print('FINAL retention failure:',repr(retention_error),flush=True)
        if retention_errors and status!='failed':raise RuntimeError('optimizer retention incomplete: '+repr(retention_errors))
    if best is None:raise RuntimeError('no full-rule measurement completed')
    chosen=selected_arrays(best['checkpoint'])
    layers=[(np.asarray(a),np.asarray(b)) for a,b in unpack(chosen)]
    bank=SpatialBank(layers,np.eye(4),dict(bindings,trained=bool(adam_step or completed_lbfgs),actual_applied_adam_steps=adam_step,accepted_lbfgs_steps=completed_lbfgs,selected_checkpoint=best,status=status))
    bank_sha=bank.save(out/'spatial-bank-raw.npz')
    (out/'MODEL-RECEIPT.json').write_text(json.dumps({'schema':'ns-obstacle-supervised-neural-model-v2','bank_file':'spatial-bank-raw.npz','bank_sha256':bank_sha,'selected_checkpoint':best,'selected_theta_sha256':digest(chosen),'config_sha256':hashlib.sha256((out/'CONFIG.json').read_bytes()).hexdigest(),'runtime_sha256':hashlib.sha256((out/'RUNTIME.json').read_bytes()).hexdigest(),'optimizer_status':status,'basis_role':'dimensionless normalized velocity basis; coefficients reconstruct u/U','ordering_applied':False,'physical_admission':False},indent=2)+'\n')
    return bank


def train(x,w,target,gradient,singular_values,qualification,config,directory,deadline,resume=None,teacher_identity=None):
    """Outer failure guard includes native/GPU initialization and checkpoint errors."""
    out=Path(directory)
    if out.exists():raise FileExistsError('fresh training output required')
    try:return _train(x,w,target,gradient,singular_values,qualification,config,directory,deadline,resume,teacher_identity)
    except BaseException as exc:
        if out.exists() and not (out/'FINAL.json').exists():
            try:(out/'FINAL.json').write_text(json.dumps({'status':'failed','original_exception':repr(exc),'validation_seen':False,'automatic_resume':False})+'\n')
            except Exception as error:print('Could not retain FINAL:',repr(error),flush=True)
        raise
