"""Matched expanded-data global nonlinear control; not a tree/PDE qualification."""
import argparse, traceback, optax
from common import *
from neural import stable as S, fields as F, code_optimizer as C, spatial_sampling as SS
from neural.build_data import resolve
from evaluate_residual_tree import load_group, owned_job
from training_io import checkpoint, read_checkpoint
import data


def extend_network(network, codes, latent, seed):
    """Zero added codes preserve the 32-code field; extra modulation rows can learn."""
    assert latent in (32,64) and codes.shape[1] == 32
    p=jax.tree.map(jnp.asarray,network)
    if latent==64:
        weight=p['modulation']['w']; assert weight.shape[0]==32
        extra=.01*jnp.std(weight)*jax.random.normal(jax.random.PRNGKey(seed),(32,weight.shape[1]),dtype=jnp.float64)
        p=dict(p,modulation=dict(p['modulation'],w=jnp.concatenate([weight,extra])))
        codes=jnp.pad(codes,((0,0),(0,32)))
    return p,codes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--latent',type=int,choices=[32,64],default=32)
    ap.add_argument('--qualification-job',default='b8implprofile')
    ap.add_argument('--updates',type=int,default=40000)
    ap.add_argument('--seed',type=int,default=2026111994)
    a=ap.parse_args(); assert a.updates==40000;owned_job(a.qualification_job)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
    result=dict(status='preparing',latent=a.latent,final_test_opened=False,
        representation_admitted=False,pde_admitted=False,history=[],
        role='Matched expanded-data global neural control; no claim that trees win',
        architecture='B7 tanh coordinate decoder, fixed spatial Fourier inputs, trainable additive latent shifts; no online POD lift',
        selection_scope='36 development fields only; neither all-case qualification nor actual4096 output')
    save(out/'result.json',result);begin=time.perf_counter()
    try:
        assert os.environ.get('SLURM_JOB_ID'),'Scientific training requires a cluster allocation'
        ns=Path(os.environ['STUDY_NAMESPACE']);assert str(ns).startswith('/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-')
        result['meta']=preflight()
        top=resolve(ns/a.qualification_job/'output/result.json',ns)
        qp=top.parent/'jacobian/result.json';q=json.loads(qp.read_text())
        assert q['status']=='complete_profile' and len(q['states'])==3
        derivative_checks=[row['parity'][name] for row in q['states'] for name in ['root_jacobian','residual_jacobian']]
        assert all(check['passed'] and check['max_absolute']<=1e-9 and check['relative']<=1e-10 for check in derivative_checks)
        manifest=HERE/'manifests'/f'retention-{a.qualification_job}.json';retained=json.loads(manifest.read_text())
        assert retained['local_and_remote_verified']
        hashes={r['file']:r['sha256'] for r in retained['files']}
        assert digest(top)==hashes['output/result.json']
        assert digest(qp)==hashes['output/jacobian/result.json']
        pp=resolve(ns/'b7parentrepair/output/stable_parent.pkl',ns);pr=json.loads((pp.parent/'result.json').read_text())
        assert digest(pp)==pr['selected_sha256'];parent=read_checkpoint(pp)
        dp=resolve(ns/'b6parent/output/coordinate_data.npz',ns);assert digest(dp)==parent['data_sha256']
        with np.load(dp) as v:arrays={k:np.array(v[k]) for k in v.files}
        tr=load_group(ns,'train',['b8trainrefs'],arrays,parent)
        dv=load_group(ns,'dev',['b8devrefs'],arrays,parent)
        assert len(tr['labels'])==160 and len(dv['labels'])==52
        assert not set(tr['labels'])&set(dv['labels'])
        xy,w=jnp.asarray(arrays['xy']),jnp.asarray(arrays['w'])
        basecodes=jnp.asarray(parent['training_codes']);assert basecodes.shape==(1408,32)
        old_encoder=jnp.asarray(parent['training_encoder']);assert old_encoder.shape==tr['encoder'][:1408].shape
        assert np.array_equal(np.asarray(old_encoder),np.asarray(tr['encoder'][:1408]))
        nearest=jax.jit(lambda field:jnp.argmin(jnp.mean((old_encoder-field[None])**2,axis=1)))
        ids=np.asarray([int(nearest(field)) for field in tr['encoder'][1408:]],dtype=np.int64)
        codes=jnp.concatenate([basecodes,basecodes[ids]])
        network,codes=extend_network(parent['network'],codes,a.latent,a.seed)
        probe=xy[::1024];z=basecodes[:3]
        original=S.apply(parent['network'],z[:,None,:],probe[None])
        extended=S.apply(network,codes[:3,None,:],probe[None])
        preservation=float(jnp.max(abs(original-extended)));assert preservation<=1e-14
        forward=jax.jacfwd(S.apply,argnums=1)(network,codes[0],probe)
        reverse=S.code_jacobian(network,codes[0],probe)
        jacdiff=float(jnp.max(abs(forward-reverse)))
        jacrel=jacdiff/max(float(jnp.max(abs(forward))),1e-30)
        assert jacdiff<=1e-9 and jacrel<=1e-10
        rng=np.random.default_rng(a.seed)
        strata=SS.stratify(data.parameters('train',128),tr['family'][128:])
        # Immutable training-only proposal. Staleness affects variance, not expectation.
        probabilities=np.empty((len(codes),len(w)),np.float64);cdf=np.empty_like(probabilities)
        predict=jax.jit(lambda p,z:S.apply(p,z[:,None,:],xy[None]))
        for lo in range(0,len(codes),4):
            residual=np.asarray(tr['y'][lo:lo+4]-predict(network,codes[lo:lo+4]))
            part=SS.prepare(np.asarray(w),residual,.5)
            probabilities[lo:lo+4]=part['probabilities'];cdf[lo:lo+4]=part['cdf']
        sampler=dict(weights=np.asarray(w),probabilities=probabilities,cdf=cdf,cohort='train',fields=len(codes))
        rate=optax.cosine_decay_schedule(3e-3,a.updates,alpha=.1)
        netopt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(optax.cosine_decay_schedule(1e-4,a.updates,alpha=.05)))
        no=netopt.init(network);co=C.initialize(codes)
        @jax.jit
        def step(p,c,no,co,si,ci,importance,it):
            def loss(p,selected):
                pred=S.apply(p,selected[:,None,:],xy[ci]);target=tr['y'][si[:,None],ci]
                return jnp.mean(importance*((pred-target)/tr['norm'][si,None])**2)
            value,(gp,gz)=jax.value_and_grad(loss,argnums=(0,1))(p,c[si])
            update,no=netopt.update(gp,no,p);c,co=C.update(c,co,si,gz,rate(it))
            return optax.apply_updates(p,update),c,no,co,value
        fit=F.make_fit(S.apply,60,jacobian=S.code_jacobian)
        more=F.make_fit(S.apply,120,jacobian=S.code_jacobian)
        selection=[11*c+t for c in [0,3,13,20]+list(range(44,52)) for t in [0,5,10]]
        xf,wf=data.grid(256,True);xf,wf=jnp.asarray(xf),jnp.asarray(wf)
        fine={}
        for ci in sorted({i//11 for i in selection}):
            source=dv['sources'][ci];path=resolve(source['file'],ns);assert digest(path)==source['sha256']
            with np.load(path) as v:
                assert np.array_equal(v['times'],data.TIMES)
                fine[ci]=data.synth_grid(jnp.asarray(v['coefficients']),jnp.unique(xf[:,0])).reshape(11,-1)
        def finish(p,z,group,i):
            z,st=fit(p,z,xy,group['y'][i],w,group['norm'][i]);attempts=[np.asarray(st)]
            if not bool(st[-1]):
                z,st=more(p,z,xy,group['y'][i],w,group['norm'][i]);attempts.append(np.asarray(st))
            return z,st,attempts
        evaluation_serial=0
        def evaluate(p,c):
            nonlocal evaluation_serial
            evaluation_serial+=1
            rows=[]
            for i in selection:
                near,interp=S.nearest_starts(dv['encoder'][i],tr['encoder'],c)
                fits=[fit(p,start,xy,dv['y'][i],w,dv['norm'][i]) for start in [near,interp,jnp.zeros(a.latent)]]
                selected=min(range(3),key=lambda j:float(fits[j][1][0]));z,st=fits[selected]
                attempts=[dict(start=name,status=np.asarray(candidate[1])) for name,candidate in zip(['nearest','interpolated','zero'],fits)]
                if not bool(st[-1]):
                    z,st=more(p,z,xy,dv['y'][i],w,dv['norm'][i]);attempts.append(dict(start='selected_continuation',status=np.asarray(st)))
                ci,ti=divmod(i,11)
                pred=S.apply(p,z,xf);error=float(jnp.sqrt(jnp.sum(wf*(pred-fine[ci][ti])**2))/dv['norm'][i])
                rows.append(dict(index=i,case=dv['labels'][ci],time=float(data.TIMES[ti]),codes=np.asarray(z),
                    status=np.asarray(st),attempts=attempts,selected_start=selected,independent_error=error,
                    scoring_change=abs(error-float(st[0])),converged=bool(st[-1]) and np.isfinite(np.asarray(st)).all().item()))
                if len(rows)%6==0:
                    checkpoint(out/f'panel_partial_{evaluation_serial:02d}.pkl',dict(rows=rows,complete=len(rows)==len(selection),
                        expected=len(selection),role='Offline development selection fits; no all-case admission'))
            return rows
        result.update(status='training',parent_sha256=digest(pp),primary_data_sha256=digest(dp),
            jacobian_qualification_sha256=digest(qp),qualification_wrapper_sha256=digest(top),jacobian_retention_sha256=digest(manifest),
            derivative_qualification_checks=derivative_checks,overall_pde_trace_parity_passed=q['parity_passed'],
            qualification_scope='Root/residual derivative values only; overall strict PDE iteration/acceptance trace gate is not a training prerequisite and is not claimed passed',
            stable_module_sha256=digest(HERE/'neural/stable.py'),
            expanded_sources=tr['provenance']+dv['provenance'],train_labels=tr['labels'],dev_labels=dv['labels'],
            initialization='B7 weights/primary codes; each expanded snapshot copies nearest original training-field code; no extrapolated unfinished root fits',
            initial_field_preservation_max_abs=preservation,
            initial_latent_jacobian_parity=dict(max_absolute=jacdiff,relative=jacrel),
            extra_modulation_rows='For64: 0.01 times B7 modulation standard deviation times seeded normal, zero extra codes',
            training=dict(updates=40000,network_lr=1e-4,code_lr=3e-3,batch=8,points=1024,
                finishing_updates=2000,finishing_batch=4,finishing_points=len(w),per_row_adam=True,
                family_probabilities=strata['probabilities'],family_counts=strata['counts'],
                objective='Equal five-family expectation of current-time normalized physical squared L2; not uniform trajectory expectation',
                spatial_sampling='Fixed initial training residual mixture alpha0.5; unbiased w/p importance, final full quadrature'),
            selection_indices=selection,selection_metric='Worst independent256 physical error on36 development fields; strict fit/scoring flags reported separately')
        result['initialization_sha256']=checkpoint(out/'initialization.pkl',dict(network=network,training_codes=codes,
            extension_nearest_primary_indices=ids,primary_data_sha256=digest(dp),parent_sha256=digest(pp),strata=strata))
        save(out/'result.json',result);best=np.inf
        family_draws={name:0 for name in strata['probabilities']}
        for it in range(1,40001):
            full=it>38000;batch=4 if full else 8
            family,caseids=SS.draw_stratum(rng,strata,batch);family_draws[family]+=1
            si=11*caseids+rng.integers(11,size=batch)
            if full:
                ci=np.broadcast_to(np.arange(len(w)),(batch,len(w)))
                importance=np.broadcast_to(np.asarray(w)*len(w),ci.shape)
            else:
                sample=SS.draw(rng,sampler,si,1024);ci=sample['indices'];importance=sample['importance_weights']
            network,codes,no,co,loss=step(network,codes,no,co,jnp.asarray(si),jnp.asarray(ci),jnp.asarray(importance),it-1)
            if it%1000==0:
                assert np.isfinite(float(loss)), 'Nonfinite training loss'
                print('expanded_global_training',it,float(loss),family,full,flush=True)
            if it%5000:continue
            # Save before slow evaluation; interrupted evaluations preserve trained state.
            payload=dict(network=network,training_codes=codes,training_encoder=tr['encoder'],pca=parent.get('pca'),
                data_sha256=digest(dp),data_record=parent['data_record'],expanded_sources=result['expanded_sources'],
                train_labels=tr['labels'],dev_labels=dv['labels'],latent=a.latent,update=it,
                optimizer=dict(network=no,codes=co),rng_state=rng.bit_generator.state)
            path=out/f'parent_{it:05d}.pkl';rawhash=checkpoint(path,payload)
            ev=evaluate(network,codes)
            row=dict(update=it,loss=float(loss),full_quadrature=full,
                worst_independent_error=max(r['independent_error'] for r in ev),
                median_independent_error=float(np.median([r['independent_error'] for r in ev])),
                unfinished=sum(not r['converged'] for r in ev),max_scoring_change=max(r['scoring_change'] for r in ev),
                family_draws=dict(family_draws),pre_evaluation_checkpoint_sha256=rawhash)
            payload.update(evaluation=ev,metrics=row)
            row['checkpoint_sha256']=checkpoint(out/f'evaluated_parent_{it:05d}.pkl',payload)
            if row['worst_independent_error']<best:
                best=row['worst_independent_error'];result['selected_update']=it
                result['selected_sha256']=checkpoint(out/'selected_before_refresh.pkl',payload)
            result['history'].append(row);save(out/'result.json',result)
            print('expanded_global_selection',row,flush=True)
        selected=read_checkpoint(out/'selected_before_refresh.pkl');network=jax.tree.map(jnp.asarray,selected['network']);codes=jnp.asarray(selected['training_codes'])
        refreshed=[];fitrows=[]
        for i in range(len(codes)):
            z,st,attempts=finish(network,codes[i],tr,i);refreshed.append(np.asarray(z))
            fitrows.append(dict(index=i,case=tr['labels'][i//11],time=float(data.TIMES[i%11]),status=np.asarray(st),attempts=attempts))
            if (i+1)%32==0 or i+1==len(codes):
                checkpoint(out/'training_refresh_partial.pkl',dict(completed=i+1,expected=len(codes),
                    codes=np.stack(refreshed),records=fitrows,network_sha256=result['selected_sha256']))
                print('expanded_global_final_refresh',i+1,len(codes),float(st[0]),bool(st[-1]),flush=True)
        codes=jnp.asarray(np.stack(refreshed));ev=evaluate(network,codes)
        selected.update(training_codes=codes,evaluation=ev,training_fit_records=fitrows,
            optimizer=None,optimizer_note='Code refresh invalidates saved Adam state; no exact continuation from this artifact')
        result['selected_sha256']=checkpoint(out/'stable_parent.pkl',selected)
        result.update(status='complete_expanded_global_selection',
            worst_independent_error=max(r['independent_error'] for r in ev),
            unfinished_fits=sum(not r['converged'] for r in ev),max_scoring_change=max(r['scoring_change'] for r in ev),
            worst_training_error=max(float(r['status'][0]) for r in fitrows),
            unfinished_training_fits=sum(not bool(r['status'][-1]) for r in fitrows))
        result['selection_numerical_gate_pass']=result['worst_independent_error']<=.005 and result['max_scoring_change']<=.0002 and result['unfinished_fits']==0
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_wall_seconds']=time.perf_counter()-begin;save(out/'result.json',result)


if __name__=='__main__':main()
