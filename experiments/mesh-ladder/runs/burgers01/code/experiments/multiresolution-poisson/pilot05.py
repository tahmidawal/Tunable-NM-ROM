"""Frozen-checkpoint source projection x training-only initialization factorial."""
import argparse,json,os,socket
from pathlib import Path
import scipy
from speed_core import *
from kernel_solver import solution_agreement
from pilot import reference,save
from pilot02 import score


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--selected-checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    dev=jax.devices()[0];assert dev.platform=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    models={};checkpoints=[]
    for tag,path in zip(cfg['models'],(a.checkpoint,a.selected_checkpoint)):
        digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==cfg['checkpoint_sha256'][tag]
        p,z,ckcfg=sc.load_pkl(path);assert (ckcfg['N'],ckcfg['k'],ckcfg['r'])==(256,16,64)
        models[tag]=(p,z);checkpoints.append(dict(model=tag,sha256=digest,training_count=len(z),config=ckcfg))
    draws=np.concatenate((source_params(cfg['existing_development_seed'],cfg['existing_development_count']),
        source_params(cfg['fresh_development_seed'],cfg['fresh_development_count'])))
    assert sha(draws)==cfg['parameter_sha256']
    groups=['existing_development']*cfg['existing_development_count']+['fresh_development']*cfg['fresh_development_count']
    d=dict(config=cfg,checkpoints=checkpoints,cohort=dict(parameters=draws.tolist(),parameter_sha256=sha(draws),groups=groups),
        provenance=dict(commit=os.environ['COMMIT'],job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),
            gpu=dev.device_kind,backend='gpu',x64=True,matmul_precision='highest',jax=jax.__version__,scipy=scipy.__version__),
        contract=dict(input='full host nodal source',output='full host nodal field and solver/initialization metadata',
            cache='training codes and decoder-predicted scaled weak coefficients only; no evaluation answer or Gaussian descriptors',
            stopping='tau times EACH initialization own initial residual; absolute threshold retained without re-anchoring',
            cost='full query including projection, lookup, guarded solver, decoding and transfers; fused device components grouped',
            stationarity='single-call tau0 controls excluded from speed selection; failed controls remain visible',
            parity='compare projection variants only at the SAME initialization choice; means versus nearest may reach different minima',
            reference='independently refined FD-DST common observations; empirical difference, not rigorous continuum bound',
            selection='checkpoint chosen from all pilot04 development cases before this job; no new training or sealed finals'),
        setup=[],references=[],coefficient_checks=[],projection_parity=[],warmup=[],rows=[],stationary_rows=[],complete=False)
    path=a.out/'result.json';persist=lambda:save(path,d);persist()
    refs=[];obs=cfg['observation_intervals'];start=time.perf_counter()
    for case,param in enumerate(draws):
        chain=[reference(param,n) for n in cfg['reference_intervals']]
        nested=[observation(u,n,obs) for u,n in zip(chain,cfg['reference_intervals'])]
        gaps=[relative(x,y) for x,y in zip(nested[:-1],nested[1:])];assert gaps[-1]<gaps[0]/2
        refs.append((nested[-1],gaps[-1]))
        np.savez_compressed(a.out/f'reference_case{case}.npz',observation=nested[-1],coarser_observation=nested[-2])
        d['references'].append(dict(case=case,group=groups[case],successive_relative_gaps=gaps,
            reference_delta=gaps[-1],uncertainty_kind='empirical_difference',intervals=cfg['reference_intervals']))
    d['reference_setup_seconds']=time.perf_counter()-start
    fields=a.out/'fields';fields.mkdir();stored=set()
    def store(field):
        digest=sha(field)
        if digest not in stored:np.savez_compressed(fields/(digest+'.npz'),field=field);stored.add(digest)
    for n in cfg['intervals']:
        opss={};caches={};kernels={}
        for tag,(p,z) in models.items():
            ops=assemble(p,z,n,cfg['requested_modes'],cfg['lm_budget']);opss[tag]=ops
            cache=weak_code_cache(ops,z);caches[tag]=cache
            ops['info'].update(model=tag,cache=cache['info']);d['setup'].append(ops['info'])
            np.savez_compressed(a.out/f'cache_{tag}_{n}.npz',codes=np.asarray(cache['codes']),predictions=np.asarray(cache['predictions']),B=np.asarray(ops['B']))
            for proj in cfg['projection']:
                for init in cfg['initialization']:
                    kernels[tag,proj,init]=make_speed_kernel(ops,cfg['lm_budget'],proj,init,cfg['linear_backward_error_limit'])
        lam=jnp.asarray(eigenvalues(n));coarselam=jnp.asarray(eigenvalues(cfg['coarse_intervals']))
        roms=[(tag,proj,init) for tag in models for proj in cfg['projection'] for init in cfg['initialization']]
        subjects=[('dst',None,None),('dst_coarse128',None,None)]+roms
        def query(subject,source,tau):
            tag,proj,init=subject
            if tag=='dst':return fom_query(source,lam)
            if tag=='dst_coarse128':return coarse_query(source,coarselam)
            return speed_query(source,opss[tag],caches[tag],tau,kernels[tag,proj,init],proj,init)
        for case,param in enumerate(draws):
            source=full_source(n,param);same=np.asarray(dst_solve(jnp.asarray(source),lam));gates={}
            for tag in models:
                check=projection_agreement(source,opss[tag],caches[tag])
                check.update(model=tag,case=case,intervals=n,
                    coefficient_passed=check['coefficient_relative']<=cfg['projection_relative_tolerance'])
                d['coefficient_checks'].append(check);gates[tag]=check
            warm=time.perf_counter()
            for subject in subjects:
                for _ in range(cfg['warmup']):query(subject,source,cfg['primary_tau'])
            for subject in roms:query(subject,source,0.)
            d['warmup'].append(dict(intervals=n,case=case,seconds=time.perf_counter()-warm))
            local=[];answers={}
            panels=[('primary',cfg['primary_tau'],cfg['repetitions'],subjects),('stationary',0.,1,roms)]
            for panel,tau,reps,choices in panels:
                for rep in range(reps):
                    order=choices if rep%2==0 else choices[::-1]
                    for ordinal,subject in enumerate(order):
                        tag,proj,init=subject;isrom=tag in models
                        burn(cfg['burn_seconds']);field,row=query(subject,source,tau)
                        row.update(score(field,same,*refs[case],n,obs))
                        stationary=not isrom or row['stationarity']<=cfg['stationarity_tolerance']
                        tau_ok=isrom and row['reason']==2
                        row.update(model=tag if isrom else None,arm='rom_speed' if isrom else tag,
                            projection=proj,initialization=init,intervals=n,case=case,group=groups[case],panel=panel,
                            requested_modes=cfg['requested_modes'] if isrom else None,
                            retained_modes=opss[tag]['info']['retained_modes'] if isrom else None,
                            repetition=rep,order_index=ordinal,tau=tau if isrom else None,source_sha256=sha(source),
                            stationary=stationary,tau_reached=tau_ok)
                        if isrom:
                            finite=bool(np.isfinite(field).all() and np.isfinite(row['latent']).all() and np.isfinite(row['residual']) and np.isfinite(row['stationarity']))
                            row['finite']=finite
                            answers[panel,tag,proj,init,rep]=(field,row)
                        local.append(row);store(field)
            # Every parity pair uses actual recorded calls at the same start choice.
            # A different nearest index is visible and fails this equality gate;
            # mean-versus-nearest differences never enter the projection gate.
            for row in local:
                if row['model'] is None:row.update(parity_passed=True,solver_valid=row['finite']);continue
                panel,tag,init,rep=row['panel'],row['model'],row['initialization'],row['repetition']
                thin=answers[panel,tag,cfg['projection'][0],init,rep];fft=answers[panel,tag,cfg['projection'][1],init,rep]
                agreement=solution_agreement(thin,fft,cfg['agreement_limits'])
                indices_match=thin[1]['selected_training_code_index']==fft[1]['selected_training_code_index']
                initial_relative=relative(np.asarray(thin[1]['initial_latent']),np.asarray(fft[1]['initial_latent']))
                gate=gates[tag]['coefficient_passed'] and indices_match and initial_relative<=cfg['initial_latent_relative_tolerance'] and agreement['passed']
                row.update(parity_passed=gate,numerical_agreement=agreement,
                    projection_indices_match=indices_match,initial_latent_projection_relative=initial_relative,
                    solver_valid=row['finite'] and gate and row['max_linear_backward_error']<=cfg['linear_backward_error_limit'] and (row['tau_reached'] or row['stationary']))
                if row['projection']==cfg['projection'][0]:
                    d['projection_parity'].append({**agreement,'intervals':n,'case':case,'model':tag,
                        'initialization':init,'panel':panel,'repetition':rep,'indices_match':indices_match,
                        'initial_latent_relative':initial_relative,'passed':gate})
            d['rows'].extend(r for r in local if r['panel']=='primary')
            d['stationary_rows'].extend(r for r in local if r['panel']=='stationary')
            persist();print('evaluation',n,case,groups[case],flush=True)
    assert len(d['rows'])==cfg['expected_timed_invocations']
    assert len(d['stationary_rows'])==cfg['expected_stationary_invocations']
    d['complete']=True;persist();print('ALL-DONE',flush=True)

if __name__=='__main__':main()
