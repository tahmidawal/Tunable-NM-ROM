"""Predeclared coarse screen and frozen-shortlist multiresolution confirmation."""
import argparse,json,os,socket,gc
from datetime import datetime,timezone
from pathlib import Path
from tuning_core import *
from tuning_summary import summarize,select
from iterative_core import make_cg,cg_query,verify_cg
from pilot import reference,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    if a.smoke:cfg.update(intervals=[64],reference_intervals=[64,128],repetitions=1,screen_repetitions=1,warmup=1,burn_seconds=.002,presets=[p for p in cfg['presets'] if p['id'] in ['nmrom_baseline','budget_1','stationarity_1e-02','multistart4_M256']])
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    p,z,ckcfg=sc.load_pkl(a.checkpoint);assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==cfg['checkpoint_sha256']
    draws=np.concatenate((source_params(cfg['existing_development_seed'],cfg['existing_development_count']),source_params(cfg['fresh_development_seed'],cfg['fresh_development_count'])))
    if not a.smoke:assert sha(draws)==cfg['parameter_sha256']
    if a.smoke:draws=draws[:1]
    presets={q['id']:q for q in cfg['presets']}
    d=dict(config=cfg,checkpoint_config=ckcfg,verification=verify_cg(),cohort=dict(parameters=draws.tolist(),parameter_sha256=sha(draws)),
        provenance=dict(commit=os.environ.get('COMMIT','local'),job_id=os.environ.get('SLURM_JOB_ID','local'),hostname=socket.gethostname(),gpu=jax.devices()[0].device_kind,backend='gpu',x64=True,matmul_precision='highest',jax=jax.__version__),
        contract=dict(weights='one unchanged original_relative checkpoint, k16/r64',resolution='same requested full source and full output mesh for every within-panel method',
          input='full nodal source',output='full nodal field plus online solver diagnostics',cost='paired source-to-field synchronized GPU and host total; post-timer stationarity audit excluded',
          reference='2048 FD-DST with empirical 1024 refinement; current-relative full-field L2',selection='all 30 development sources; no final cohort and no per-case truth-based setting/start choices',
          exact_algebra='offline weak operators and code cache refitted at each N and M; no quadrature sampling',stopping_reasons={'0':'budget exhausted','1':'small accepted improvement or step','2':'relative residual target','3':'damping ceiling','5':'nonfinite initial residual','6':'explicit configured stationarity threshold'}),
        setup=[],references=[],rows=[],warmup=[],complete=False)
    path=a.out/'result.json';persist=lambda:save(path,d);persist();(a.out/'fields').mkdir();(a.out/'references').mkdir();stored=set();refs={}
    for case,param in enumerate(draws):
        chain=[reference(param,n) for n in cfg['reference_intervals']]
        for n in sorted(set(cfg['intervals']+[cfg['screen_intervals']])):
            lo,hi=[u[::nr//n,::nr//n] for u,nr in zip(chain,cfg['reference_intervals'])];refs[n,case]=(hi.copy(),relative(lo,hi))
            np.savez_compressed(a.out/'references'/f'n{n}_case{case}.npz',fine=hi,coarse=lo)
            d['references'].append(dict(intervals=n,case=case,empirical_delta=relative(lo,hi),reference_intervals=cfg['reference_intervals']))
        print('reference',case,flush=True)
    persist()
    def run_panel(phase,n,preset_ids,reps):
        opses={};caches={};kernels={}
        for M in sorted({presets[name]['requested_modes'] for name in preset_ids}):
            ops=assemble(p,z,n,M,300);assert ops['info']['retained_modes']>=4*len(z[0]);cache=weak_code_cache(ops,z);opses[M]=ops;caches[M]=cache
            d['setup'].append({**ops['info'],'phase':phase,'cache':cache['info']});np.savez_compressed(a.out/f'cache_{phase}_n{n}_M{M}.npz',codes=np.asarray(cache['codes']),predictions=np.asarray(cache['predictions']),B=np.asarray(ops['B']))
        for name in preset_ids:
            preset=presets[name];ops=opses[preset['requested_modes']]
            kernels[name]=make_speed_kernel(ops,preset['budget'],cfg['projection'],cfg['initialization'],cfg['linear_backward_error_limit']) if name=='nmrom_baseline' else make_tuning_kernel(ops,preset,cfg['linear_backward_error_limit'])
        cg=make_cg(n,cfg['cg_maxiter']);lam=jnp.asarray(eigenvalues(n));methods=preset_ids+[f'cg_{t:.0e}' for t in cfg['cg_tolerances']]+['dst']
        def query(name,source):
            if name in presets:
                preset=presets[name];ops=opses[preset['requested_modes']];cache=caches[preset['requested_modes']]
                if name=='nmrom_baseline':
                    field,row=speed_query(source,ops,cache,preset['tau'],kernels[name],cfg['projection'],cfg['initialization']);row['stationary']=row['stationarity']<=cfg['stationarity_tolerance']
                    row['all_start_linear_valid']=row['max_linear_backward_error']<=cfg['linear_backward_error_limit'];row['solver_valid']=bool(np.isfinite(field).all() and row['all_start_linear_valid'] and (row['stationary'] or row['reason']==2))
                    fields=['latent','initial_latent','selected_training_code_index','residual','initial_residual','absolute_tau_threshold','reason','attempts','accepted','jacobians','stationarity','stationary','max_linear_backward_error','fallback_count','max_proposed_backward_error']
                    row.update(starts=[{k:row[k] for k in fields}],selected_start=0,total_attempts=row['attempts'],total_jacobians=row['jacobians'])
                else:field,row=tuning_query(source,ops,cache,preset,kernels[name],cfg)
                row.update(preset=preset,requested_modes=preset['requested_modes'],retained_modes=ops['info']['retained_modes'],operator_sha256=ops['info']['operator_sha256']);return field,row
            if name=='dst':
                field,row=fom_query(source,lam);row['fused_device_seconds']=row['solver_seconds'];row['solver_valid']=bool(np.isfinite(field).all());return field,row
            return cg_query(source,cg,float(name.removeprefix('cg_')))
        for case,param in enumerate(draws):
            source=full_source(n,param);same=reference(param,n);fine,delta=refs[n,case];warm=time.perf_counter()
            for name in methods:
                for _ in range(cfg['warmup']):query(name,source)
            d['warmup'].append(dict(phase=phase,intervals=n,case=case,seconds=time.perf_counter()-warm))
            for rep in range(reps):
                order=methods if rep%2==0 else methods[::-1]
                for ordinal,name in enumerate(order):
                    burn(cfg['burn_seconds']);field,row=query(name,source);digest=sha(field)
                    if digest not in stored:np.savez_compressed(a.out/'fields'/(digest+'.npz'),field=field);stored.add(digest)
                    error=relative(field,fine);stride=n//cfg['observation_intervals']
                    row.update(phase=phase,intervals=n,nodes_per_axis=n+1,case=case,repetition=rep,order_index=ordinal,method=name,source_sha256=sha(source),field_sha256=digest,physical_error=error,same_grid_error=relative(field,same),common_observation_error=relative(field[::stride,::stride],fine[::stride,::stride]),reference_delta=delta,conservative_physical_error=(error+delta)/(1-delta),finite=bool(np.isfinite(field).all()))
                    d['rows'].append(row)
            persist();print('evaluation',phase,n,case,flush=True)
        result=summarize([r for r in d['rows'] if r['phase']==phase and r['intervals']==n],cfg)
        del kernels,opses,caches,cg;gc.collect();jax.clear_caches();return result
    screen=run_panel('screen',cfg['screen_intervals'],list(presets),cfg['screen_repetitions']);selections=select(screen,list(presets));shortlist=['nmrom_baseline']
    for name in selections.values():
        if name is not None and name not in shortlist:shortlist.append(name)
    freeze=dict(screen_methods=screen,selections=selections,confirmed_preset_ids=shortlist,presets={name:presets[name] for name in shortlist},frozen_at=datetime.now(timezone.utc).isoformat(),selection_rules=cfg['selection_rules'],scope='coarse development selection; all meshes use this same shortlist, before any confirmation timing')
    freeze_path=a.out/'SELECTION.json';save(freeze_path,freeze);d['selection']=freeze;d['selection_sha256']=hashlib.sha256(freeze_path.read_bytes()).hexdigest();persist();print('FROZEN SELECTION',selections,flush=True)
    for n in cfg['intervals']:run_panel('confirmation',n,shortlist,cfg['repetitions'])
    assert len(d['rows'])==len(draws)*(cfg['screen_repetitions']*(len(presets)+len(cfg['cg_tolerances'])+1)+len(cfg['intervals'])*cfg['repetitions']*(len(shortlist)+len(cfg['cg_tolerances'])+1))
    d['complete']=True;persist();print('ONLINE TUNING COMPLETE',flush=True)


if __name__=='__main__':main()
