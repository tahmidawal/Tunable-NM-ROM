"""Current frozen Poisson NMROM versus original iterative CG, one GPU ladder."""
import argparse,json,os,socket,gc
from pathlib import Path
from speed_core import *
from iterative_core import make_cg,cg_query,verify_cg
from pilot import reference,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cfg=json.loads(args.config.read_text());args.out.mkdir(parents=True,exist_ok=False)
    if args.smoke:cfg.update(intervals=[64],reference_intervals=[64,128],repetitions=1,warmup=1,burn_seconds=.02)
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    p,z,ckcfg=sc.load_pkl(args.checkpoint)
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==cfg['checkpoint_sha256']
    draws=np.concatenate((source_params(cfg['existing_development_seed'],cfg['existing_development_count']),
        source_params(cfg['fresh_development_seed'],cfg['fresh_development_count'])))
    assert sha(draws)==cfg['parameter_sha256']
    if args.smoke:draws=draws[:1]
    d=dict(config=cfg,checkpoint_config=ckcfg,verification=verify_cg(),cohort=dict(parameters=draws.tolist(),parameter_sha256=sha(draws)),
        provenance=dict(commit=os.environ.get('COMMIT','local'),job_id=os.environ.get('SLURM_JOB_ID','local'),
            hostname=socket.gethostname(),gpu=jax.devices()[0].device_kind,backend='gpu',x64=True,matmul_precision='highest',jax=jax.__version__),
        contract=dict(input='full supplied nodal source',output='full nodal solution and solver diagnostics',
            primary_cost='synchronized GPU input through source projection/initialization/solve/full GPU output',
            host_cost='same invocation including full input and output transfers plus diagnostics',
            baseline='unpreconditioned CG, zero start, same grid, compiled loop, true residual charged',
            error='full requested mesh relative L2 against restricted 2048-interval FD-DST reference',
            reference='empirical 1024/2048 refinement, not rigorous continuum bound',
            cohort='all existing development sources; no final cohort, retraining or censoring'),setup=[],references=[],rows=[],warmup=[],complete=False)
    path=args.out/'result.json';persist=lambda:save(path,d);persist()
    (args.out/'fields').mkdir();(args.out/'references').mkdir();stored=set()
    refs={};start=time.perf_counter()
    for case,param in enumerate(draws):
        chain=[reference(param,n) for n in cfg['reference_intervals']]
        for n in cfg['intervals']:
            lo,hi=[u[::nr//n,::nr//n] for u,nr in zip(chain,cfg['reference_intervals'])]
            refs[n,case]=(hi.copy(),relative(lo,hi))
            np.savez_compressed(args.out/'references'/f'n{n}_case{case}.npz',fine=hi,coarse=lo)
            d['references'].append(dict(intervals=n,case=case,empirical_delta=relative(lo,hi),reference_intervals=cfg['reference_intervals']))
        print('reference',case,flush=True)
    d['reference_setup_seconds']=time.perf_counter()-start;persist()
    for n in cfg['intervals']:
        ops=assemble(p,z,n,cfg['requested_modes'],cfg['lm_budget']);cache=weak_code_cache(ops,z)
        d['setup'].append({**ops['info'],'cache':cache['info']})
        np.savez_compressed(args.out/f'cache_n{n}.npz',codes=np.asarray(cache['codes']),predictions=np.asarray(cache['predictions']),B=np.asarray(ops['B']))
        kernel=make_speed_kernel(ops,cfg['lm_budget'],cfg['projection'],cfg['initialization'],cfg['linear_backward_error_limit'])
        cg=make_cg(n,cfg['cg_maxiter']);lam=jnp.asarray(eigenvalues(n))
        methods=['nmrom']+[f'cg_{tol:.0e}' for tol in cfg['cg_tolerances']]+['dst']
        def query(name,source):
            if name=='nmrom':
                field,row=speed_query(source,ops,cache,cfg['tau'],kernel,cfg['projection'],cfg['initialization'])
                row['stationary']=row['stationarity']<=cfg['stationarity_tolerance']
                row['solver_valid']=bool(np.isfinite(field).all() and row['max_linear_backward_error']<=cfg['linear_backward_error_limit'] and (row['stationary'] or row['reason']==2))
                return field,row
            if name=='dst':
                field,row=fom_query(source,lam);row['fused_device_seconds']=row['solver_seconds'];row['solver_valid']=bool(np.isfinite(field).all())
                return field,row
            return cg_query(source,cg,float(name.removeprefix('cg_')))
        for case,param in enumerate(draws):
            source=full_source(n,param);same=reference(param,n);fine,delta=refs[n,case]
            warm=time.perf_counter()
            for name in methods:
                for _ in range(cfg['warmup']):query(name,source)
            d['warmup'].append(dict(intervals=n,case=case,seconds=time.perf_counter()-warm))
            for rep in range(cfg['repetitions']):
                order=methods if rep%2==0 else methods[::-1]
                for ordinal,name in enumerate(order):
                    burn(cfg['burn_seconds']);field,row=query(name,source)
                    digest=sha(field)
                    if digest not in stored:
                        np.savez_compressed(args.out/'fields'/(digest+'.npz'),field=field);stored.add(digest)
                    error=relative(field,fine);obs=cfg['observation_intervals']
                    row.update(intervals=n,nodes_per_axis=n+1,case=case,repetition=rep,order_index=ordinal,method=name,
                        source_sha256=sha(source),field_sha256=digest,physical_error=error,same_grid_error=relative(field,same),
                        common_observation_error=relative(field[::n//obs,::n//obs],fine[::n//obs,::n//obs]),
                        reference_delta=delta,conservative_physical_error=(error+delta)/(1-delta),finite=bool(np.isfinite(field).all()))
                    d['rows'].append(row)
            persist();print('evaluation',n,case,flush=True)
        del ops,cache,kernel,cg;gc.collect()
    assert len(d['rows'])==len(draws)*len(cfg['intervals'])*cfg['repetitions']*(2+len(cfg['cg_tolerances']))
    d['complete']=True;persist();print('ITERATIVE POISSON COMPLETE',flush=True)


if __name__=='__main__':main()
