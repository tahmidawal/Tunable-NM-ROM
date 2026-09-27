"""Complete-query test of a guarded tiny SPD solver on frozen Poisson weights."""
import argparse,json,os,socket
from pathlib import Path
import scipy
from kernel_solver import *
from followup import fused_kernel,fused_query,parity
from pilot import reference,save
from pilot02 import score
from test_kernel import synthetic_checks


def trace_audit(solver,ops,source,tau,path):
    fm=ops['project'](jnp.asarray(source),ops['S'],ops['I'],ops['J'],ops['W'])
    ans,stats,trace=solver(ops['z0'],fm,jnp.asarray(tau));ans,stats,trace=jax.device_get((ans,stats,trace))
    count=int(ans[5]);A,b,x,lam,eta,fallback,z,accept=[np.asarray(a)[:count] for a in trace]
    conditions=[];backwards=[];forwards=[];eigmins=[]
    for aa,bb,xx in zip(A,b,x):
        ref=np.linalg.solve(aa,bb)
        conditions.append(float(np.linalg.cond(aa)))
        backwards.append(float(np.linalg.norm(aa@xx-bb)/(np.linalg.norm(aa)*np.linalg.norm(xx)+np.linalg.norm(bb)+1e-300)))
        forwards.append(relative(xx,ref));eigmins.append(float(np.linalg.eigvalsh(aa).min()))
    np.savez_compressed(path,A=A,b=b,step=x,damping=lam,backward=eta,fallback=fallback,z_before=z,accepted=accept,final_z=ans[0])
    return dict(attempts=count,condition_numbers=conditions,condition_max=max(conditions,default=0.),
        minimum_eigenvalues=eigmins,backward_errors=backwards,backward_max=max(backwards,default=0.),
        cpu_solve_relative_differences=forwards,cpu_solve_difference_max=max(forwards,default=0.),
        fallback_count=int(stats[1]),reason=int(ans[6]),accepted=int(ans[4]),jacobians=int(ans[3]),
        latent=ans[0].tolist(),residual=float(ans[1]),initial_residual=float(ans[2]),
        artifact=path.name,scope='Untimed deterministic trajectory replay; matched to timed output separately')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());args.out.mkdir(parents=True,exist_ok=False)
    dev=jax.devices()[0];assert dev.platform=='gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    params,codes,ckcfg=sc.load_pkl(args.checkpoint);assert ckcfg['k']==16 and ckcfg['r']==64 and ckcfg['N']==256
    draws=source_params(cfg['seed'],cfg['cohort_count']);obs=cfg['observation_intervals']
    d=dict(config=cfg,checkpoint_config=ckcfg,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        provenance=dict(commit=os.environ.get('COMMIT','local'),job_id=os.environ.get('SLURM_JOB_ID'),
            hostname=socket.gethostname(),gpu=dev.device_kind,backend='gpu',x64=True,matmul_precision='highest',jax=jax.__version__,scipy=scipy.__version__),
        cohort=dict(seed=cfg['seed'],count=len(draws),parameters=draws.tolist(),parameter_sha256=sha(draws),kind='same_development_sources_as_pilot01_and_pilot02'),
        contract=dict(input='full host source nodal array',output='full host solution nodal array and solver counters',
            physical_metric='relative L2 against refined reference at fixed common observation nodes',
            specialized='symmetrically diagonal-scaled unrolled Gauss-Jordan; online backward-error check and generic fallback charged',
            original='untouched ctol_tol.lm_tau_poisson; original modular and fused complete queries',
            solver_controls='same objective, initial code, damping, acceptance, trust and stopping rules; floating-point trajectories may differ',
            reference='empirical nested SciPy FD-DST differences, no rigorous bound',
            trajectory='saved normal systems from untimed replays; matched against each timed solver output; not used for timing or cost/error pairing',
            aggregation=cfg['speedup_statistic']),
        synthetic_controls=synthetic_checks(cfg),references=[],setup=[],parity=[],trajectories=[],rows=[],complete=False)
    out=args.out/'result.json';persist=lambda:save(out,d);persist()
    refs=[]
    for case,param in enumerate(draws):
        chain=[reference(param,n) for n in cfg['reference_intervals']]
        nested=[observation(u,n,obs) for u,n in zip(chain,cfg['reference_intervals'])]
        gaps=[relative(a,b) for a,b in zip(nested[:-1],nested[1:])];delta=gaps[-1]
        assert gaps[-1]<gaps[0]/2
        refs.append((nested[-1],delta));np.savez_compressed(args.out/f'reference_case{case}.npz',observation=nested[-1],coarser_observation=nested[-2])
        d['references'].append(dict(case=case,intervals=cfg['reference_intervals'],successive_relative_gaps=gaps,reference_delta=delta,uncertainty_kind='empirical_difference',reference_converging=True));persist()
    for n in cfg['intervals']:
        print('assemble',n,flush=True)
        opss={m:assemble(params,codes,n,m,cfg['lm_budget']) for m in cfg['requested_modes_ladder']}
        original={m:fused_kernel(op) for m,op in opss.items()}
        special={m:specialized_kernel(op,make_lm_kernel(op,cfg['lm_budget'],True,False,cfg['linear_backward_limit'])) for m,op in opss.items()}
        trace_generic={m:make_lm_kernel(op,cfg['lm_budget'],False,True,cfg['linear_backward_limit']) for m,op in opss.items()}
        trace_special={m:make_lm_kernel(op,cfg['lm_budget'],True,True,cfg['linear_backward_limit']) for m,op in opss.items()}
        for op in opss.values():d['setup'].append(op['info'])
        lam=jnp.asarray(eigenvalues(n));coarse_lam=jnp.asarray(eigenvalues(128))
        def query(arm,m,tau,src):
            if arm=='dst':return fom_query(src,lam)
            if arm=='dst_coarse128':return coarse_query(src,coarse_lam)
            if arm=='rom_modular':return rom_query(src,opss[m],tau)
            if arm=='rom_fused':return fused_query(src,opss[m],tau,original[m])
            return specialized_query(src,opss[m],tau,special[m])
        subjects=[('dst',None,None),('dst_coarse128',None,None)]+[(arm,m,tau) for m in cfg['requested_modes_ladder'] for tau in cfg['taus'] for arm in ('rom_modular','rom_fused','rom_gj')]
        for case,param in enumerate(draws):
            source=full_source(n,param);same=np.asarray(dst_solve(jnp.asarray(source),lam));fom_parity=relative(same,reference(param,n));assert fom_parity<1e-11
            controls={};gates={};traces={}
            for m in cfg['requested_modes_ladder']:
                for tau in cfg['taus']:
                    control=query('rom_modular',m,tau,source);fused=query('rom_fused',m,tau,source);candidate=query('rom_gj',m,tau,source)
                    exact=parity(control,fused);assert exact['passed']
                    agreement=solution_agreement(control,candidate,cfg['agreement_limits'])
                    agreement.update(intervals=n,case=case,requested_modes=m,tau=tau,original_fusion=exact)
                    d['parity'].append(agreement);controls[(m,tau)]=control;gates[(m,tau)]=agreement['passed']
                    for arm,solver in [('generic',trace_generic[m]),('gj',trace_special[m])]:
                        record=trace_audit(solver,opss[m],source,tau,args.out/f'trace_n{n}_case{case}_M{m}_tau{tau}_{arm}.npz')
                        expected=control if arm=='generic' else candidate
                        record.update(intervals=n,case=case,requested_modes=m,tau=tau,linear_solver=arm,
                            replay_latent_relative=relative(record['latent'],expected[1]['latent']),
                            replay_counter_agreement=all(record[k]==expected[1][k] for k in ('reason','accepted','attempts','jacobians')))
                        d['trajectories'].append(record);traces[(m,tau,arm)]=record
                        if record['replay_latent_relative']>cfg['agreement_limits']['latent_relative']:gates[(m,tau)]=False
                    if not gates[(m,tau)]:print('AGREEMENT-FAILED',agreement,flush=True)
            warm=time.perf_counter()
            for arm,m,tau in subjects:
                for _ in range(cfg['warmup']):query(arm,m,tau,source)
            d.setdefault('warmup',[]).append(dict(intervals=n,case=case,seconds=time.perf_counter()-warm))
            for rep in range(cfg['repetitions']):
                order=subjects if rep%2==0 else subjects[::-1]
                for ordinal,(arm,m,tau) in enumerate(order):
                    burn(cfg['burn_seconds']);field,row=query(arm,m,tau,source)
                    row.update(score(field,same,*refs[case],n,obs));isrom=arm.startswith('rom')
                    stationary=not isrom or row['stationarity']<=cfg['stationarity_tolerance'];tau_ok=isrom and row['reason']==2
                    gate=True
                    if arm=='rom_gj':
                        inv=solution_agreement(controls[(m,tau)],(field,row),cfg['agreement_limits'])
                        gate=gates[(m,tau)] and inv['passed'] and row['max_linear_backward_error']<=cfg['linear_backward_limit']
                        row['numerical_agreement']=inv
                    if isrom:
                        fm=opss[m]['project'](jnp.asarray(source),opss[m]['S'],opss[m]['I'],opss[m]['J'],opss[m]['W'])
                        z=jnp.asarray(row['latent']);fun=lambda z:opss[m]['B']@sc.head(params,z)-fm
                        residual=fun(z);jac=jax.jacfwd(fun)(z)
                        row['gradient_norm']=float(jnp.linalg.norm(jac.T@residual));row['jacobian_frobenius']=float(jnp.linalg.norm(jac))
                    row.update(arm=arm,requested_modes=m,retained_modes=opss[m]['info']['retained_modes'] if m else None,
                        intervals=n,case=case,repetition=rep,order_index=ordinal,tau=tau,source_sha256=sha(source),
                        scipy_dst_parity=fom_parity,stationary=stationary,tau_reached=tau_ok,parity_passed=gate,
                        solver_valid=row['finite'] and gate and (not isrom or tau_ok or stationary))
                    d['rows'].append(row)
                    if rep==0:np.savez_compressed(args.out/f'field_n{n}_case{case}_{arm}_M{m}_tau{tau}.npz',field=field)
                persist()
            print('finished',n,case,'specialized fallbacks',sum(r.get('fallback_count',0) for r in d['rows'] if r['intervals']==n and r['case']==case),flush=True)
    d['complete']=True;persist();print('ALL-DONE',flush=True)

if __name__=='__main__':main()
