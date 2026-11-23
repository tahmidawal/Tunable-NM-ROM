"""Focused untouched NS trajectories with qualified actual initializer calls."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import support as S
import legacy_phase as P
import fit_api as A
import frozen_contract as FC
np,jax,jnp,H,F,SR=S.np,S.jax,S.jnp,S.H,S.F,S.SR
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());frozen,registration=FC.verify(cfg,args.config)
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    args.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    report={'schema':'ns-repaired-fresh-phase-v1','config':cfg,'frozen_protocol':frozen,
        'registration':registration,'source_commit':os.environ.get('SOURCE_COMMIT'),
        'job_id':os.environ.get('SLURM_JOB_ID'),'backend':jax.default_backend(),
        'x64':jax.config.jax_enable_x64,'matmul_precision':str(jax.config.jax_default_matmul_precision),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader'],text=True).strip(),
        'cases':[],'optional_diagnostics':{'manifold_distance':'not run','physical_tangent':'not run','same_state_reset':'not run'}}
    def save():report['elapsed_diagnostic_seconds']=time.perf_counter()-start;S.dump(args.out/'summary.json',report)
    save();n=cfg['n'];k=cfg['k'];dt=cfg['dt'];steps=round(cfg['horizon']/dt)
    G,bank=S.learned_basis(n);basis=jnp.asarray(G);report['bank']=bank
    report['basis_sha256']=hashlib.sha256(np.asarray(G).tobytes()).hexdigest()
    params,codes=S.load_head(HERE/'assets/head_k8_n64.npz');candidates=np.asarray(H.head_apply(params,jnp.asarray(codes)))
    opsall,checks=SR.build_operators_fast(G,n,cfg['expanded_rows'],check_modes=8)
    report['operator_checks']=checks
    operator_pass=bool(max(checks[name] for name in ('A_relative','derivative_relative','diffusion_relative',
        'tensor_relative','subset_test_orthogonality'))<=cfg['qualification']['operator'] and checks['shift_sign_relative']<=1e-5)
    report['operator_pass']=operator_pass
    np.savez_compressed(args.out/'operator_arrays.npz',**{name:opsall[name] for name in ('A','T','lam','Dd')})
    geom=F.geometry(n);truth_dt=cfg['reference']['dt'];fine_dt=cfg['reference']['check_dt']
    truth_run=F.make_solver(truth_dt,round(cfg['horizon']/truth_dt),round(dt/truth_dt),scheme='cnab2')
    fine_run=F.make_solver(fine_dt,round(cfg['horizon']/fine_dt),round(dt/fine_dt),scheme='cnab2')
    values=FC.parameters(cfg,registration)
    for ci,par in enumerate(values):
        directory=args.out/f'case_{ci:03d}';directory.mkdir()
        initial=F.initial(n,par);norm=float(np.linalg.norm(initial))
        initial_fields={};initial_records={}
        for fit_tol in cfg['fit_gtols']:
            field,record=A.fit_initial_field(initial,basis,params,codes,candidates,fit_tol,normalization=norm)
            A.save_invocation(directory/f'initial_g{fit_tol:g}',record,params)
            initial_fields[fit_tol]=field;initial_records[fit_tol]=record
        nominal_fit=initial_records[max(cfg['fit_gtols'])];tight_fit=initial_records[min(cfg['fit_gtols'])]
        ic_gap=float(np.linalg.norm(initial_fields[max(cfg['fit_gtols'])]-initial_fields[min(cfg['fit_gtols'])])/norm)
        init_qualified=all(r['qualified'] for r in initial_records.values()) and ic_gap<=cfg['qualification']['initial_agreement']
        # Reuse the exact selected code/coefficient/frame of this actual initializer invocation.
        z0=np.asarray(tight_fit['code']);ahead=np.asarray(tight_fit['coefficients'])
        center=np.asarray(tight_fit['center']);projected=np.asarray(tight_fit['target'])
        parity=float(np.linalg.norm(np.asarray(P.reconstruct(basis,jnp.asarray(ahead),jnp.asarray(center)))-initial_fields[min(cfg['fit_gtols'])])/norm)
        truth=np.asarray(truth_run(jnp.asarray(initial),float(par[-1]),geom))
        fine=np.asarray(fine_run(jnp.asarray(initial),float(par[-1]),geom))
        ref_gap=np.linalg.norm((truth-fine).reshape(steps+1,-1),axis=1)/norm
        fine_initial_projection_gap=float(np.linalg.norm(fine[0]-initial)/norm)
        del fine
        case={'index':ci,'parameters':par,'normalization':norm,'initial_qualification':{
            'passed':bool(init_qualified),'actual_invocations':{str(t):r['qualified'] for t,r in initial_records.items()},
            'physical_tolerance_gap':ic_gap,'bound':cfg['qualification']['initial_agreement']},
            'shared_ic_parity':parity,'head_initial_error':tight_fit['initial_field_error'],
            'span_native_initial_error':float(np.linalg.norm(np.asarray(P.reconstruct(basis,jnp.asarray(projected),jnp.asarray(center)))-initial)/norm),
            'reference':{'dt':truth_dt,'check_dt':fine_dt,'per_time_gap':ref_gap,'max_gap':float(ref_gap.max()),
                'initial_projection_gap':float(np.linalg.norm(truth[0]-initial)/norm),
                'fine_initial_projection_gap':fine_initial_projection_gap,
                'truth_sha256':hashlib.sha256(truth.tobytes()).hexdigest(),'scope':'same-grid discrete CNAB2 reference; no continuum claim'},
            'runs':[],'tight_tighter':[]}
        report['cases'].append(case);save()
        _,prepare,_=SR.make_frozen_step(dt,n,len(ahead),1,1e-6)
        oa=tuple(jnp.asarray(opsall[name]) for name in ('A','T','lam','Dd'))
        wide=prepare(float(par[-1]),*oa);rows=cfg['rows']
        selected=prepare(float(par[-1]),oa[0][:rows],oa[1][:rows],oa[2][:rows],oa[3][:,:rows])
        cache={}
        for tol in cfg['solver_gtols']:
            for arm in cfg['arms']:
                kind='head' if arm=='head_common' else 'span';a0=projected if arm=='span_native' else ahead
                tag=f'{arm}_g{tol:g}';tick=time.perf_counter()
                ans=P.rollout(kind,a0,z0,center,params,selected,wide,dt,n,k,steps,tol,cfg['budget'],
                    basis,truth,geom,norm,codes,candidates,False,rows)
                # Keep original Jacobian-action qualification without optional manifold/tangent fits.
                action=[]
                if tol==min(cfg['solver_gtols']):
                    _,ev=P.make_eval(dt,n,len(ahead),k,kind)
                    for t in range(steps):
                        check=P.action_check(ev,ans['states'][t+1],
                            (jnp.asarray(ans['coefficients'][t]),selected,params),700+t)
                        action.append(check);ans['records'][t]['jacobian_action']=check
                err=np.asarray([m['relative_error'] for m in ans['metrics']])
                item={'tag':tag,'arm':arm,'gtol':tol,'dt':dt,'rows':rows,'times':np.arange(steps+1)*dt,
                    'initial_error':float(err[0]),'evolved_max_error':float(err[1:].max()),
                    'metrics':ans['metrics'],'all_steps_stationary':all(r['reason']=='stationarity' for r in ans['records']),
                    'max_gradient':max(r['gradient'] for r in ans['records']),
                    'termination_counts':{reason:sum(r['reason']==reason for r in ans['records']) for reason in sorted({r['reason'] for r in ans['records']})},
                    'jacobian_action_checks':action,'jacobian_action_checked':tol==min(cfg['solver_gtols']),
                    'jacobian_action_pass':all(x['minimum']<=cfg['qualification']['jacobian_action'] for x in action) if action else None,
                    'elapsed_diagnostic_seconds':time.perf_counter()-tick,'timing_claim':False,
                    'records_file':tag+'.json','state_file':tag+'.npz'}
                np.savez_compressed(directory/(tag+'.npz'),states=ans['states'],centers=ans['centers'],
                    coefficients=ans['coefficients'],times=np.arange(steps+1)*dt)
                S.dump(directory/(tag+'.json'),{'records':ans['records'],'optional_manifold_distance':'not run'})
                case['runs'].append(item);cache[(arm,tol)]=ans;save()
        for arm in cfg['arms']:
            loose=cache[(arm,max(cfg['solver_gtols']))];tight=cache[(arm,min(cfg['solver_gtols']))]
            gaps=np.linalg.norm((loose['fields']-tight['fields']).reshape(steps+1,-1),axis=1)/norm
            evolved=max(m['relative_error'] for m in tight['metrics'][1:])
            bound=min(cfg['qualification']['field_absolute'],cfg['qualification']['field_fraction']*evolved)
            case['tight_tighter'].append({'arm':arm,'per_time_gap':gaps,'max_gap':float(gaps.max()),'bound':bound,'passed':bool(gaps.max()<=bound)})
        actual_common=[]
        for tol in cfg['solver_gtols']:
            head0=cache[('head_common',tol)]['fields'][0];span0=cache[('span_common',tol)]['fields'][0]
            fit0=initial_fields[min(cfg['fit_gtols'])]
            actual_common.append({'gtol':tol,'head_vs_span':float(np.linalg.norm(head0-span0)/norm),
                'head_vs_qualified_fit':float(np.linalg.norm(head0-fit0)/norm),
                'span_vs_qualified_fit':float(np.linalg.norm(span0-fit0)/norm)})
        case['actual_common_initial_state_checks']=actual_common
        common_pass=all(max(c[k] for k in ('head_vs_span','head_vs_qualified_fit','span_vs_qualified_fit'))
                        <=cfg['qualification']['common_state'] for c in actual_common)
        reference_bound=min(cfg['qualification']['reference_absolute'],
                            cfg['qualification']['reference_fraction']*min(r['evolved_max_error'] for r in case['runs']))
        case['reference'].update(bound=reference_bound,passed=bool(ref_gap.max()<=reference_bound
            and max(case['reference']['initial_projection_gap'],fine_initial_projection_gap)<=1e-12))
        case['qualification']={'initialization':bool(init_qualified),'common_state':common_pass and parity<=cfg['qualification']['common_state'],
            'operator':operator_pass,'rollout':all(r['all_steps_stationary'] for r in case['runs']),
            'jacobian':sum(len(r['jacobian_action_checks']) for r in case['runs'])==len(cfg['arms'])*steps
                and all(r['jacobian_action_pass'] for r in case['runs'] if r['jacobian_action_checked']),
            'refinement':all(r['passed'] for r in case['tight_tighter']),'reference':case['reference']['passed']}
        case['qualification']['passed']=all(case['qualification'].values())
        np.savez_compressed(directory/'truth_slices.npz',times=np.arange(steps+1)*dt,velocity_zmid=truth[:,:,:,n//2,:])
        del cache,truth;save()
        print(f'fresh phase case={ci} qualification={case["qualification"]}',flush=True)
    ids=[c['index'] for c in report['cases']]
    complete=ids==list(range(cfg['eval_cases']))
    report['qualification']={'complete':complete,'cases_completed':len(ids),'expected_cases':cfg['eval_cases'],
        'all_cases_pass':complete and all(c['qualification']['passed'] for c in report['cases'])}
    save();print('fresh phase complete; no case-dependent repair',flush=True)

if __name__=='__main__':main()
