"""Frozen test-space ladder, oracle diagnosis and fused/modular paired queries."""
import argparse
import json
import os
import socket
from pathlib import Path
from followup import *
from pilot import reference,save
import scipy


def score(field,same,ref_obs,delta,n,obs):
    error=relative(observation(field,n,obs),ref_obs)
    return dict(same_grid_error=relative(field,same),physical_error=error,
        reference_delta=delta,uncertainty=delta,conservative_physical_error=(error+delta)/(1-delta),
        field_sha256=sha(field),finite=bool(np.isfinite(field).all()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text());args.out.mkdir(parents=True,exist_ok=False)
    dev=jax.devices()[0]
    assert dev.platform=='gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    params,ztrain,ckcfg=sc.load_pkl(args.checkpoint)
    assert ckcfg['N']==256 and ckcfg['k']==16 and ckcfg['r']==64
    draws=source_params(cfg['seed'],cfg['cohort_count'])
    obs=cfg['observation_intervals']
    report=dict(config=cfg,checkpoint_config=ckcfg,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        provenance=dict(commit=os.environ.get('COMMIT','local'),job_id=os.environ.get('SLURM_JOB_ID'),
            hostname=socket.gethostname(),gpu=dev.device_kind,backend='gpu',x64=True,
            matmul_precision='highest',jax=jax.__version__,scipy=scipy.__version__),
        cohort=dict(kind='same_development_cohort_as_pilot01',count=len(draws),seed=cfg['seed'],
            parameter_columns=['cx','cy','width','amplitude'],parameters=draws.tolist(),
            parameter_sha256=sha(draws),descriptors_never_passed_to_head=True),
        contract=dict(input='full host nodal source',output='full host nodal solution plus solver counters',
            online_initialization='mean training latent, no oracle fields or parameter descriptors',
            reference='independent nested SciPy DST finite-difference solves, empirical differences only',
            error_normalization='uniform L2 over common nested observation nodes relative to refined reference',
            qualification='all declared sources pass solver/parity checks, (error+delta)/(1-delta)<=epsilon and delta<=.1epsilon',
            fused_components='input, fused projection+solve+decode device time, and full output; internal stage times unavailable',
            segmented_components='same-invocation input, projection/init, solver and output times',
            oracle='reference-only exact QR objective, fixed deterministic starts, never online init or selection',
            original_solver='unchanged ctol_tol.lm_tau_poisson including generic jnp.linalg.solve',
            aggregation='median of case-median latency; median of ratios of paired case-median latencies'),
        references=[],setup=[],oracles=[],parity=[],rows=[],complete=False)
    out=args.out/'result.json'
    def persist():save(out,report)
    persist()
    references=[]
    start=time.perf_counter()
    for case,param in enumerate(draws):
        chain=[reference(param,n) for n in cfg['reference_intervals']]
        nested=[observation(field,n,obs) for field,n in zip(chain,cfg['reference_intervals'])]
        gaps=[relative(x,y) for x,y in zip(nested[:-1],nested[1:])]
        delta=float(np.linalg.norm(nested[-2]-nested[-1])/np.linalg.norm(nested[-1]))
        assert gaps[1]<gaps[0]/2 and delta<1
        references.append((nested[-1],delta))
        report['references'].append(dict(case=case,intervals=cfg['reference_intervals'],
            successive_relative_gaps=gaps,refinement_ratio=gaps[0]/gaps[1],
            reference_norm=float(np.linalg.norm(nested[-1])),reference_delta=delta,uncertainty=delta,
            uncertainty_kind='empirical_reference_difference_no_rigorous_bound',
            finest_reference_sha256=sha(chain[-1]),observation_sha256=sha(nested[-1]),reference_converging=True))
        np.savez_compressed(args.out/f'reference_case{case}.npz',observation=nested[-1],coarser_observation=nested[-2])
        print('reference',case,'delta',delta,flush=True);persist()
    report['reference_setup_seconds']=time.perf_counter()-start
    for n in cfg['intervals']:
        print('assembling interval ladder',n,flush=True)
        ops_by_m={m:assemble(params,ztrain,n,m,cfg['lm_budget']) for m in cfg['requested_modes_ladder']}
        kernels={m:fused_kernel(ops) for m,ops in ops_by_m.items()}
        base=ops_by_m[cfg['requested_modes_ladder'][0]]
        assert all(ops['info']['bank_sha256']==base['info']['bank_sha256'] for ops in ops_by_m.values())
        lam=jnp.asarray(eigenvalues(n));coarse_lam=jnp.asarray(eigenvalues(128))
        q,r,qr_info=qr_bank(base)
        assert qr_info['relative_reconstruction']<1e-12 and qr_info['orthogonality_frobenius']<1e-11
        oracle_lm=oracle_solvers(params,r,ckcfg['k'],cfg['oracle_budgets'],base['info']['trust_delta'])
        starts=[ztrain.mean(0)]+[ztrain[index] for index in cfg['oracle_training_code_indices']]
        for m,ops in ops_by_m.items():
            # Independent matrix/FD action gates at fixed checkpoint latents.
            gates=[]
            for znp in (ztrain.mean(0),ztrain[0]):
                z=jnp.asarray(znp)
                u=ops['decode'](z,ops['bank'],params)
                rhs=jnp.pad(mp.neg_lap_interior(u[1:-1,1:-1],n+1),1)
                projected=ops['project'](rhs,ops['S'],ops['I'],ops['J'],ops['W'])
                reduced=ops['B']@sc.head(params,z)
                gates.append(relative(reduced,projected))
            assert max(gates)<1e-9
            ops['info'].update(weak_operator_relative_gates=gates,qr_bank=qr_info)
            report['setup'].append(ops['info'])
        persist()
        def query(arm,m,tau,src):
            if arm=='dst':return fom_query(src,lam)
            if arm=='dst_coarse128':return coarse_query(src,coarse_lam)
            if arm=='rom_modular':return rom_query(src,ops_by_m[m],tau)
            return fused_query(src,ops_by_m[m],tau,kernels[m])
        subjects=[('dst',None,None),('dst_coarse128',None,None)]
        subjects += [(arm,m,tau) for m in cfg['requested_modes_ladder'] for tau in cfg['taus'] for arm in ('rom_modular','rom_fused')]
        for case,param in enumerate(draws):
            source=full_source(n,param)
            same=np.asarray(dst_solve(jnp.asarray(source),lam))
            same_parity=relative(same,reference(param,n));assert same_parity<1e-11
            start=time.perf_counter()
            diagnostic,bankfield,headfield=oracle_fit(same,base,q,r,oracle_lm,starts)
            diagnostic.update(case=case,intervals=n,diagnostic_total_seconds=time.perf_counter()-start,
                start_labels=['training_mean']+[f'training_code_{idx}' for idx in cfg['oracle_training_code_indices']],
                bank_metrics=score(bankfield,same,*references[case],n,obs),
                best_head_metrics=score(headfield,same,*references[case],n,obs))
            assert max(x['qr_identity_absolute'] for x in diagnostic['rows'])<1e-10
            report['oracles'].append(diagnostic)
            np.savez_compressed(args.out/f'oracle_n{n}_case{case}.npz',bank_projection=bankfield,best_head=headfield)
            gates={}
            for m in cfg['requested_modes_ladder']:
                for tau in cfg['taus']:
                    mod=query('rom_modular',m,tau,source);fused=query('rom_fused',m,tau,source)
                    gate=parity(mod,fused)
                    gate.update(case=case,intervals=n,requested_modes=m,tau=tau)
                    report['parity'].append(gate);gates[(m,tau)]=gate['passed']
                    if not gate['passed']:print('PARITY-FAILED',gate,flush=True)
            # Failed parity is retained and disqualifies fused measurements, rather
            # than silently relabeling a different solve as a speed optimization.
            warm_start=time.perf_counter()
            for arm,m,tau in subjects:
                for _ in range(cfg['warmup']):query(arm,m,tau,source)
            report.setdefault('warmup',[]).append(dict(intervals=n,case=case,seconds=time.perf_counter()-warm_start))
            for rep in range(cfg['repetitions']):
                order=subjects if rep%2==0 else subjects[::-1]
                for order_index,(arm,m,tau) in enumerate(order):
                    burn(cfg['burn_seconds'])
                    field,row=query(arm,m,tau,source)
                    row.update(score(field,same,*references[case],n,obs))
                    isrom=arm.startswith('rom')
                    stationary=not isrom or row['stationarity']<=cfg['stationarity_tolerance']
                    tau_reached=isrom and row['reason']==2
                    gate=gates.get((m,tau),True) if arm=='rom_fused' else True
                    row.update(arm=arm,requested_modes=m,retained_modes=ops_by_m[m]['info']['retained_modes'] if m else None,
                        tau=tau,intervals=n,case=case,repetition=rep,order_index=order_index,
                        source_sha256=sha(source),stationary=stationary,tau_reached=tau_reached,
                        parity_passed=gate,solver_valid=row['finite'] and gate and (not isrom or tau_reached or stationary),
                        scipy_dst_parity=same_parity)
                    report['rows'].append(row)
                    if rep==0:np.savez_compressed(args.out/f'field_n{n}_case{case}_{arm}_M{m}_tau{tau}.npz',field=field)
                persist()
            print('finished',n,'case',case,'bank/head error',diagnostic['full_bank_same_grid_error'],diagnostic['best_same_grid_error'],flush=True)
    report['complete']=True;persist();print('ALL-DONE',flush=True)

if __name__=='__main__':main()
