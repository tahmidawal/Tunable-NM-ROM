"""Bounded development pilot; all performance observations are same-job paired."""
import argparse
from pathlib import Path
import hashlib
import json
import os
import time
import socket
import numpy as np
from scipy.fft import dstn
import scipy
import jax
import jax.numpy as jnp
from core import *


def save(path,obj):
    def clean(x):
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x): return None
        return x
    path.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')


def reference(param, intervals):
    F=full_source(intervals,param)
    c=dstn(F[1:-1,1:-1],type=1,norm='ortho',workers=1)
    return np.pad(dstn(c/eigenvalues(intervals),type=1,norm='ortho',workers=1),1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    dev=jax.devices()[0]
    assert dev.platform=='gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    params,ztrain,ckcfg=sc.load_pkl(args.checkpoint)
    assert ckcfg['N']==256 and ckcfg['k']==16 and ckcfg['r']==64
    assert all(np.asarray(v).dtype==np.float64 for v in jax.tree.leaves(params))
    draws=source_params(cfg['seed'],cfg['cohort_count'])
    report=dict(config=cfg,checkpoint_config=ckcfg,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        provenance=dict(commit=os.environ.get('COMMIT','local'),job_id=os.environ.get('SLURM_JOB_ID'),
            hostname=socket.gethostname(),gpu=dev.device_kind,backend=dev.platform,x64=True,
            matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],jax=jax.__version__,scipy=scipy.__version__),
        cohort=dict(kind='new_development_only',count=len(draws),seed=cfg['seed'],
            parameter_columns=['cx','cy','width','amplitude'],parameters=draws.tolist(),
            parameter_sha256=sha(draws),descriptors_never_passed_to_head=True),
        contract=dict(input='host float64 full nodal source; boundary RHS zero and unused',
            output='host float64 full nodal zero-Dirichlet solution',
            observations='common nested nodes, uniform discrete relative L2; boundary solution is zero',
            solver='unchanged ctol_tol.lm_tau_poisson, z0=mean training latent; no data-fitting cold start',
            stationarity='||J^T r||/(||J||F ||r||), evaluation-only after query timer',
            timing='synchronized component timestamps within total, full host transfers; each timed output scored',
            reference='independent SciPy DST-I FD solves with nested refinement; empirical uncertainty, not a rigorous bound',
            qualification='all sources: max(error + uncertainty) <= eps, uncertainty <= .1 eps; preserve nonstationary/stopped cases'),
        setup=[],references=[],rows=[],complete=False)
    output=args.out/'result.json'
    save(output,report)
    obs=cfg['observation_intervals']
    refs=[]
    ref_start=time.perf_counter()
    for case,param in enumerate(draws):
        chain=[reference(param,n) for n in cfg['reference_intervals']]
        restricted=[observation(u,n,obs) for u,n in zip(chain,cfg['reference_intervals'])]
        gaps=[relative(a,b) for a,b in zip(restricted[:-1],restricted[1:])]
        # Conservative empirical bound equal to last refinement difference,
        # in fine-reference normalization. No optimistic Richardson /3 scaling.
        normfine=float(np.linalg.norm(restricted[-1]))
        delta=float(np.linalg.norm(restricted[-2]-restricted[-1])/normfine)
        empirical_u=delta/(1-delta) if delta<1 else 1e100
        refs.append((chain[-1],restricted[-1],empirical_u))
        report['references'].append(dict(case=case,intervals=cfg['reference_intervals'],
            successive_relative_gaps=gaps, refinement_ratio=gaps[0]/gaps[1],
            reference_norm=normfine,uncertainty=empirical_u,
            finest_reference_sha256=sha(chain[-1]), observation_sha256=sha(restricted[-1]),
            refined_difference_normalization='||ref1024-ref2048|| / ||ref2048|| on common observation nodes',
            reference_converging=bool(gaps[1]<gaps[0]/2)))
        np.savez_compressed(args.out/f'reference_case{case}.npz',observation=restricted[-1],coarser_observation=restricted[-2])
        print('reference',case,gaps,'uncertainty',empirical_u,flush=True)
        save(output,report)
    report['reference_setup_seconds']=time.perf_counter()-ref_start
    for n in cfg['intervals']:
        print('assembling intervals',n,flush=True)
        ops=assemble(params,ztrain,n,cfg['requested_modes'],cfg['lm_budget'])
        info=ops['info']
        lam=jnp.asarray(eigenvalues(n))
        coarse_lam=jnp.asarray(eigenvalues(128))
        def query(arm,tau,source):
            if arm=='dst': return fom_query(source,lam)
            if arm=='dst_coarse128': return coarse_query(source,coarse_lam)
            return rom_query(source,ops,tau)
        # Exact full-grid weak residual and Jacobian gates on seeded latent states.
        zs=[ztrain.mean(0),ztrain[0],ztrain[17]]
        gates=[]
        for z in zs:
            z=jnp.asarray(z)
            source=full_source(n,draws[0])
            fm=ops['project'](jnp.asarray(source),ops['S'],ops['I'],ops['J'],ops['W'])
            def full_r(z):
                u=(ops['bank']@sc.head(params,z)).reshape((n-1,n-1))
                residual=mp.neg_lap_interior(u,n+1)-jnp.asarray(source[1:-1,1:-1])
                return ops['project'](jnp.pad(residual,1),ops['S'],ops['I'],ops['J'],ops['W'])
            reduced=lambda z:ops['B']@sc.head(params,z)-fm
            rr=np.asarray(reduced(z)); rf=np.asarray(full_r(z))
            jf=np.asarray(jax.jacfwd(full_r)(z)); jr=np.asarray(jax.jacfwd(reduced)(z))
            gates.append(dict(residual_relative=relative(rr,rf),jacobian_relative=relative(jr,jf)))
        info['weak_operator_gates']=gates
        assert max(max(g.values()) for g in gates)<1e-9
        report['setup'].append(info)
        save(output,report)
        for case,param in enumerate(draws):
            source=full_source(n,param)
            same=np.asarray(dst_solve(jnp.asarray(source),lam))
            scipy_same=reference(param,n)
            parity=relative(same,scipy_same)
            assert parity<1e-11
            # Warm every path, including output and validation diagnostics.
            warm_start=time.perf_counter()
            subjects=[('dst',None),('dst_coarse128',None)]+[('rom',tau) for tau in cfg['taus']]
            for arm,tau in subjects:
                for _ in range(cfg['warmup']):
                    query(arm,tau,source)
            warm_seconds=time.perf_counter()-warm_start
            info.setdefault('warmup_seconds_by_case',[]).append(warm_seconds)
            for rep in range(cfg['repetitions']):
                order=subjects if rep%2==0 else subjects[::-1]
                for order_index,(arm,tau) in enumerate(order):
                    burn(cfg['burn_seconds'])
                    field,record=query(arm,tau,source)
                    obs_field=observation(field,n,obs)
                    physical=relative(obs_field,refs[case][1])
                    finite=bool(np.isfinite(field).all())
                    stat=(arm!='rom' or record['stationarity']<=cfg['stationarity_tolerance'])
                    # A deliberate deployment tau stop need not be a stationary optimum;
                    # tau=0 convergence qualification requires measured stationarity.
                    tau_reached=(arm=='rom' and record['reason']==2)
                    valid=finite and (arm!='rom' or tau_reached or stat)
                    record.update(arm=arm,tau=tau,intervals=n,case=case,repetition=rep,
                        order_index=order_index,source_sha256=sha(source),field_sha256=sha(field),
                        same_grid_error=relative(field,same),physical_error=physical,
                        uncertainty=refs[case][2],conservative_physical_error=(physical+refs[case][2])/(1-refs[case][2]),
                        finite=finite,stationary=stat,tau_reached=tau_reached,solver_valid=valid,
                        scipy_dst_parity=parity)
                    report['rows'].append(record)
                    if rep==0:
                        np.savez_compressed(args.out/f'field_n{n}_case{case}_{arm}_tau{tau}.npz',field=field)
                save(output,report)
            print('finished',n,'case',case,flush=True)
    report['complete']=True
    save(output,report)
    print('ALL-DONE',flush=True)

if __name__=='__main__':
    main()
