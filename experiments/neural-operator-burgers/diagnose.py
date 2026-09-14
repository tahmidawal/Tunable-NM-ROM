"""Same-case evolved bank, nonlinear-fit, online-ROM and efficient FOM pilot.

The inherited checkpoint has unmatched training history: no matched-data claim.
Oracle fits are offline diagnostics, never deployed solver timing baselines.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import data as d

CHECKPOINT = d.ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'


def burn(jax, jnp, seconds=2.):
    q = jax.jit(lambda x: x @ x)
    a = jnp.eye(512, dtype=jnp.float64)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        q(a).block_until_ready()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-dt', type=float, default=.0003125)
    parser.add_argument('--cases', type=int, default=8)
    parser.add_argument('--intervals', type=int, default=256)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.cases <= 8 and args.reps >= 1
    jax, e = d.gpu_modules()
    import jax.numpy as jnp
    import accuracy_paths as ap
    ref = json.loads(args.reference_index.read_text())
    assert ref['pde'] == 'burgers' and ref['provenance']['backend'] == 'gpu'
    assert ref['provenance']['f64'] and ref['provenance']['matmul_precision'] == 'highest'
    out = d.make_output(args.out)
    params, Zold, _ = e.sc.load_pkl(CHECKPOINT)
    K, R = Zold.shape[1], params['h_lin'].shape[1]
    L, M, m = args.intervals, 4*K, 16*K
    cfg = dict(reference_dt=args.reference_dt,dt=.005, strict=dict(ic_budget=400, step_budget=180, gtol=1e-6),
               oracle_starts=8, oracle_budget=400, oracle_gradient_tolerance=1e-7,
               reference_interpretation='analytic Gaussian continuum-family pilot; fine reference initial data use generation descriptors offline; no arbitrary sampled-field claim')
    if args.smoke:
        assert L == 64 and args.cases == 1
        archive = np.load(d.ROOT / 'consolidated/fixtures/burgers/operators.npz')
        G = e.sc.SeparableDecoder(params,K,R).feat_at(e.coords(L),chunk=8192)
        operators = (G, *[jnp.asarray(archive[k]) for k in ['A','lam','G5','Pq']], None,None,jnp.asarray(archive['candidate_Z']))
        hrot = e.sc.head(params,operators[7]) @ jnp.asarray(archive['cold_R']).T
        cold = tuple(jnp.asarray(archive[k]) for k in ['cold_xy','cold_w','cold_Q','cold_R']) + (hrot,jnp.sum(hrot*hrot,1))
        raw = json.loads((d.ROOT/'consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/accuracy09/archive/out/result.json').read_text())
        setup = next(x for x in raw['mesh_setup'] if x['intervals']==64 and x['model']=='frozen')
        cold_setup = dict(scope='saved operators for bounded local smoke only')
        cfg.update(oracle_starts=2, oracle_budget=20)
    else:
        operators, setup = e.build_rom(params,Zold,L,M,m,candidate_cap=8192,fit_states=64)
        cold, cold_setup = e.build_gauss_cold(params,operators[7])
    G = operators[0]
    Q, triangular = jnp.linalg.qr(G, mode='reduced')
    jax.block_until_ready(triangular)
    bank_singular_values = np.asarray(jnp.linalg.svd(triangular, compute_uv=False))
    assert np.isfinite(bank_singular_values).all() and bank_singular_values[-1] > 0
    hp = ap.whiten_head(params, triangular)
    candidate_z = operators[7]
    candidate_h = e.sc.head(hp,candidate_z)
    fit_one = ap.make_stationary_lm(lambda z,y:e.sc.head(hp,z)-y,K,cfg['oracle_budget'],gtol=cfg['oracle_gradient_tolerance'])
    fit = jax.jit(jax.vmap(lambda z,y:fit_one(z,(y,),0.)))
    rom = ap.make_rom(params,L,cfg['dt'],setup['trust_radius'],**cfg['strict'])
    presets = [dict(name='same_nt1e-2_dt005',mesh=L,dt=.005,ntol=.01,ltol=.5),
               dict(name='same_nt1e-4_dt005',mesh=L,dt=.005,ntol=1e-4,ltol=.01),
               dict(name='same_nt1e-6_dt005',mesh=L,dt=.005,ntol=1e-6,ltol=1e-8),
               dict(name='same_nt1e-2_dt01',mesh=L,dt=.01,ntol=.01,ltol=.5),
               dict(name='coarse_half_dt005',mesh=L//2,dt=.005,ntol=1e-4,ltol=.01),
               dict(name='coarse_quarter_dt01',mesh=L//4,dt=.01,ntol=1e-4,ltol=.01)]
    if args.smoke:
        presets = presets[:1]
    def make_supplied_fom(p):
        native=e.make_fom(p['mesh'],p['dt'],target=L)[0]
        @jax.jit
        def query(u0,nu,ntol,ltol):
            fields,it,rn=native(u0,nu,ntol,ltol)
            return fields.at[0].set(u0),it,rn
        return query
    foms = {p['name']:make_supplied_fom(p) for p in presets}
    report = dict(schema_version=1,pde='burgers',kind='inherited-checkpoint-diagnosis',complete=False,
                  provenance=d.provenance(jax), checkpoint_sha256=d.sha(CHECKPOINT),
                  own_source_sha256=d.sha(Path(__file__)), accuracy_paths_sha256=d.sha(Path(ap.__file__)),
                  config=cfg, intervals=L,K=K,R=R,M=M,m=m,setup=setup,cold_setup=cold_setup,
                  bank_singular_values=bank_singular_values.tolist(),bank_condition_number=float(bank_singular_values[0]/bank_singular_values[-1]),
                  reference_index_sha256=d.sha(args.reference_index),reference_gate=ref.get('gate',ref.get('profile_gate')),
                  physical_accuracy_status='provisional whenever empirical reference gate fails or is incomplete',
                  training_history='original checkpoint unmatched to new case bank; pilot only',
                  timing_contract='same GPU interleaved complete supplied-device-field to six device-fields; setup and compilation excluded; host transfers separately retained',
                  oracle_interpretation='best found finite multi-start nonlinear fit, not certified global minimum; error diagnostics are not additive',
                  presets=presets,repetitions=args.reps,cases=[],invocations=[])
    index_path=out/'index.json'
    d.write_json(index_path,report)
    for case in range(args.cases):
        record=d.case_record('calibration',case)
        physical=e.params_draw(record['seed'],1)[0]
        anchors=[s for s in ref.get('solves', []) if s['case_id']==record['case_id'] and s['intervals']==4096 and s['dt']==args.reference_dt]
        if args.smoke:
            anchor=ref['records'][0]
            artifact=args.reference_index.parent/anchor['path']
            assert d.sha(artifact)==anchor['sha256']
            reference=np.load(artifact)['target'][:,0]
            assert reference.shape[-1]==L+1
        else:
            assert len(anchors)==1, f'missing completed anchor for case {case}'
            anchor=anchors[0];artifact=args.reference_index.parent/anchor['path']
            assert d.sha(artifact)==anchor['sha256']
            reference=d.restrict(np.load(artifact)['fields'],L)
        supplied=e.initial(L,physical)
        assert np.array_equal(reference[0],supplied)
        u0=jnp.asarray(supplied);nu=jnp.asarray(physical[4])
        result=jax.block_until_ready(rom(u0,nu,operators,cold))
        online_fields=np.asarray(result[0]);online_z=np.asarray(result[4])
        targets=jnp.asarray(reference[:,1:-1,1:-1].reshape(6,-1))
        y=targets@Q
        free_coefficients=jax.scipy.linalg.solve_triangular(triangular,y.T,lower=False).T
        bank_coefficient_replay=float(jnp.linalg.norm(free_coefficients@G.T-y@Q.T)/jnp.linalg.norm(y@Q.T))
        assert bank_coefficient_replay < 1e-8, "ill-conditioned free-bank reconstruction"
        bank_fields=np.zeros_like(reference)
        bank_fields[:,1:-1,1:-1]=np.asarray(y@Q.T).reshape(6,L-1,L-1)
        nonlinear_fields=np.zeros_like(reference)
        latent=[];fits=[]
        for ti in range(6):
            distance=jnp.sum(candidate_h*candidate_h,1)-2*candidate_h@y[ti]
            ids=jnp.argsort(distance)[:cfg['oracle_starts']]
            starts=jnp.concatenate((candidate_z[ids],jnp.asarray(online_z[ti])[None]))
            if latent:
                starts=jnp.concatenate((starts,jnp.asarray(latent[-1])[None]))
            fitted=jax.tree_util.tree_map(np.asarray,fit(starts,jnp.broadcast_to(y[ti],(len(starts),R))))
            # Preserve original starts too: optimizer failure cannot make an
            # oracle upper bound worse than the actual online latent state.
            allz=jnp.concatenate((starts,jnp.asarray(fitted[0])))
            losses=np.asarray(jnp.sum((e.sc.head(hp,allz)-y[ti])**2,axis=1))
            valid=np.isfinite(losses);assert valid.any()
            best=int(np.argmin(np.where(valid,losses,np.inf)));z=np.asarray(allz[best]);latent.append(z)
            nonlinear_fields[ti,1:-1,1:-1]=np.asarray(G@e.sc.head(params,jnp.asarray(z))).reshape(L-1,L-1)
            fits.append(dict(time=float(d.TIMES[ti]),selected_candidate=best,candidate_ids=np.asarray(ids).tolist(),
                squared_projected_losses=losses.tolist(),optimization_iterations=fitted[2].tolist(),
                optimization_reasons=fitted[3].tolist(),optimization_gradients=fitted[4].tolist()))
        errors={name:d.fixed_initial_errors(field,reference) for name,field in
                [('bank',bank_fields),('nonlinear_best_found',nonlinear_fields),('online',online_fields)]}
        assert np.all(np.asarray(errors['bank']['per_time'])<=np.asarray(errors['nonlinear_best_found']['per_time'])+1e-9)
        assert np.all(np.asarray(errors['nonlinear_best_found']['per_time'])<=np.asarray(errors['online']['per_time'])+1e-9)
        name=record['case_id']+'.diagnosis.npz'
        np.savez(out/name,input=supplied,reference=reference,bank=bank_fields,nonlinear=nonlinear_fields,
                 online=online_fields,online_z=online_z,nonlinear_z=np.asarray(latent),free_coefficients=np.asarray(free_coefficients),
                 step_iterations=np.asarray(result[1]),step_reasons=np.asarray(result[3]),
                 initial_iterations=np.asarray(result[5]),initial_reason=np.asarray(result[6]),
                 step_gradients=np.asarray(result[8]),initial_gradient=np.asarray(result[9]))
        report['cases'].append(dict(**record,anchor_sha256=anchor['sha256'],path=name,sha256=d.sha(out/name),
                errors=errors,free_bank_coefficient_replay_relative=bank_coefficient_replay,oracle_fits=fits,online_stationary=bool(np.max(np.asarray(result[8]))<=1e-6 and float(result[9])<=1e-6)))
        d.write_json(index_path,report)
        # Each setting compiles before timing; every repetition keeps its own
        # full fields, stopping evidence, and latency from the same invocation.
        for p in presets:
            jax.block_until_ready(foms[p['name']](u0,nu,p['ntol'],p['ltol']))
        order=['rom']+[p['name'] for p in presets]
        rng=np.random.default_rng(20260914+case)
        for rep in range(args.reps):
            for method in rng.permutation(order):
                burn(jax,jnp,.1 if args.smoke else 2.)
                complete_start=time.perf_counter()
                invocation_u0=jax.device_put(supplied)
                invocation_nu=jax.device_put(np.asarray(physical[4],dtype=np.float64))
                jax.block_until_ready((invocation_u0,invocation_nu))
                inputseconds=time.perf_counter()-complete_start
                start=time.perf_counter()
                if method=='rom':
                    timed=jax.block_until_ready(rom(invocation_u0,invocation_nu,operators,cold))
                else:
                    p=next(p for p in presets if p['name']==method)
                    timed=jax.block_until_ready(foms[method](invocation_u0,invocation_nu,p['ntol'],p['ltol']))
                seconds=time.perf_counter()-start
                hoststart=time.perf_counter();host=jax.tree_util.tree_map(np.asarray,timed);hostseconds=time.perf_counter()-hoststart
                complete_seconds=time.perf_counter()-complete_start
                fields=host[0].copy()
                assert fields.dtype==np.float64 and np.isfinite(fields).all()
                name=f"{record['case_id']}_{method}_rep{rep}.npz"
                audit_arrays = dict(fields=fields,iterations=host[1],residuals=host[2])
                if method=='rom':
                    audit_arrays.update(step_reasons=host[3],output_latents=host[4],initial_iterations=host[5],
                        initial_reason=host[6],internal_latents=host[7],step_gradients=host[8],initial_gradient=host[9])
                np.savez(out/name,**audit_arrays)
                row=dict(case_id=record['case_id'],method=str(method),rep=rep,gpu_seconds=seconds,
                         input_transfer_seconds=inputseconds,output_transfer_seconds=hostseconds,host_to_host_seconds=complete_seconds,error=d.fixed_initial_errors(fields,reference),
                         path=name,sha256=d.sha(out/name),newton_or_lm_iterations=int(host[1].sum()))
                if method=='rom':
                    row.update(stationary=bool(np.max(host[8])<=1e-6 and float(host[9])<=1e-6),initial_iterations=int(host[5]))
                else:
                    row.update(max_relative_residual=float(host[2].max()),converged=bool(np.isfinite(host[2]).all() and host[2].max()<=p['ntol']))
                report['invocations'].append(row)
                d.write_json(index_path,report)
        print('DIAGNOSIS COMPLETE',record['case_id'],json.dumps(errors),flush=True)
    report['complete']=True
    d.write_json(index_path,report)


if __name__=='__main__':
    main()
