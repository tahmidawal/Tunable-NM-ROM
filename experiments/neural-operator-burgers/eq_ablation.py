"""Isolate m=256/512/1024 with the original head, bank, cold fit and weak modes.

The m=256 control reuses the exact accepted weights. New rules refit the same
recorded decoder-output snapshot pool. Full queries are paired within one GPU.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import data as d
from refine import configure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,required=True)
    a=p.parse_args();configure();selected,_=d.read_calibration(a.reference,256)
    control=json.loads(a.diagnosis.read_text());reference=json.loads(a.reference.read_text())
    assert control['complete'] and control['m']==256 and control['M']==64 and control['K']==16
    assert d.sha(a.reference)==control['reference_index_sha256']
    jax,e=d.gpu_modules()
    import jax.numpy as jnp
    import accuracy_paths as ap
    from diagnose import burn
    out=d.make_output(a.out);started=time.monotonic()
    checkpoint=d.ROOT/'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
    assert d.sha(checkpoint)==control['checkpoint_sha256']
    params,Z,_=e.sc.load_pkl(checkpoint);K,R=control['K'],control['R'];L=256;M=64
    dec=e.sc.SeparableDecoder(params,K,R);G=dec.feat_at(e.coords(L),chunk=8192)
    phi,lam,mode_ids=e.modes(L,M);assert np.array_equal(mode_ids,np.asarray(control['setup']['mode_ids']))
    A=jnp.asarray(phi).T@G
    candidate_z=jnp.asarray(Z[::max(1,len(Z)//8192)])
    cold,cold_info=e.build_gauss_cold(params,candidate_z)
    pos=np.asarray(control['setup']['eq_indices']);weights=np.asarray(control['setup']['eq_weights'])
    ij=np.stack(np.unravel_index(pos,(L-1,L-1)),1)+1
    offsets=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    G5=dec.feat_at(((ij[:,None,:]+offsets[None,:,:])/L).reshape(-1,2)).reshape(256,5,R)
    operators={'m256':(G,A,jnp.asarray(lam),G5,jnp.asarray(phi[pos]*weights[:,None]),None,None,candidate_z)}
    report=dict(schema_version=1,pde='burgers',kind='frozen-head-EQ-ablation',complete=False,
                provenance=d.provenance(jax),source_sha256=d.sha(Path(__file__)),diagnosis_source_sha256=d.sha(d.HERE/'diagnose.py'),
                checkpoint_sha256=d.sha(checkpoint),reference_index_sha256=d.sha(a.reference),control_index_sha256=d.sha(a.diagnosis),
                K=K,R=R,L=L,M=M,m_levels=[256,512,1024],dt=.005,strict=control['config']['strict'],
                control='exact original256 positions/weights; same bank/A/lambda/candidate codes/cold matrices for every arm',
                training_history='unchanged inherited checkpoint; unmatched-data pilot only',
                output_policy='native fitted t0 retained for isolated comparison; supplied-t0 final-panel wrapper remains separate',
                cold_setup=cold_info,setup=[dict(arm='m256',cache_reused=True,original_setup=control['setup'])],invocations=[],
                reference_setting=selected,timing_seed=2026091411,repetitions=3,
                reference_interpretation=d.PROTOCOL['reference_interpretation'])
    path=out/'index.json';d.write_json(path,report)
    for m in [512,1024]:
        if time.monotonic()-started+300>a.seconds:
            report['stop_reason']='budget before EQ refit';d.write_json(path,report);return
        print('REFIT EQ',m,flush=True)
        new,info=e.build_rom(params,Z,L,M,m,eq_seed=control['setup']['eq_seed'],candidate_cap=8192,fit_states=64)
        assert info['eq_fit_rows']==control['setup']['eq_fit_rows']
        assert info['mode_ids']==control['setup']['mode_ids']
        assert np.array_equal(np.asarray(new[7]),np.asarray(candidate_z))
        bank_parity=float(jnp.linalg.norm(new[0]-G)/jnp.linalg.norm(G))
        linear_parity=float(jnp.linalg.norm(new[1]-A)/jnp.linalg.norm(A))
        assert bank_parity<1e-12 and linear_parity<1e-12
        # Only the advection rule varies; freeze all shared arrays exactly.
        operators[f'm{m}']=(G,A,jnp.asarray(lam),new[3],new[4],None,None,candidate_z)
        report['setup'].append(dict(arm=f'm{m}',cache_reused=False,info=info,bank_parity=bank_parity,linear_parity=linear_parity))
        d.write_json(path,report);del new
    q=ap.make_rom(params,L,.005,control['setup']['trust_radius'],**control['config']['strict'])
    foms={}
    for dt in [.005,.01]:
        native=e.make_fom(L,dt,target=L)[0]
        def make(native):
            @jax.jit
            def query(u0,nu):
                fields,it,rn=native(u0,nu,.01,.5)
                return fields.at[0].set(u0),it,rn
            return query
        foms[f'fom_dt{dt}']=make(native)
    arms=list(operators)+list(foms)
    for case in range(8):
        record=d.case_record('calibration',case)
        anchor=next(r for r in reference['solves'] if r['case_id']==record['case_id'] and r['intervals']==4096 and r['dt']==.00015625)
        artifact=a.reference.parent/anchor['path'];assert d.sha(artifact)==anchor['sha256']
        truth=d.restrict(np.load(artifact)['fields'],L)
        physical=e.params_draw(record['seed'],1)[0];supplied=e.initial(L,physical)
        assert np.array_equal(supplied,truth[0]);u0=jnp.asarray(supplied);nu=jnp.asarray(physical[4])
        for arm in arms:
            if arm in operators:jax.block_until_ready(q(u0,nu,operators[arm],cold))
            else:jax.block_until_ready(foms[arm](u0,nu))
        rng=np.random.default_rng(report['timing_seed']+case)
        for rep in range(3):
            for arm in rng.permutation(arms):
                if time.monotonic()-started+120>a.seconds:
                    report['stop_reason']='budget checkpoint before timed query';d.write_json(path,report);return
                burn(jax,jnp,2.)
                t0=time.perf_counter();ui=jax.device_put(supplied);nui=jax.device_put(np.asarray(physical[4],dtype=np.float64));jax.block_until_ready((ui,nui));input_seconds=time.perf_counter()-t0
                t=time.perf_counter()
                if arm in operators:result=jax.block_until_ready(q(ui,nui,operators[arm],cold))
                else:result=jax.block_until_ready(foms[arm](ui,nui))
                gpu_seconds=time.perf_counter()-t;t=time.perf_counter();result=jax.tree_util.tree_map(np.asarray,result);output_seconds=time.perf_counter()-t;total_seconds=time.perf_counter()-t0
                arrays=dict(fields=result[0],iterations=result[1],residuals=result[2])
                row=dict(case_id=record['case_id'],seed=record['seed'],arm=str(arm),rep=rep,gpu_seconds=gpu_seconds,
                    input_transfer_seconds=input_seconds,output_transfer_seconds=output_seconds,host_to_host_seconds=total_seconds,
                    error=d.fixed_initial_errors(result[0],truth),step_iteration_total=int(result[1].sum()))
                if arm in operators:
                    arrays.update(step_reasons=result[3],output_latents=result[4],initial_iterations=result[5],initial_reason=result[6],internal_latents=result[7],step_gradients=result[8],initial_gradient=result[9])
                    row.update(stationary=bool(result[8].max()<=1e-6 and result[9]<=1e-6),initial_iterations=int(result[5]))
                else:row.update(converged=bool(np.isfinite(result[2]).all() and result[2].max()<=.01))
                name=f"{record['case_id']}_{arm}_rep{rep}.npz";np.savez(out/name,**arrays)
                row.update(path=name,sha256=d.sha(out/name));report['invocations'].append(row);d.write_json(path,report)
        print('EQ CASE COMPLETE',record['case_id'],flush=True)
    report['complete']=True;report['elapsed_seconds']=time.monotonic()-started;d.write_json(path,report)


if __name__=='__main__':main()
