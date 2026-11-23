"""All exposed initial fits; never generate a new test cohort in this driver."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
import support as S
import shared_initializer as I
import legacy_phase as P
np,jax,jnp,H,F,SR=S.np,S.jax,S.jnp,S.H,S.F,S.SR
HERE=Path(__file__).resolve().parent

def summarize(rows):
    keys={(x['family'],x['input_path'],x['gtol'],x['case_key']) for x in rows}
    complete=len(rows)==384 and len(keys)==384
    out={'records':len(rows),'expected_records':384,'unique_records':len(keys),'complete':complete,
         'all_repaired_solver_pass':complete and all(x['repaired_pass'] for x in rows),
         'all_initializer_qualified':complete and all(x.get('initializer_qualified',False) for x in rows),
         'selected_fits_with_tolerance_pair_checks':sum('initial_field_agreement_pass' in x for x in rows),
         'selected_fits_passing_tolerance_pair_checks':sum(x.get('initial_field_agreement_pass',False) for x in rows),
         'selected_fits_passing_old_new_initial_agreement':sum(x.get('old_new_initial_agreement_pass',False) for x in rows),
         'groups':{}}
    for family in ('learned','pod'):
        for path in ('phase','query'):
            for tol in (1e-7,1e-9,1e-11):
                group=[x for x in rows if x['family']==family and x['input_path']==path and x['gtol']==tol]
                out['groups'][f'{family}/{path}/{tol:g}']={
                    'count':len(group),
                    'original_phase_failures':[x['case_key'] for x in group if not x['original_phase_pass']],
                    'legacy_compiled_core_failures':[x['case_key'] for x in group if not x['original_compiled_pass']],
                    'common_evaluator_baseline_failures':[x['case_key'] for x in group if not x['common_baseline_pass']],
                    'qr_only_failures':[x['case_key'] for x in group if not x['qr_only_pass']],
                    'repaired_failures':[x['case_key'] for x in group if not x['repaired_pass']],
                    'phase_objective_guard_failures':[x['case_key'] for x in group if not x['phase_objective_quality']],
                    'initializer_qualification_failures':[x['case_key'] for x in group if not x.get('initializer_qualified',False)],
                    'max_old_new_field_gap':max([x['old_new_field_gap'] for x in group],default=0),
                    'max_selected_gradient':max([x['repaired_gradient'] for x in group],default=0)}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());args.out.mkdir(parents=True,exist_ok=False)
    assert cfg['cohorts']==[{'seed':202609202,'count':16},{'seed':2026111901,'count':16}]
    assert cfg['gtols']==[1e-7,1e-9,1e-11] and cfg['policy']==I.POLICY
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    manifest=json.loads((HERE/'PROVENANCE.json').read_text())
    for rel,entry in manifest['preserved_sources'].items():assert S.sha(HERE/rel)==entry['sha256'],rel
    S.dump(args.out/'provenance.json',{'config':cfg,'config_sha256':S.sha(args.config),
        'source_commit':os.environ.get('SOURCE_COMMIT'),'job_id':os.environ.get('SLURM_JOB_ID'),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader'],text=True).strip(),
        'backend':jax.default_backend(),'x64':jax.config.jax_enable_x64,'matmul_precision':str(jax.config.jax_default_matmul_precision),
        'sources':{str(p.relative_to(HERE)):S.sha(p) for p in sorted(HERE.rglob('*.py'))},'assets':manifest})
    rows=[];setup=[];begin=time.perf_counter();geom=F.geometry(cfg['n'])
    for family in cfg['families']:
        start=time.perf_counter()
        basis,binfo=S.learned_basis(cfg['n']) if family=='learned' else S.pod_basis(cfg['n'])
        basis=jnp.asarray(basis)
        params,codes=S.load_head(HERE/'assets'/('head_k8_n64.npz' if family=='learned' else 'pod_head_k8_n64.npz'))
        candidates=np.asarray(H.head_apply(params,jnp.asarray(codes)))
        setup.append({'family':family,'wall_seconds':time.perf_counter()-start,'basis':binfo})
        S.dump(args.out/'setup.json',setup)
        for cohort in cfg['cohorts']:
            parameters=F.parameters(cohort['seed'],cohort['count'])
            for ci,par in enumerate(parameters):
                initial=F.initial(cfg['n'],par);norm=float(np.linalg.norm(initial))
                for path in cfg['input_paths']:
                    physical=jnp.asarray(initial) if path=='phase' else S.query_input(jnp.asarray(initial),geom)
                    target,center=S.target_projection(physical,basis);target=np.asarray(target)
                    path_rows=[];path_fields={}
                    for gtol in cfg['gtols']:
                        key=f"{cohort['seed']}/{ci}";stamp=f"{family}_{path}_{cohort['seed']}_{ci:02d}_{gtol:g}"
                        tick=time.perf_counter()
                        oldcode,old=P.fit_head(target,params,codes,candidates,starts=4,gtol=gtol,budget=250)
                        old_time=time.perf_counter()-tick
                        tick=time.perf_counter();code,fit=I.fit_head(target,params,codes,candidates,gtol=gtol)
                        fit_time=time.perf_counter()-tick
                        old_selected=old['starts'][old['best_index']]
                        orig=fit['original_compiled_infos'][fit['original_compiled_selected']]
                        common=fit['baseline_starts'][fit['baseline_selected']]
                        qr=fit['qr_only_starts'][fit['qr_only_selected']]
                        rawold=fit['baseline_starts'][fit['baseline_raw']]
                        _,_,old_direct=I.snapshot(oldcode,jnp.asarray(target),params)
                        old_coeff=np.asarray(H.head_apply(params,jnp.asarray(oldcode)))
                        new_coeff=np.asarray(H.head_apply(params,jnp.asarray(code)))
                        old_field=np.asarray(S.reconstruct_initial(jnp.asarray(old_coeff),center,basis))
                        new_field=np.asarray(S.reconstruct_initial(jnp.asarray(new_coeff),center,basis))
                        path_fields[gtol]=new_field
                        phase_snapshots=[I.snapshot(z,jnp.asarray(target),params)[2] for z in old['all_final_codes']]
                        phase_raw=min(range(len(phase_snapshots)),key=lambda i:phase_snapshots[i]['cost'])
                        phase_raw_check=phase_snapshots[phase_raw]
                        residual0,jac0,_=I.snapshot(rawold['code'],jnp.asarray(target),params)
                        residual1,jac1,new_direct=I.snapshot(code,jnp.asarray(target),params)
                        r0,r1=np.asarray(residual0),np.asarray(residual1)
                        phase_res,phase_jac,_=I.snapshot(oldcode,jnp.asarray(target),params)
                        auditpath=args.out/'audit_arrays'/f'{stamp}.npz';auditpath.parent.mkdir(exist_ok=True)
                        np.savez_compressed(auditpath,returned_code=code,target=target,
                            returned_residual=r1,returned_jacobian=np.asarray(jac1),
                            baseline_raw_code=rawold['code'],baseline_raw_residual=r0,
                            baseline_raw_jacobian=np.asarray(jac0),phase_old_code=oldcode,
                            phase_old_residual=np.asarray(phase_res),phase_old_jacobian=np.asarray(phase_jac))
                        detail={'family':family,'input_path':path,'cohort':cohort,'case':ci,'gtol':gtol,
                            'parameter':par,'original_initial_norm':norm,'center':np.asarray(center),'target':target,
                            'initial_input_sha256':hashlib.sha256(np.asarray(physical).tobytes()).hexdigest(),
                            'original_phase':old,'original_phase_code':oldcode,'original_phase_fresh_common_check':old_direct,
                            'original_phase_all_common_checks':phase_snapshots,'original_phase_common_raw':phase_raw,
                            'repair':fit,'original_phase_seconds':old_time,'repair_including_baseline_seconds':fit_time,
                            'control_scope':'legacy host phase and standalone compiled-core replay; not a bitwise whole-query replay',
                            'selected_objective_delta_direct':new_direct['cost']-rawold['cost'],
                            'selected_objective_delta_residual_identity':float(.5*np.dot(r1-r0,r1+r0)),
                            'audit_arrays':str(auditpath.relative_to(args.out)),
                            'audit_arrays_sha256':S.sha(auditpath),
                            'phase_objective_quality':I.endpoint_pass(new_direct['cost'],phase_raw_check['cost']),
                            'initial_field_error':float(np.linalg.norm(new_field-np.asarray(physical))/norm),
                            'original_phase_initial_field_error':float(np.linalg.norm(old_field-np.asarray(physical))/norm),
                            'old_new_field_gap':float(np.linalg.norm(new_field-old_field)/norm),
                            'old_new_coefficient_gap':float(np.linalg.norm(new_coeff-old_coeff)/norm)}
                        S.dump(args.out/'fits'/f'{stamp}.json',detail)
                        row={'case_key':key,'family':family,'input_path':path,'gtol':gtol,'detail':f'fits/{stamp}.json',
                            'original_phase_pass':old_selected['reason']=='stationarity',
                            'original_compiled_pass':orig['reason']=='stationarity',
                            'common_baseline_pass':common['reason']=='stationarity',
                            'qr_only_pass':qr['reason']=='stationarity',
                            'repaired_pass':fit['qualified'],'repaired_gradient':new_direct['gradient'],
                            'old_new_field_gap':detail['old_new_field_gap'],
                            'phase_objective_quality':detail['phase_objective_quality'],
                            'initial_field_error':detail['initial_field_error']}
                        rows.append(row);path_rows.append(row)
                    pairs=[]
                    for ia,tol0 in enumerate(cfg['gtols']):
                        for tol1 in cfg['gtols'][ia+1:]:
                            gap=float(np.linalg.norm(path_fields[tol0]-path_fields[tol1])/norm)
                            pairs.append({'gtols':[tol0,tol1],'physical_relative_gap':gap,'passed':gap<=1e-6})
                    agreement=all(p['passed'] for p in pairs)
                    S.dump(args.out/'agreement'/f'{family}_{path}_{cohort["seed"]}_{ci:02d}.json',
                           {'pairs':pairs,'all_pass':agreement,'normalization':'original physical initial field norm'})
                    for row in path_rows:
                        row['initial_field_agreement_pass']=agreement
                        row['old_new_initial_agreement_pass']=row['old_new_field_gap']<=1e-6
                        row['initializer_qualified']=bool(row['repaired_pass'] and row['phase_objective_quality']
                            and agreement and row['old_new_initial_agreement_pass'])
                    S.dump(args.out/'records.json',rows)
                print(f'{family} seed={cohort["seed"]} case={ci} complete; failures={sum(not r["repaired_pass"] for r in rows)}',flush=True)
                S.dump(args.out/'summary.json',{**summarize(rows),'elapsed_seconds':time.perf_counter()-begin})
    assert len(rows)==384
    print('development complete; all outcomes retained',flush=True)

if __name__=='__main__':main()
