"""Prepare immutable offline assets and evaluate matched complete NS3D queries.

Preparation is development-only. Final evaluation loads the actual prepared
bank, head, correction directions, POD bases and linear operators; it never
fits any of them. Every timed query includes initial fitting and field output.
"""
from __future__ import annotations
import gc,json,os,pickle,shutil,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.signal import resample
import ns3d_fom as F
import ns3d_model as D
import ns3d_rom as R
import operator_adapter as O
from comparison import burn
from extra03 import file_hash,load
from pilot import generate,write,sha


def finite_metadata(value):
    """Retain failed-solve metadata as explicit nulls while arrays keep NaNs."""
    if isinstance(value,np.ndarray):return finite_metadata(value.tolist())
    if isinstance(value,(tuple,list)):return [finite_metadata(x) for x in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value


def verify_assets(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['files'] and all(file_hash(path/name)==value for name,value in manifest['files'].items())
    return manifest


def prepare_assets(cfg,reuse,out):
    """All data entering this routine are recorded training/development assets."""
    reuse=Path(reuse);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    record=json.loads((reuse/'REUSE.json').read_text())
    assert all(file_hash(reuse/row['path'])==row['sha256'] for row in record['files'])
    bank=load(reuse/'frozen_bank.pkl');head=load(reuse/'head.pkl')
    assert head['config']['k']==cfg['k'] and head['config']['r']==cfg['r']
    assert bank['extra']['bank'].shape==(3*cfg['n']**3,cfg['r'])
    assert cfg['test_modes']>=4*(cfg['k']+max(cfg['q_values']))
    assert head['info']['training']['steps']>0,'A trained incumbent must compete before selecting an affine initialization'
    for name in ('frozen_bank.pkl','head.pkl','operator_statistics.pkl'):
        shutil.copy2(reuse/name,out/name)
    for spec in cfg['operators']:
        (out/spec['kind']).mkdir()
        shutil.copy2(reuse/spec['kind']/'best.pkl',out/spec['kind']/'best.pkl')
    with np.load(reuse/'pod.npz') as f:P=f['basis'][:,:cfg['k']+max(cfg['q_values'])]
    np.savez_compressed(out/'pod.npz',basis=P)
    G=jnp.asarray(bank['extra']['bank']);Q=jnp.asarray(bank['extra']['qr_Q']);Rb=jnp.asarray(bank['extra']['qr_R'])
    coefficients=jnp.asarray(bank['extra']['train_coefficients']);codes=jnp.asarray(head['codes'])
    residual=(coefficients-D.head(jax.device_put(head['params']),codes))@Rb.T
    eig,V=jnp.linalg.eigh(residual.T@residual)
    C=np.asarray(jnp.linalg.solve(Rb,V[:,::-1]));singular=np.sqrt(np.maximum(np.asarray(eig[::-1]),0.))
    Phi,lam,ids=R.test_modes(cfg['n'],cfg['test_modes']);A=np.asarray(jnp.asarray(Phi).T@G)
    Lfree=R.dense_galerkin_linear(np.asarray(Q),cfg['n']);Lpod=R.dense_galerkin_linear(P,cfg['n'])
    Apod=np.asarray(jnp.asarray(Phi).T@jnp.asarray(P))
    np.savez_compressed(out/'dense_weak_operators.npz',A=A,lam=lam,C=C,correction_singular=singular,
        pod_A=Apod,free_L=Lfree,pod_L=Lpod)
    write(ids,out/'dense_weak_test_modes.json')
    metric=np.asarray(Rb)@C
    assert np.max(abs(metric.T@metric-np.eye(cfg['r'])))<1e-7
    metadata=dict(source_reuse=record,configuration=cfg,selected_head=cfg['selected_head'],
        bank_rank=cfg['r'],latent_dimension=cfg['k'],final_cohort_opened=False,
        correction_direction_metric_orthogonality=float(np.max(abs(metric.T@metric-np.eye(cfg['r'])))),
        preparation='Training-only head residual covariance eigenvectors, physical weak projections and classical Galerkin diffusion matrices',
        runtime='Full-grid dense weak residual; initial projection, cold fitting, evolution and six vector-field outputs are charged')
    write(metadata,out/'metadata.json')
    write(dict(files={str(p.relative_to(out)):file_hash(p) for p in sorted(out.rglob('*')) if p.is_file()},
        final_cohort_opened=False),out/'manifest.json')
    del G,Q,Rb,coefficients,codes,residual,Phi,bank,head;gc.collect();jax.clear_caches()
    return metadata


def methods(cfg,assets):
    assets=Path(assets);verify_assets(assets);bank=load(assets/'frozen_bank.pkl');head=load(assets/'head.pkl')
    n=cfg['n'];k=cfg['k'];geom=F.geometry(n);dt=cfg['rom_dt'];steps=round(cfg['horizon']/dt)
    with np.load(assets/'pod.npz') as f:P=f['basis']
    with np.load(assets/'dense_weak_operators.npz') as f:ops={key:f[key] for key in f}
    ids=json.loads((assets/'dense_weak_test_modes.json').read_text())
    selector=(np.asarray([x['wave'] for x in ids])%n,np.asarray([x['polarization'] for x in ids]),np.asarray([x['kind']=='cos' for x in ids]),geom)
    shared=jax.device_put((bank['extra']['bank'],bank['extra']['qr_Q'],bank['extra']['qr_R'],ops['A'],selector,ops['lam']))
    theta=jax.device_put({key:head['params'][key] for key in ('h','h_lin')});codes=jax.device_put(head['codes']);result=[]
    for q in cfg['q_values']:
        result.append((f'nmrom_{cfg["selected_head"]}_q{q}','weak',
            R.make_run(dt,steps,steps//5,k,q,budget=cfg['rom_budget'],gtol=cfg['gtol'],cold_starts=cfg['cold_starts'],retain_states=True,dense_n=n),
            shared+(jax.device_put(ops['C'][:,:q]),theta,codes)))
    for rank in sorted({k+q for q in cfg['q_values']}):
        basis=P[:,:rank]
        args=(basis,basis,np.eye(rank),ops['pod_A'][:,:rank],selector,ops['lam'],np.empty((rank,0)),{},np.empty((1,0)))
        result.append((f'pod_weak_{rank}','weak',R.make_run(dt,steps,steps//5,0,rank,linear=True,budget=cfg['rom_budget'],gtol=cfg['gtol'],retain_states=True,dense_n=n),jax.device_put(args)))
        result.append((f'pod_galerkin_{rank}','galerkin',R.make_dense_galerkin_run(dt,steps,steps//5,n),
            jax.device_put((basis,ops['pod_L'][:rank,:rank],geom))))
    result.append(('free_bank_galerkin','galerkin',R.make_dense_galerkin_run(dt,steps,steps//5,n),
        (shared[1],jax.device_put(ops['free_L']),geom)))
    stats=load(assets/'operator_statistics.pkl')
    for spec in cfg['operators']:
        model=load(assets/spec['kind']/'best.pkl');assert model['spec']==spec
        for projected in cfg['operator_projection_variants']:
            result.append((spec['kind']+'_increment'+('_projected' if projected else '_raw'),'operator',O.make_query(spec,n,projected),
                jax.device_put((model['params'],stats['input_mean'],stats['input_std'],stats['output_scale'],geom))))
    for dt in cfg['fom_dts']:
        count=round(cfg['horizon']/dt);assert count%5==0
        result.append((f'fom_dt{dt}','fom',F.make_solver(dt,count,count//5),(geom,)))
    return result


def verify_replay(cfg,assets,source,limit=1e-8):
    """Real checkpoint and offline-asset replay before opening final cases."""
    source=Path(source);rows=json.loads((source/'timing_rows.json').read_text());rows={r['method']:r for r in rows if r['case']==0 and r['repetition']==0}
    parameter=F.parameters(cfg['dev_seed'],1)[0];u0=jnp.asarray(F.initial(cfg['n'],parameter));checks=[]
    with np.load(source/'timed_fields.npz') as fields:
        for name,kind,fn,args in methods(cfg,assets):
            expected=rows[name];value=jax.tree_util.tree_map(np.asarray,fn(u0,parameter[-1],*args));field=(value[0] if kind=='weak' else value).reshape(6,3,cfg['n'],cfg['n'],cfg['n'])
            reference=fields[f'{name}__case0']
            if expected['finite']:
                difference=float(np.linalg.norm(field-reference)/max(np.linalg.norm(reference),1e-300));field_passed=bool(np.isfinite(difference) and difference<=limit)
            else:
                difference=None;field_passed=bool(np.allclose(field,reference,rtol=limit,atol=limit,equal_nan=True))
            stopping=True
            if kind=='weak':stopping=bool(np.array_equal(value[1][1:3],np.asarray(expected['cold'])[1:3]) and np.array_equal(value[2][1],expected['steps'][1]) and np.array_equal(value[2][2],expected['steps'][2]))
            checks.append(dict(method=name,field_relative_difference=difference,expected_numerical_failure=not expected['finite'],identical_stopping=stopping,passed=field_passed and stopping))
    report=dict(passed=all(x['passed'] for x in checks),relative_tolerance=limit,checks=checks,final_cohort_opened=False)
    return report


def evaluate(cfg,assets,out,seed,count,report=None,smoke_methods=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);panel=methods(cfg,assets);n=cfg['n'];h=cfg['horizon']
    if smoke_methods is not None:
        assert cfg['evaluation_cohort']=='development' and n<=8 and count<=2
        panel=[row for row in panel if row[0] in smoke_methods]
        assert len(panel)==len(smoke_methods)
    report={} if report is None else report
    report.update(config=cfg,source_commit=os.environ.get('SOURCE_COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),
        backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
        final_cohort_opened=cfg['evaluation_cohort']=='final',complete=False,kind='NS3D frozen matched comparison',
        evaluation_cohort=cfg['evaluation_cohort'],cohort_seed=seed,cohort_count=count,method_names=[x[0] for x in panel],
        limitations=['Dense full-grid weak evaluation; no hyper-reduction speed claim','Warm bank lineage is not an independent complete training seed'])
    def stage(name):report['stage']=name;write(report,out/'result.json');print('STAGE',name,flush=True)
    stage('evaluation_data_generation')
    states,report['dev_data']=generate(n,cfg['dt'],h,count,seed,out/'dev_data.npz');parameters=F.parameters(seed,count)
    stage('reference_generation');refs=[];fine=[];refinement=[];fine_n=cfg['fine_reference_n'];check_n=cfg['refinement_n']
    def solver(nn,dt):
        steps=round(h/dt);return F.make_solver(dt,steps,steps//5),F.geometry(nn)
    reference,geom=solver(n,cfg['reference_dt']);fine_fn,fg=solver(fine_n,cfg['fine_reference_dt']);check_fn,cg=solver(check_n,cfg['refinement_dt'])
    fine_dir=out/'refinement';fine_dir.mkdir(exist_ok=True)
    for case,parameter in enumerate(parameters):
        same=np.asarray(reference(jnp.asarray(states[case,0]),parameter[-1],geom));small=np.asarray(fine_fn(jnp.asarray(F.initial(fine_n,parameter)),parameter[-1],fg))
        large=np.asarray(check_fn(jnp.asarray(F.initial(check_n,parameter)),parameter[-1],cg));lifted=small
        for axis in (-3,-2,-1):lifted=resample(lifted,check_n,axis=axis)
        difference=np.linalg.norm((lifted-large).reshape(6,-1),axis=1)/np.linalg.norm(large[0])
        np.savez_compressed(fine_dir/f'case{case}.npz',fine=large,parameter=parameter)
        refinement.append(dict(case=case,errors=difference,passed=bool(np.max(difference)<=cfg['refinement_budget']),fine_sha256=sha(large)))
        refs.append(same);fine.append(small)
    refs=np.stack(refs);fine=np.stack(fine)
    np.savez_compressed(out/'timing_references.npz',same_grid=refs,fine=fine,parameters=parameters)
    report['reference_refinement']=refinement;stage('compile_complete_queries')
    for name,kind,fn,args in panel:
        jax.block_until_ready(fn(jnp.asarray(states[0,0]),parameters[0,-1],*args));print('COMPILED',name,flush=True)
    rows=[];fields={};histories={};burn(3.);stage('paired_timing')
    for case,parameter in enumerate(parameters):
        u0=jnp.asarray(states[case,0]);nu=parameter[-1]
        for repetition in range(cfg['repetitions']):
            burn(1.);order=np.roll(np.arange(len(panel)),case+repetition)
            if repetition%2:order=order[::-1]
            for order_index,index in enumerate(order):
                name,kind,fn,args=panel[index];start=time.perf_counter();value=fn(u0,nu,*args);jax.block_until_ready(value);gpu=time.perf_counter()-start
                host=jax.tree_util.tree_map(np.asarray,value);field=(host[0] if kind=='weak' else host).reshape(6,3,n,n,n);total=time.perf_counter()-start
                finite=bool(np.isfinite(field).all());key=f'{name}__case{case}'
                if repetition and sha(fields[key])!=sha(field):key+=f'__rep{repetition}'
                fields[key]=field;errors=None;physical=None
                if finite:
                    errors=np.linalg.norm((field-refs[case]).reshape(6,-1),axis=1)/np.linalg.norm(states[case,0]);lifted=field
                    for axis in (-3,-2,-1):lifted=resample(lifted,fine_n,axis=axis)
                    physical=np.linalg.norm((lifted-fine[case]).reshape(6,-1),axis=1)/np.linalg.norm(fine[case,0])
                    if not np.isfinite(errors).all() or not np.isfinite(physical).all():finite=False;errors=None;physical=None
                row=dict(case=case,repetition=repetition,order_index=order_index,method=name,kind=kind,gpu_seconds=gpu,with_host_seconds=total,
                    field_sha256=sha(field),finite=finite,same_grid_errors=errors,physical_errors=physical)
                if kind=='weak':
                    metadata_finite=all(np.isfinite(np.asarray(x)).all() for x in (*host[1],*host[2]))
                    row.update(cold=finite_metadata(host[1]),steps=finite_metadata(host[2]),weak_metadata_finite=bool(metadata_finite),state_sha256=sha(host[3]));histories[f'{name}__case{case}__rep{repetition}']=host[3]
                rows.append(row)
            write(rows,out/'timing_rows.json');print('TIMED',case,repetition,flush=True)
    np.savez_compressed(out/'timed_fields.npz',**fields);np.savez_compressed(out/'latent_histories.npz',**histories)
    report['timing_summary']={}
    for name,kind,fn,args in panel:
        arm=[r for r in rows if r['method']==name];times=np.asarray([r['gpu_seconds'] for r in arm]);valid=[r for r in arm if r['finite']]
        report['timing_summary'][name]=dict(gpu_seconds_median=float(np.median(times)),invocations=len(arm),nonfinite_invocations=len(arm)-len(valid),
            timing_outliers_above_three_median=int(np.sum(times>3*np.median(times))),
            worst_evolved_same_grid=max(max(r['same_grid_errors'][1:]) for r in valid) if len(valid)==len(arm) else None,
            worst_evolved_physical=max(max(r['physical_errors'][1:]) for r in valid) if len(valid)==len(arm) else None)
    report['complete']=True;stage('complete');return report
