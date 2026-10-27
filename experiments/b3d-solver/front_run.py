"""Paired full64 validation: every error belongs to its timed invocation on this allocation."""
from pathlib import Path
import gc
import hashlib
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import c3common as C3
import qn3
import s3common as S
import fomrun
import psjax
import refstore as RS
import front_models as FM
import pod_policy
from front_metrics import summarize
from pod_tune import ah,score
from time_run import adaptive,fielddist
from projection_budget import ProjectionBudget

HERE=Path(__file__).resolve().parent
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(cx,result):
    cx.rep['frontier_progress']=dict(subjects_complete=list(result['subjects']),timing_rows=len(result['timing_repetitions']))
    cx.save();C3.C.dump(cx.out/'front_result.json',C3.C.clean(result))

def phase_records(cx,sub,result,j):
    u=jax.block_until_ready(sub['prepare'](j));nu=cx.nu[j]
    initial=u if sub['phase_init'] is None else jax.block_until_ready(sub['phase_init'](u))
    evolved=jax.block_until_ready(sub['phase_evolve'](initial,nu))
    fields=jax.block_until_ready(sub['phase_decode'](evolved))
    combined=jax.block_until_ready(sub['run'](u,nu));F=sub['output'](combined)
    phase_F=np.asarray(fields).reshape(6,-1);prod_F=np.asarray(F).reshape(6,-1)
    relative=float(np.linalg.norm(phase_F-prod_F)/(np.linalg.norm(prod_F)+1e-300))
    assert relative<=1e-12,(sub['id'],j,'split/fused control',relative)
    for rep in range(cx.cfg['phase_repetitions']):
        if sub['phase_init'] is None:
            result['phase_repetitions'].append(dict(subject=sub['id'],case=j,rep=rep,phase='initialization',seconds=None,
                output_sha256=ah(u),scope='four diagnostic cases only; indicative components',status=sub['initialization_status']))
        else:
            C3.burn(cx.cfg['burn_seconds']);t=time.perf_counter();initial=jax.block_until_ready(sub['phase_init'](u));sec=time.perf_counter()-t
            result['phase_repetitions'].append(dict(subject=sub['id'],case=j,rep=rep,phase='initialization',seconds=sec,
                output_sha256=ah(initial[0]),scope='four diagnostic cases only; indicative components'))
        C3.burn(cx.cfg['burn_seconds']);t=time.perf_counter();evolved=jax.block_until_ready(sub['phase_evolve'](initial,nu));sec=time.perf_counter()-t
        result['phase_repetitions'].append(dict(subject=sub['id'],case=j,rep=rep,phase='evolve',seconds=sec,
            output_sha256=ah(evolved['W_all'] if sub['kind']=='bank' else evolved[0] if sub['kind']=='POD-full-residual' else evolved['out'] if sub['kind']=='FD-central' else evolved),
            scope='four diagnostic cases only; indicative components'))
        C3.burn(cx.cfg['burn_seconds']);t=time.perf_counter();fields=jax.block_until_ready(sub['phase_decode'](evolved));sec=time.perf_counter()-t
        result['phase_repetitions'].append(dict(subject=sub['id'],case=j,rep=rep,phase='decode',seconds=sec,
            output_sha256=ah(fields),scope='four diagnostic cases only; indicative components',split_fused_relative=relative))

def certify_case(cx,sub,o,j,archive,archive_meta):
    W=sub['weights'](o);C=sub['state'](o);valid=sub['valid'](o);extra={}
    arrays={}
    if sub['kind']=='bank':
        cfg=sub['config'];st=sub['st'];steps=int(round(.25/cfg['dt']))
        gamma=np.asarray(sub['certifier'](jnp.asarray(C),cx.nu[j],sub['data']))
        arrays['fresh_gamma']=gamma
        cert=dict(valid=valid,steps=steps,nonstationary=int(np.sum(~(gamma<=1e-3))),fresh_gamma=gamma.tolist())
        if cfg.get('archived_key'):
            key=cfg['archived_key'];old=archive[key][j];entry=archive_meta['entries'][key];source=entry['qualification']
            assert source['qualified'] and source['Q1_pass']
            delta=fielddist(W,old,st.lean['L'],float(jnp.linalg.norm(cx.U[j])))
            relative=float(np.linalg.norm(W-old)/(np.linalg.norm(old)+1e-300))
            original=entry['original_qualification_record']
            old_distance=original['dist_adt'][j] if original and 'dist_adt' in original else source['Q1_source_record']['dist_worst']
            transfer=bool(relative<=1e-9 and delta<=1e-8 and old_distance+delta<=2.5e-4)
            q2_margin=source['Q2']['worst_dist_qp']+delta<=2.5e-4
            cert.update(distance_tight_bound=old_distance+delta,source_field_distance=delta,source_coefficient_relative=relative,
                        anchors_usable=True,common_rollout_check=bool(valid and transfer),Q1_transfer=transfer,
                        certification_basis='exact archived trajectory identity plus triangle bound and current fresh certificates')
            extra.update(archived_Q2_pass=bool(q2_margin),archived_Q2_source=entry['qualification'],source_pins=entry)
        else:
            anchors={};anchor_valid=[]
            for level,gt in [('tight',1e-7),('tighter',1e-8)]:
                if level not in sub['adaptive_queries']:sub['adaptive_queries'][level]=adaptive(st,cfg['dt'],gt,scheme=cfg['scheme'])
                q,sch=sub['adaptive_queries'][level]
                a=jax.block_until_ready(q(cx.U[j],cx.nu[j],sub['data'],sch));g=np.asarray(sub['certifier'](a['stats']['states'],cx.nu[j],sub['data']))
                av=bool(np.isfinite(np.asarray(a['W'])).all() and int(a['stats']['nfail'])==0 and np.isfinite(g).all() and np.all(g<=gt))
                anchors[level]=np.asarray(a['W']);anchor_valid.append(av)
                arrays.update({level+'_W':np.asarray(a['W']),level+'_states':np.asarray(a['stats']['states']),level+'_fresh_gamma':g})
                cert[level]=dict(valid=av,iterations=int(a['stats']['it_sum']),fresh_gamma=g.tolist())
            tt=fielddist(anchors['tight'],anchors['tighter'],st.lean['L'],float(jnp.linalg.norm(cx.U[j])))
            distance=fielddist(W,anchors['tight'],st.lean['L'],float(jnp.linalg.norm(cx.U[j])))
            usable=bool(all(anchor_valid) and tt<=2.5e-5)
            cert.update(distance_tight=distance,tight_tighter_distance=tt,anchors_usable=usable,
                        common_rollout_check=bool(valid and usable and distance<=2.5e-4))
        return cert,extra,arrays
    if sub['kind']=='POD-full-residual':
        detail=sub['pod']['validate'](cx.U[j],cx.nu[j],prod=o,need_tighter=True)
        for k in ('Wtight','Ctight','gamma_tight','Wtighter','Ctighter','gamma_tighter'):arrays[k]=np.asarray(detail.pop(k))
        pc=detail['production_certificate'];valid=pc['valid']
        usable=bool(detail['tight_certificate']['anchor_ok'] and detail['tighter_certificate']['anchor_ok'] and detail['tight_tighter_distance']<=2.5e-5)
        cert=dict(detail,valid=valid,steps=sub['pod']['steps'],nonstationary=pc['nonstat'],anchors_usable=usable,
                  common_rollout_check=bool(valid and usable and detail['prod_to_tight_distance']<=2.5e-4))
        return cert,dict(full_residual=True,hyper_reduced=False,sampled_full_gradient='not applicable'),arrays
    return dict(valid=valid,steps=None,nonstationary=None,common_rollout_check=valid,
                complete_stage_evidence=fomrun.FOMs.converged(sub['spec'],o['info'],sub['output'](o))[1]),dict(Q2='not applicable to full solver'),arrays

def own_quadrature(cx,sub,result,prod):
    cfg=sub['config'];st=sub['st'];main=cfg['rule'];next_rule,finest=cfg['successors'];W={j:prod[j]['W'] for j in cx.cases}
    adv=jax.jit(qn3.adv);a={j:np.asarray(adv(jnp.asarray(W[j]),sub['data'])) for j in cx.cases}
    dnext=st.data(next_rule);q=S.make_query(sub['spec'],st);next_W={};records={}
    cx.save()
    for j in cx.cases:
        o=jax.block_until_ready(q(cx.U[j][None],jnp.asarray([cx.nu[j]]),dnext));w=np.asarray(o['W'][:,0]);C=np.asarray(o['W_all'][:,0]);next_W[j]=w
        av=np.asarray(adv(jnp.asarray(W[j]),dnext));den=np.linalg.norm(av,axis=1);num=np.linalg.norm(a[j]-av,axis=1)
        rho=np.divide(num,den,out=np.full_like(num,np.inf),where=den>0);rho=np.where((den==0)&(np.linalg.norm(a[j],axis=1)==0),0.,rho)
        gamma=np.asarray(sub['certifier'](o['W_all'][:,0],cx.nu[j],dnext))
        dist=fielddist(W[j],w,st.lean['L'],float(jnp.linalg.norm(cx.U[j])))
        records[str(j)]=dict(valid_next=S.run_valid(sub['spec'],o,0),distance=dist,rho=rho.tolist(),fine_nonstationary=int(np.sum(~(gamma<=1e-3))),fine_steps=q.steps)
        np.savez_compressed(cx.out/'data'/cfg['id']/f'quad_c{j:03d}.npz',W=W[j],W_next=w,C_next=C,adv_main=a[j],adv_next=av,rho=rho,fresh_gamma_next=gamma)
        cx.save()
    st.drop(next_rule);del dnext;gc.collect()
    dfine=st.data(finest)
    cx.save()
    for j in cx.cfg['finer_check_cases']:
        o=jax.block_until_ready(q(cx.U[j][None],jnp.asarray([cx.nu[j]]),dfine));w=np.asarray(o['W'][:,0]);g=np.asarray(sub['certifier'](o['W_all'][:,0],cx.nu[j],dfine))
        records[str(j)].update(valid_finer=S.run_valid(sub['spec'],o,0),finer_distance=fielddist(next_W[j],w,st.lean['L'],float(jnp.linalg.norm(cx.U[j]))),
                               finest_nonstationary=int(np.sum(~(g<=1e-3))),finest_steps=q.steps)
        np.savez_compressed(cx.out/'data'/cfg['id']/f'finer_c{j:03d}.npz',W=w,C=np.asarray(o['W_all'][:,0]),fresh_gamma=g)
        cx.save()
    passed=all(r['valid_next'] and r['distance']<=2.5e-4 and np.isfinite(r['rho']).all() and max(r['rho'])<=.116 for r in records.values())
    passed=passed and all(records[str(j)]['valid_finer'] and records[str(j)]['finer_distance']<=2.5e-4 for j in cx.cfg['finer_check_cases'])
    result['subjects'][cfg['id']]['qualification']['Q2']=dict(passed=bool(passed),target=main,next_rule=next_rule,finest_rule=finest,
        scope='all64 own-trajectory rho and main→next; original first8 next→finest validity/distance',case_records=records)
    result['subjects'][cfg['id']]['qualification']['method_specific_extra_masks']['finer_witness_fresh_stationarity']=dict(
        next_bad=sum(r['fine_nonstationary'] for r in records.values()),next_steps=sum(r['fine_steps'] for r in records.values()),
        finest_bad=sum(r.get('finest_nonstationary',0) for r in records.values()),finest_steps=sum(r.get('finest_steps',0) for r in records.values()),
        interpretation='supplementary assessment retained separately; original Q2 validity/distance predicate unchanged')
    st.drop(finest);del dfine;gc.collect()

def run(cx,POD):
    cfg=cx.cfg;assert cx.cases==list(range(64)) and cfg['timing_reps']>=3
    am=json.loads((HERE/'inputs/front_archive.json').read_text());assert file_sha(HERE/'inputs/front_archive.npz')==am['archive_sha256']
    archive=np.load(HERE/'inputs/front_archive.npz',allow_pickle=False)
    common_setup=time.perf_counter()-cx.t0
    shared={};before=FM.memory();t=time.perf_counter();pod_subjects=pod_policy.build(POD,HERE.parent/'b3d-pod/inputs/Phi.npz',True)
    shared['POD']=dict(seconds=time.perf_counter()-t,memory_before=before,memory_after=FM.memory(),metadata=pod_subjects[0]['metadata'])
    for j in cx.cases:cx.ref(j)
    cohort_audit=json.loads((cx.out/'cohort_identity.json').read_text())
    times=[round(.05*k,10) for k in range(6)]
    result=dict(schema_version=1,cohort=dict(name='val64',seed=923801,count=64,ordered_cases=cx.cases,ordered_case_ids=cx.cases,
        eligible_cases=[j for j in cx.cases if cx.refs[j][2]['eligible']],table_sha256=cx.tab['sha256'],canonical_table_sha256=cx.tab['sha256'],
        raw_regeneration_sha256=cohort_audit['actual_sha256'],normalisation_restore=cohort_audit['normalisation_audit'],exposed_validation=True),
        reference=dict(dimension=3,cohort='val64',kind='ST',field='finest',store_version=RS.store_version(),
            status_sha256=file_sha(HERE/'vendor/e0/store/STATUS.json'),uncertainty='conditional Richardson',output_times=times,
            scoring_lattice=dict(nodes_per_axis=65,interior_per_axis=63,flattening='x slowest',shape=[6,63**3])),
        contracts=dict(primary='descriptor',descriptor=dict(contains_canonical_s_star=True,cohort_and_descriptor_construction='common offline setup',
            includes_native_input_preparation=True),prepared_native=dict(cross_method_inputs_differ=True),
            output=dict(shape=[6,63**3],points='fixed interior of65-node mesh',residency='device; synchronization included; host transfer/scoring/storage excluded for every method')),
        equations=dict(bank='inherited analytic continuum advection quadrature Bc*Dc; mesh-discrete sine diffusion eigenvalues and DST test moments; original equations unchanged',
                       POD='full central finite-difference residual projected on frozen POD basis; no hyper-reduction',
                       FD='central finite differences and damped CN startup',PS='oversampled sine pseudo-spectral ETDRK4; not exact de-aliasing'),
        subjects={},case_results={},timing_repetitions=[],phase_repetitions=[],matching={},frontiers={},shared_setup=shared,
        common_setup=dict(seconds=common_setup,scope='shared provenance/descriptor construction and diagnostic input cache; online descriptor preparation still timed on every invocation'),
        execution=dict(source_commit=cx.rep['source_commit'],job_id=cx.rep['job_id'],gpu=cx.rep['gpu'],config_sha256=cx.rep['config_sha256'],
                       backend=cx.rep['backend'],jax=cx.rep['jax'],x64=True,matmul_precision='highest',complete=False))
    foms=fomrun.FOMs();anchor_cfg=next(s for s in cfg['subjects'] if s['id']==cfg['anchor'])
    frozen_path=HERE/'inputs/frozen-fom.json';frozen=json.loads(frozen_path.read_text())
    assert frozen['cases']['table_sha256']==cx.tab['sha256']
    ps_control=psjax.gate(cx.tab,cases=(0,5),N=14,dt=.025)
    assert ps_control['pass'] and ps_control['control_fires'],ps_control
    result['implementation_controls']=dict(N14_port_and_half_step=ps_control)
    projection=ProjectionBudget(cx,result)
    anchor=FM.bank(cx,anchor_cfg,shared);anchor_hashes={};rng=np.random.default_rng(cfg['timing_seed'])
    subjects=[cfg['subjects'][i] for i in rng.permutation(len(cfg['subjects']))]
    for sc in subjects:
        sub=anchor if sc['id']==cfg['anchor'] else FM.bank(cx,sc,shared) if sc['kind']=='bank' else FM.pod(cx,sc,pod_subjects,shared) if sc['kind']=='POD' else FM.fom(cx,sc,foms)
        sid=sc['id'];folder=cx.out/'data'/sid;folder.mkdir(parents=True,exist_ok=False)
        operator_identity={}
        if sub['kind']=='bank':
            X,w=sub['st'].rules[sc['rule']]
            mode_hash=C3.sha(np.asarray(sub['st'].tb.kx[:sc['M']],np.int64))
            rule_hash=hashlib.sha256(np.ascontiguousarray(X,np.float64).tobytes()+np.ascontiguousarray(w,np.float64).tobytes()).hexdigest()
            operator_identity=dict(mode_hash=mode_hash,rule_sha256=rule_hash,M=sc['M'],r=sc['r'],m=len(w),
                quadrature_Gram_condition=sub['st'].mf_cond(sc['rule'])['cond'])
            if sc.get('archived_key'):
                original=am['entries'][sc['archived_key']]['original_qualification_record']
                if original and 'mode_hash' in original:assert mode_hash==original['mode_hash']
                if original and 'rule_sha256' in original:assert rule_hash==original['rule_sha256']
        result['subjects'][sid]=dict(config=sc,provenance=dict(bank_sha256=sub['st'].bk['sha256']) if sub['kind']=='bank' else sub.get('pod',{}).get('metadata',{}),
            setup_seconds=sub['setup_seconds'],setup_peak_bytes=sub['setup_memory_after']['whole_job_cumulative_peak_bytes'],
            setup_memory=dict(before=sub['setup_memory_before'],after=sub['setup_memory_after'],peak_scope='cumulative whole job'),
            shared_setup_keys=sub['shared_setup_keys'],operator_identity=operator_identity,input_contract=sub['input_contract'],qualification=dict(status='pending',Q1={},Q2={},scope='full64 exposed validation',source_pins={},common_rollout_check={},method_specific_extra_masks={}))
        if sc['kind'] in ('PS','FD'):
            source=next((v for v in frozen['picks'].values() if v['key']==fomrun.key(tuple(sc['spec']))),None)
            if sc['kind']=='FD':
                assert source and source['calibration']['status']=='qualified' and source['calibration']['chosen']==sc['spec'][4]
            result['subjects'][sid]['provenance']=dict(frozen_manifest_sha256=file_sha(frozen_path),source=frozen['source'],source_sha256=frozen['sha256'],
                calibration=source.get('calibration') if source else None,N14_cpu_prior=frozen.get('N14_cpu_prior') if sc['spec'][1]==14 else None,
                old_timing_values_not_used=True)
        result['case_results'][sid]={};production={}
        warm=time.perf_counter()
        for subject in (anchor,sub):jax.block_until_ready(subject['run'](jax.block_until_ready(subject['prepare'](0)),cx.nu[0]))
        result['subjects'][sid]['compile_and_warm_seconds']=time.perf_counter()-warm
        for j in rng.permutation(cx.cases):
            j=int(j);first=None;expected=None
            for rep in range(cfg['max_timing_reps']):
                trip=[]
                for role,subject in [('A1',anchor),('B',sub),('A2',anchor)]:
                    C3.burn(cfg['burn_seconds']);t0=time.perf_counter();u=jax.block_until_ready(subject['prepare'](j));t1=time.perf_counter()
                    o=jax.block_until_ready(subject['run'](u,cx.nu[j]));t2=time.perf_counter();F=subject['output'](o)
                    oh=ah(F);state=subject['state'](o);ch=ah(state) if state is not None else None
                    if subject['id']==cfg['anchor']:
                        if j in anchor_hashes:assert anchor_hashes[j]==(oh,ch)
                        anchor_hashes[j]=(oh,ch)
                    row=dict(invocation_id=f'{sid}:c{j}:r{rep}:{role}',subject=subject['id'],case=j,rep=rep,bracket=f'{sid}:c{j}:r{rep}',role=role,
                        given_native_seconds=t2-t1,input_prep_seconds=t1-t0,descriptor_seconds=t2-t0,score=score(cx,F,j),
                        output_sha256=oh,coefficient_sha256=ch)
                    trip.append(row)
                    if role=='B':
                        if expected is not None:assert expected==(oh,ch)
                        expected=(oh,ch)
                        if first is None:first=(o,row)
                dr=trip[2]['descriptor_seconds']/trip[0]['descriptor_seconds'];nr=trip[2]['given_native_seconds']/trip[0]['given_native_seconds']
                for row in trip:row.update(anchor_drift_ratio=dr,timing_admitted=bool(1/1.1<=dr<=1.1),native_anchor_drift_ratio=nr,native_timing_admitted=bool(1/1.1<=nr<=1.1))
                result['timing_repetitions']+=trip
                seen=[r for r in result['timing_repetitions'] if r['subject']==sid and r['case']==j and r['role']=='B']
                if rep+1>=cfg['timing_reps'] and any(r['timing_admitted'] for r in seen) and any(r['native_timing_admitted'] for r in seen):break
            o,row=first;cert,extra,arrays=certify_case(cx,sub,o,j,archive,am);W=sub['weights'](o);C=sub['state'](o)
            if C is None:arrays['fields']=np.asarray(sub['output'](o))
            else:arrays.update(W=W,C=C)
            path=folder/f'c{j:03d}.npz';np.savez_compressed(path,**arrays)
            production[j]=dict(W=W,C=C)
            result['case_results'][sid][str(j)]=dict(score=row['score'],invocation_id=row['invocation_id'],output_sha256=row['output_sha256'],
                coefficient_sha256=row['coefficient_sha256'],raw=dict(path=str(path.relative_to(cx.out)),sha256=file_sha(path)),
                certificate=cert,valid=cert['valid'],common_rollout_check=cert['common_rollout_check'],method_specific_extra_masks=extra,work=sub['work'](o),
                descriptor=C3.C.b3.table_row(cx.tab,j))
            if W is not None and row['score'].get('finite',True):
                result['case_results'][sid][str(j)]['projection_budget']=projection.measure(sub,j,W,row['score']['e_t'])
            save(cx,result)
        rows=list(result['case_results'][sid].values());qual=result['subjects'][sid]['qualification']
        common=all(r['common_rollout_check'] for r in rows);valid=all(r['valid'] for r in rows)
        nonstat=sum(r['certificate']['nonstationary'] or 0 for r in rows);steps=sum(r['certificate']['steps'] or 0 for r in rows)
        q1=bool(common and valid and (not steps or nonstat<=.01*steps))
        qual.update(Q1=dict(passed=q1,nonstationary=nonstat,steps=steps,scope='all64 production, full-trajectory tight and tighter anchors or explicit original identity-bound transfer'),
                    common_rollout_check=dict(passed=common,scope='finite/valid trajectory plus verified full-field tight distance'),
                    method_specific_extra_masks=dict(production_valid=valid,hyper_reduced=False if sc['kind']=='POD' else None))
        if sub['kind']=='bank':
            if sc.get('archived_key'):
                qp=all(r['method_specific_extra_masks']['archived_Q2_pass'] for r in rows)
                qual['Q2']=dict(passed=qp,scope='original qualified quadrature evidence bound to reproduced trajectories',source=am['entries'][sc['archived_key']])
            else:own_quadrature(cx,sub,result,production)
        else:qual['Q2']=dict(passed=True,applicability='not a quadrature-reduced bank solve')
        qual['status']='qualified' if q1 and qual['Q2']['passed'] else 'unqualified; retained diagnostic data'
        for j in cfg['phase_cases']:phase_records(cx,sub,result,j)
        result['subjects'][sid]['memory_after_all_work']=FM.memory();save(cx,result)
        if sub is not anchor and sub['kind']=='bank':
            for rule in list(sub['st']._d):sub['st'].drop(rule)
        del sub,production,first,o;gc.collect()
    manifest=[dict(path=str(p.relative_to(cx.out)),sha256=file_sha(p),bytes=p.stat().st_size) for p in sorted((cx.out/'data').rglob('*.npz'))]
    result['execution'].update(complete=True,elapsed_seconds=time.perf_counter()-cx.t0,whole_job_memory=FM.memory(),raw_file_manifest=manifest,
        driver_gpu_hours_lower_bound=(time.perf_counter()-cx.t0)/3600,Slurm_gpu_hours='collector JOB-LEDGER after completion')
    summarize(result);save(cx,result);cx.rep['scope']='same-allocation full64 validation frontier'
