"""Bounded four-case fixed-mode mesh cost and separately labeled operator diagnostics."""
from pathlib import Path
from types import SimpleNamespace
import gc
import hashlib
import time
import numpy as np
import jax
import jax.numpy as jnp
import c3common as C3
import s3common as S
import qn3
import refstore as RS
import frozen_tables as FT
import mesh_models as MM
from front_models import memory
from pod_tune import ah,score
from projection_budget import ProjectionBudget

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(cx,result):
    cx.rep['mesh_progress']=dict(cells=sum(len(x) for x in result['case_results'].values()),timing_rows=len(result['timing_repetitions']))
    result['execution']['checkpoint_memory']=memory()
    C3.C.dump(cx.out/'mesh_result.json',C3.C.clean(result));cx.save()

def setup(cx,result,name,r,rules):
    before=memory();t=time.perf_counter();bank=C3.load_bank(name)
    tb=C3.Tables(65,bank,[512] if name=='M2' else [256,512,768,1024])
    st=S.Setting(dict(name=name,bank=name,Rp=r,a=3 if name=='M2' else 6),{name:bank},{name:tb},S.load_rules(rules))
    k=np.asarray(tb.kx[:st.M],np.int64).copy();assert st.M==(2052 if name=='M2' else 4097)
    mode=FT.build(65,bank,r,k);fresh,meta65=mode
    checks=dict(rows=MM.row_equivalence(fresh['GTb'],st.lean['GTb']),
                **{key:MM.relative(fresh[key],st.lean[key]) for key in ('L','A','lam')})
    assert max(checks.values())<=1e-12,checks
    checks['direct65']=FT.direct_checks(65,fresh,k)
    del fresh,mode;gc.collect()
    native,meta129=FT.build(129,bank,r,k);checks['direct129']=FT.direct_checks(129,native,k)
    result['frozen_modes'][name]=dict(bank_sha256=bank['sha256'],r=r,M=st.M,mode_hash=C3.sha(k),
        bank_rotation_prefix_sha256=C3.sha(bank['T'][:,:r]),meshes=[meta65,meta129],controls=checks)
    folder=cx.out/'data/operators';folder.mkdir(parents=True,exist_ok=True);p=folder/(name+'.npz')
    np.savez_compressed(p,kxyz=k,L65=np.asarray(st.lean['L']),L129=np.asarray(native['L']),
        A65=np.asarray(st.lean['A']),A129=np.asarray(native['A']),lambda65=np.asarray(st.lean['lam']),lambda129=np.asarray(native['lam']))
    result['operator_differences'][name]=dict(raw=dict(path=str(p.relative_to(cx.out)),sha256=sha(p)),
        A129_normalized_relative_to65=MM.relative(native['A']/np.sqrt(8),st.lean['A']),
        lambda129_relative_to65=MM.relative(native['lam'],st.lean['lam']),
        lambda_intervention='discrete diffusion eigenvalues AND their induced inherited residual/Jacobian row weighting',
        analytic_advection='unchanged continuum u*(ux+uy+uz) quadrature, not an upwind stencil')
    result['shared_setup'].append(dict(bank=name,seconds=time.perf_counter()-t,before=before,after=memory(),
        scope='shared frozen-bank65 validation and129 basis/Gram/test-moment assembly; cumulative whole-job peak'))
    norms={65:{},129:{}}
    for j in cx.cases:
        u129=jnp.asarray(C3.C.initial_interior(129,cx.tab,j));norms[65][j]=float(jnp.linalg.norm(cx.U[j]));norms[129][j]=float(jnp.linalg.norm(u129))
        c65,_=MM.ready(C3.TB.project(cx.U[j],st.lean));c129,_=MM.ready(C3.TB.project(u129,native))
        result['initial_projection_differences'][f'{name}_native129_c{j}']=dict(common65_distance=float(jnp.linalg.norm((c129-c65)@st.lean['L'])/cx.ref(j)[1]['judge']['n0']),
            used_in_operator_queries=False if name=='W1024' else None,held_projection='native65 for W1024 operator interventions')
    save(cx,result)
    return SimpleNamespace(name=name,bank=bank,st=st,native=native,k=k,norms=norms)

def rule_data(cx,result,b,rule,n,intervention=None):
    before=memory();t=time.perf_counter();base=b.st.data(rule)
    lean=b.st.lean if n==65 else b.native
    P=base['P'] if n==65 else base['P']*np.sqrt(8)
    if n==129:
        direct=C3.OM.test_block(129,*b.st.rules[rule],b.k)
        assert MM.relative(direct,P)<=1e-12
        del direct
    if intervention:
        assert b.name=='W1024'
        A=b.native['A']/np.sqrt(8) if intervention in ('composite','moments') else b.st.lean['A']
        lam=b.native['lam'] if intervention in ('composite','lambda') else b.st.lean['lam']
        lean=b.st.lean;P=base['P']
        d=MM.data_for(lean,base,A=A,lam=lam,P=P)
        if intervention=='composite':
            selected=np.arange(0,len(base['P']),max(1,len(base['P'])//1024))[:1024]
            X,w=b.st.rules[rule]
            P129_selected=C3.OM.test_block(129,X[selected],w[selected],b.k)
            assert MM.relative(P129_selected,base['P'][selected]*np.sqrt(8))<=1e-12
            native=dict(d,A=b.native['A'])
            result['operator_differences'][b.name].setdefault('normalization_controls',{})[rule]=MM.normalized_control(native,d,b.st.Rp)
            del native
    else:d=MM.data_for(lean,base,P=P)
    result['incremental_setup'].append(dict(bank=b.name,rule=rule,n=n,intervention=intervention,
        seconds=time.perf_counter()-t,before=before,after=memory(),predictor_QR='recomputed for actual A',m=len(b.st.rules[rule][1])))
    save(cx,result);return d

def prepare(cx,b,path,n,rule,j):
    if path=='mf':
        X=b.st.rules[rule][0]
        return jnp.asarray(C3.C.b3.blob_ic_3d(n,cx.tab,j,coords=X))
    return jnp.asarray(C3.C.initial_interior(n,cx.tab,j))

def score_returned(cx,F,j,n):return score(cx,MM.restrict(F,n),j)

def record_fields(cx,folder,F):
    # Native129 fields are deliberately split: each uncompressed time slice is below50MB.
    records=[]
    for i,field in enumerate(np.asarray(F)):
        p=folder/f'field_t{i}.npz';np.savez_compressed(p,field=field)
        records.append(dict(path=str(p.relative_to(cx.out)),sha256=sha(p),bytes=p.stat().st_size))
    return records

def validate(cx,result,b,qid,runner,u,scale,j,o,out_n):
    folder=cx.out/'data'/qid/f'c{j:03d}';folder.mkdir(parents=True,exist_ok=True)
    W=np.asarray(o['W'][:,0]);C=np.asarray(o['W_all'][:,0]);F=np.asarray(o['fields'])
    valid=bool(runner.valid(o) and np.isfinite(F).all())
    gamma=np.asarray(runner.certify(jnp.asarray(C),cx.nu[j],runner.data))
    arrays=dict(W=W,C=C,fresh_gamma=gamma);anchors={};av=[]
    n0=float(cx.ref(j)[1]['judge']['n0']);native_n=129 if b.name=='W1024' else runner.n
    native_n0=b.norms[native_n][j];native_L=b.native['L'] if native_n==129 else b.st.lean['L']
    for label,(query,sch,gt) in runner.anchors.items():
        a=MM.ready(query(u,cx.nu[j],scale,runner.data,sch));states=np.asarray(a['stats']['states'])
        g=np.asarray(runner.certify(jnp.asarray(states),cx.nu[j],runner.data))
        usable=bool(np.isfinite(states).all() and np.isfinite(g).all() and int(a['stats']['nfail'])==0 and np.all(g<=gt))
        anchors[label]=np.asarray(a['W']);av.append(usable)
        arrays.update({label+'_W':anchors[label],label+'_C':states,label+'_fresh_gamma':g,
                       label+'_step_reason':np.asarray(a['stats']['step_reason'])[:runner.steps],
                       label+'_step_it':np.asarray(a['stats']['step_it'])[:runner.steps]})
    common=MM.metric_distance(W,anchors['tight'],b.st.lean['L'],n0)
    tt=MM.metric_distance(anchors['tight'],anchors['tighter'],b.st.lean['L'],n0)
    native=MM.metric_distance(W,anchors['tight'],native_L,native_n0)
    native_tt=MM.metric_distance(anchors['tight'],anchors['tighter'],native_L,native_n0)
    usable=bool(all(av) and tt<=2.5e-5 and (out_n==65 or native_tt<=2.5e-5))
    sc=score_returned(cx,F,j,out_n)
    p=folder/'trajectory.npz';np.savez_compressed(p,**arrays)
    row=dict(score=sc,valid=valid,common_distance_tight=common,common_tight_tighter_distance=tt,
        native_distance_tight=native,native_tight_tighter_distance=native_tt,native_initial_norm=native_n0,common_initial_reference_norm=n0,
        native_distance_scope='full returned native field' if out_n==129 else f'additional native{native_n} field-span diagnostic; fixed-output contract judged on65',
        anchor_usable=usable,anchor_valid=av,nonstationary=int(np.sum(~(gamma<=1e-3))),steps=runner.steps,
        common_rollout_check=bool(valid and usable and common<=2.5e-4),
        native_rollout_check=bool(valid and usable and native<=2.5e-4) if out_n==129 else None,
        work=runner.work(o),output_sha256=ah(F),coefficient_sha256=ah(C),
        raw=dict(path=str(p.relative_to(cx.out)),sha256=sha(p)),actual_returned_fields=record_fields(cx,folder,F))
    result['case_results'].setdefault(qid,{})[str(j)]=row;save(cx,result)
    assert valid and all(av) and usable,(qid,j,'integrity/nonfinite/exhausted or inconsistent anchor; diagnostic stopped')
    return row,W

def qualify(result,qid):
    rows=list(result['case_results'][qid].values());steps=sum(r['steps'] for r in rows);bad=sum(r['nonstationary'] for r in rows)
    passed=bool(all(r['common_rollout_check'] and r['native_rollout_check'] is not False for r in rows) and bad<=.01*steps)
    return dict(passed=passed,nonstationary=bad,steps=steps,scope='four exposed cases only; common65 and declared growing-native distances, fresh every-step gamma; not full64')

def quadrature(cx,result,b,main_id,definitions,outputs):
    main,next_id,finer_id=definitions
    rows=[]
    for j in result['queries'][main_id]['cases']:
        Wa,Wb,Wc=[outputs[(qid,j)] for qid,runner in definitions]
        da,db=main[1].data,next_id[1].data
        va=np.asarray(qn3.adv(jnp.asarray(Wa),da));vb=np.asarray(qn3.adv(jnp.asarray(Wa),db))
        num=np.linalg.norm(va-vb,axis=1);den=np.linalg.norm(vb,axis=1)
        rho=np.divide(num,den,out=np.full_like(num,np.inf),where=den>0)
        rho=np.where((den==0)&(np.linalg.norm(va,axis=1)==0),0.,rho)
        n0=float(cx.ref(j)[1]['judge']['n0'])
        dist=MM.metric_distance(Wa,Wb,b.st.lean['L'],n0);fine=MM.metric_distance(Wb,Wc,b.st.lean['L'],n0)
        valid=all(result['case_results'][qid][str(j)]['valid'] for qid,_ in definitions)
        p=cx.out/'data'/main_id/f'quadrature_c{j:03d}.npz';np.savez_compressed(p,W=Wa,adv_target=va,adv_successor=vb,rho=rho)
        rows.append(dict(case=j,rho=rho.tolist(),distance=dist,finer_distance=fine,all_rungs_valid=valid,
            passed=bool(valid and np.isfinite(rho).all() and rho.max()<=.116 and dist<=2.5e-4 and fine<=2.5e-4),
            raw=dict(path=str(p.relative_to(cx.out)),sha256=sha(p))))
    result['queries'][main_id]['Q2']=dict(passed=all(r['passed'] for r in rows),cases=rows,
        target=main[0],successors=[next_id[0],finer_id[0]],scope='four-case own-new-state Q2; successor stationarity retained separately, no full-population certificate')
    save(cx,result)

def timings(cx,result,b,qid,runner,path,n,rule,anchor,anchor_prepare):
    outputs={};output_n=n if path=='mesh' else 65
    for j in cx.cases:
        first=None;expected=None
        for rep in range(3):
            trip=[]
            for role,q,prep,Nout in [('A1',anchor,anchor_prepare,65),('B',runner,lambda j:prepare(cx,b,path,n,rule,j),output_n),('A2',anchor,anchor_prepare,65)]:
                C3.burn(cx.cfg['burn_seconds']);t0=time.perf_counter();u=jax.block_until_ready(prep(j));t1=time.perf_counter()
                o=MM.ready(q.run(u,cx.nu[j]));t2=time.perf_counter();F=np.asarray(o['fields']);ch=ah(o['W_all']);fh=ah(F)
                assert np.asarray(u).dtype==F.dtype==np.float64
                if role!='B' or qid=='M2_N65_mf_gl32':
                    prior=result['execution'].setdefault('anchor_hashes',{}).setdefault(str(j),[fh,ch])
                    assert prior==[fh,ch]
                row=dict(invocation_id=f'{qid}:c{j}:r{rep}:{role}',query=qid if role=='B' else 'M2_N65_mf_gl32',case=j,N=n if role=='B' else 65,
                    contract=path if role=='B' else 'mf',rep=rep,bracket=f'{qid}:c{j}:r{rep}',role=role,
                    given_native_seconds=t2-t1,input_prep_seconds=t1-t0,descriptor_seconds=t2-t0,
                    score=score_returned(cx,F,j,Nout),output_sha256=fh,coefficient_sha256=ch)
                trip.append(row)
                if role=='B':
                    if expected is not None:assert expected==(fh,ch)
                    expected=(fh,ch)
                    if first is None:first=(o,u,row)
            dr=trip[2]['descriptor_seconds']/trip[0]['descriptor_seconds'];nr=trip[2]['given_native_seconds']/trip[0]['given_native_seconds']
            for row in trip:row.update(anchor_drift_ratio=dr,timing_admitted=bool(1/1.1<=dr<=1.1),native_anchor_drift_ratio=nr,native_timing_admitted=bool(1/1.1<=nr<=1.1))
            result['timing_repetitions']+=trip
        o,u,timed=first;row,W=validate(cx,result,b,qid,runner,u,0.,j,o,output_n)
        row.update(invocation_id=timed['invocation_id']);outputs[(qid,j)]=W
        initial=MM.ready(runner.initialize(u));evolved=MM.ready(runner.evolve(initial,cx.nu[j]));split=MM.ready(runner.decode(evolved))
        rel=MM.relative(split,o['fields']);assert rel<=1e-12
        for rep in range(3):
            values=[('initialization',lambda:runner.initialize(u)),('evolve',lambda:runner.evolve(initial,cx.nu[j])),('decode',lambda:runner.decode(evolved))]
            for phase,fn in values:
                C3.burn(cx.cfg['burn_seconds']);t=time.perf_counter();answer=MM.ready(fn());seconds=time.perf_counter()-t
                result['phase_repetitions'].append(dict(query=qid,case=j,N=n,contract=path,rep=rep,phase=phase,seconds=seconds,split_fused_relative=rel,
                    output_leaf_sha256=[ah(x) for x in jax.tree_util.tree_leaves(answer)],scope='indicative separately compiled components; four cases only'))
        save(cx,result)
    return outputs

def run(cx):
    assert cx.cases==[0,5,30,43]
    assert cx.cfg['nodes']==[65,129] and cx.cfg['timing_repetitions']==cx.cfg['phase_repetitions']==3
    assert cx.cfg['M2']==dict(r=512,M=2052,rules=['gl32','gl40','gl48'],scheme='BDF2',dt=.05/3,refresh=4,contracts=['mf','mesh'])
    assert cx.cfg['W1024']==dict(r=1024,M=4097,rules=['gl56','gl64','gl80'],schemes=[['CN',.01,'fs',1],['BDF2',.00625,'qn',1]],interventions=['baseline','composite','lambda','moments'],held_projection_nodes=65)
    identity=__import__('json').loads((cx.out/'cohort_identity.json').read_text())
    result=dict(schema_version=1,cohort=dict(name='val64',seed=923801,ordered_cases=cx.cases,canonical_table_sha256=cx.tab['sha256'],
            raw_regeneration_table_sha256=identity['actual_sha256'],normalisation_audit=identity['normalisation_audit']),
        reference=dict(dimension=3,kind='ST',field='finest',store_version=RS.store_version(),approved_status_sha256=C3.sha_file(Path(__file__).parent/'vendor/e0/store/STATUS.json'),
            scoring_lattice_nodes=65,output_times=[0.,.05,.1,.15,.2,.25],time0_excluded_from_error_max=True),
        contracts=dict(mf='analytic initial values at fixed Gauss points; fixed63^3 device output',mesh='entire native mesh input and growing full native device output',
            held='supplied fixed65-projected coefficients; operator intervention, not end-to-end growing-mesh query',normalisation_scalar='canonical s_star supplied in common offline descriptor'),
        frozen_modes={},shared_setup=[],incremental_setup=[],queries={},case_results={},timing_repetitions=[],phase_repetitions=[],
        operator_differences={},initial_projection_differences={},projection_budgets={},execution=dict(complete=False,source_commit=cx.rep['source_commit'],job_id=cx.rep['job_id'],gpu=cx.rep['gpu'],config_sha256=cx.rep['config_sha256']))
    projection=ProjectionBudget(cx,result);save(cx,result)
    b=setup(cx,result,'M2',512,['gl32','gl40','gl48']);runners={};outputs={}
    for n in (65,129):
        for path in ('mf','mesh'):
            defs=[]
            for rule in ('gl32','gl40','gl48'):
                d=rule_data(cx,result,b,rule,n);qid=f'M2_N{n}_{path}_{rule}'
                q=MM.make(b.st,('qn','BDF2',.05/3,4),path,n,d,b.st.lean if path=='mf' else d)
                runners[qid]=q;defs.append((qid,q));result['queries'][qid]=dict(bank='M2',N=n,r=512,M=2052,m=len(b.st.rules[rule][1]),mode_hash=C3.sha(b.k),
                    contract=path,scheme='BDF2',dt=.05/3,refresh=4,trust=b.st.trust,cases=cx.cases,rule=rule,output_nodes=65 if path=='mf' else n,
                    input_shape=[len(b.st.rules[rule][1])] if path=='mf' else [(n-2)**3],output_shape=[6,63**3 if path=='mf' else (n-2)**3],boundary_values='homogeneous Dirichlet zeros implicit')
                t=time.perf_counter();u=prepare(cx,b,path,n,rule,0);MM.ready(q.run(u,cx.nu[0]))
                result['queries'][qid]['compile_and_warm_seconds']=time.perf_counter()-t
                result['queries'][qid]['effective_parameters']=dict(q_max=12,g_in=5e-4,slow_threshold=6,refresh=4,LMM_startup_steps=1)
                if rule=='gl32':
                    anchor=runners['M2_N65_mf_gl32'];ap=lambda j:prepare(cx,b,'mf',65,'gl32',j)
                    outputs.update(timings(cx,result,b,qid,q,path,n,rule,anchor,ap))
                else:
                    for j in cx.cases:
                        u=prepare(cx,b,path,n,rule,j);o=MM.ready(q.run(u,cx.nu[j]));row,W=validate(cx,result,b,qid,q,u,0.,j,o,65 if path=='mf' else n);outputs[(qid,j)]=W
                result['queries'][qid]['Q1']=qualify(result,qid)
            quadrature(cx,result,b,defs[0][0],defs,outputs)
            for j in cx.cases:
                u=prepare(cx,b,path,n,'gl32',j);c=MM.ready(defs[0][1].initialize(u))[0][0]
                c65=MM.ready(runners['M2_N65_mf_gl32'].initialize(prepare(cx,b,'mf',65,'gl32',j)))[0][0]
                result['initial_projection_differences'][f'{n}_{path}_c{j}']=dict(common65_distance=float(jnp.linalg.norm((c-c65)@b.st.lean['L'])/cx.ref(j)[1]['judge']['n0']),baseline='N65 fixed-Gauss projection')
            save(cx,result)
    result['cost_summaries']={}
    for qid,query in result['queries'].items():
        if query['rule']!='gl32':continue
        costs={}
        for contract,metric,admission in [('descriptor','descriptor_seconds','timing_admitted'),('given_native','given_native_seconds','native_timing_admitted')]:
            medians={}
            for j in cx.cases:
                vals=[r[metric] for r in result['timing_repetitions'] if r['query']==qid and r['role']=='B' and r['case']==j and r[admission]]
                if vals:medians[str(j)]=float(np.median(vals))
            costs[contract]=dict(complete_case_coverage=len(medians)==4,per_case_medians=medians,
                median_seconds=float(np.median(list(medians.values()))) if len(medians)==4 else None,
                aggregation='median of per-case medians; missing admission on any case leaves cost unresolved')
        result['cost_summaries'][qid]=costs
    del b,runners,outputs,defs,q,d,anchor,o,u;gc.collect()
    b=setup(cx,result,'W1024',1024,['gl56','gl64','gl80']);outputs={};runners={}
    for scheme,dt,solver,policy in [('CN',.01,'fs',1),('BDF2',.00625,'qn',1)]:
        for intervention,cases in [('baseline',cx.cases),('composite',cx.cases),('lambda',[5]),('moments',[5])]:
            defs=[]
            for rule in ('gl56','gl64','gl80'):
                d=rule_data(cx,result,b,rule,65,None if intervention=='baseline' else intervention)
                qid=f'W1024_{scheme}_{intervention}_{rule}';q=MM.make(b.st,(solver,scheme,dt,policy),'held',65,d,b.st.lean)
                runners[qid]=q;defs.append((qid,q));result['queries'][qid]=dict(bank='W1024',N=65 if intervention=='baseline' else 129,r=1024,M=4097,m=len(b.st.rules[rule][1]),
                    mode_hash=C3.sha(b.k),contract='held',scheme=scheme,dt=dt,solver=solver,work_policy=policy,trust=b.st.trust,cases=cases,rule=rule,
                    intervention=intervention,residual_weights_follow_actual_lambda=True,output_nodes=65)
                for j in cases:
                    c0,s0=MM.ready(C3.TB.project(cx.U[j],b.st.lean));scale=float(jnp.linalg.norm(s0))
                    t=time.perf_counter();o=MM.ready(q.run(c0,cx.nu[j],scale));elapsed=time.perf_counter()-t
                    row,W=validate(cx,result,b,qid,q,c0,scale,j,o,65);outputs[(qid,j)]=W
                    row['untimed_diagnostic_call_seconds_with_possible_first_compile']=elapsed
                    if rule=='gl56':row['projection_budget']=projection.measure(dict(kind='bank',st=b.st),j,W,row['score']['e_t'])
                    weights={}
                    for step in (0,3):
                        _,_,Srow=MM.weighted_pair(c0,c0,c0,cx.nu[j],d,scheme,dt,step);weights[f'S_step{step}']=np.asarray(Srow)
                    wp=cx.out/'data'/qid/f'c{j:03d}'/'residual_weights.npz';np.savez_compressed(wp,lambda_used=np.asarray(d['lam']),**weights)
                    row['residual_weights']=dict(path=str(wp.relative_to(cx.out)),sha256=sha(wp),meaning='lambda induces both physical diffusion term and inherited residual/Jacobian weights')
                result['queries'][qid]['Q1']=qualify(result,qid)
            quadrature(cx,result,b,defs[0][0],defs,outputs)
    result['execution'].update(complete=True,elapsed_seconds=time.perf_counter()-cx.t0,whole_job_memory=memory(),
        raw_file_manifest=[dict(path=str(p.relative_to(cx.out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((cx.out/'data').rglob('*.npz'))],
        scope='four-case fixed-mode65/129 diagnostics; no exponent or full64 new-mesh qualification')
    save(cx,result)
