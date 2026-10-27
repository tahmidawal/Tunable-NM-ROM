"""Fair frozen five-schedule POD development on the four predeclared validation cases."""
from pathlib import Path
import hashlib
import time
import numpy as np
import jax
import jax.numpy as jnp
import c3common as C3
import pod_policy
import pod_identity as ID

HERE=Path(__file__).resolve().parent
def ah(x):return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()
def score(cx,F,j):
    R,meta,status=cx.ref(j);n0=float(meta['judge']['n0'])
    if not np.isfinite(np.asarray(F)).all():
        return dict(e=None,e_t=[None]*6,epsilon=status['eps_ref'],eligible=status['eligible'],normalisation=n0,finite=False)
    et=np.linalg.norm(np.asarray(F)-R,axis=1)/n0
    return dict(e=float(et[1:].max()),e_t=et.tolist(),epsilon=status['eps_ref'],eligible=status['eligible'],normalisation=n0,finite=True)

def run(cx,POD):
    t=time.perf_counter()
    subjects=(pod_policy.build(POD,HERE.parent/'b3d-pod/inputs/Phi.npz',True) if cx.cfg.get('production_gtol') else POD.build(HERE.parent/'b3d-pod/inputs/Phi.npz'))
    cx.rep['setup']['POD']=dict(seconds=time.perf_counter()-t,metadata=subjects[0]['metadata'])
    assert cx.cases==[0,5,30,43] and len(subjects)==(7 if cx.cfg.get('production_gtol') else 5)
    cx.rep.update(scope='five frozen POD schedules on four exposed validation cases; developmental selection only',selection={})
    baseline=next(s for s in subjects if s['steps']==15 and s.get('production_gtol',1e-3)==1e-3)
    gram=np.asarray(jax.block_until_ready(subjects[0]['data']['Phi'].T@subjects[0]['data']['Phi']))
    previous=None
    if cx.cfg.get('reuse_qualified'):
        assert cx.cfg['identity_field_tolerance']==ID.FIELD_TOL and cx.cfg['identity_error_tolerance']==ID.ERROR_TOL
        pin=cx.cfg['reuse_qualified']['result'];path=HERE/pin['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pin['sha256']
        previous=__import__('json').loads(path.read_text());assert previous['complete'] and previous['cohort_sha256']==cx.tab['sha256']
    # Warm all query paths and bind each supplied production output to its exact subject/case call.
    for s in subjects:
        for j in cx.cases:
            o=jax.block_until_ready(s['query'](cx.U[j],cx.nu[j],s['data'],s['sch']))
            expected=jax.block_until_ready(s['initialization_query'](cx.U[j],s['data']))[0]
            assert np.allclose(np.asarray(o['c0']),np.asarray(expected),rtol=1e-12,atol=1e-12)
            reuse=previous is not None
            if reuse:
                oldname=s['name'].removesuffix('_g1e-4');key=f"{oldname}|case{j}";prior=previous['runs'][key]
                assert (prior['case'],prior['scheme'],prior['dt'],prior['steps'])==(j,s['scheme'],s['dt'],s['steps'])
                assert prior['input_sha256']==ah(cx.U[j])
                assert previous['setup']['POD']['metadata']['basis_sha256']==s['metadata']['basis_sha256']
                assert previous['setup']['POD']['metadata']['trust_radius']==s['metadata']['trust_radius']
                pin=cx.cfg['reuse_qualified']['arrays'][key];path=HERE/pin['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==pin['sha256']
                z=np.load(path,allow_pickle=False)
                assert z['C'].shape==np.asarray(o['C']).shape==(s['steps']+1,s['r'])
                n0=float(prior['certificate']['n0'])
                c0_identity=ID.field_distance(np.asarray(o['C'])[:1],z['C'][:1],gram,n0)
                if c0_identity>ID.FIELD_TOL:
                    cx.rep.setdefault('identity_failures',{})[key]=dict(C0_distance=c0_identity,input_sha256=ah(cx.U[j]),source=pin)
                    np.savez_compressed(cx.out/f"identity_failed_C0_{s['name']}_c{j:03d}.npz",C=np.asarray(o['C']),W=np.asarray(o['W']),
                        fields=np.asarray(o['fields']),old_C=np.asarray(z['C']),old_W=np.asarray(z['W']),Gram=gram)
                    cx.save()
                assert c0_identity<=ID.FIELD_TOL
                if s.get('production_gtol',1e-3)==1e-3:
                    matches=ah(o['C'])==prior['coefficient_sha256'] and ah(o['fields'])==prior['output_sha256']
                    old_fields=jax.block_until_ready(s['decode_query'](jnp.asarray(z['W']),s['data']))
                    numerical_identity=ID.original_policy(o['C'],z['C'],o['fields'],old_fields,gram,n0,score(cx,o['fields'],j)['e'],prior['score']['e'])
                    if not matches:
                        probe=dict(C_current_sha256=ah(o['C']),C_prior_sha256=prior['coefficient_sha256'],
                            fields_current_sha256=ah(o['fields']),fields_prior_sha256=prior['output_sha256'],old_decoded_fields_sha256=ah(old_fields),
                            C_max_absolute=float(np.max(np.abs(np.asarray(o['C'])-z['C']))),
                            C_relative=float(np.linalg.norm(np.asarray(o['C'])-z['C'])/(np.linalg.norm(z['C'])+1e-300)),
                            C0_bitwise_equal=bool(np.array_equal(np.asarray(o['C'])[0],z['C'][0])),
                            W_max_absolute=float(np.max(np.abs(np.asarray(o['W'])-z['W']))),
                            decoded_field_distance=float(np.linalg.norm(np.asarray(o['fields'])-np.asarray(old_fields))/np.linalg.norm(np.asarray(cx.U[j]))),
                            current_score=score(cx,o['fields'],j),prior_score=prior['score'],old_decoded_score=score(cx,old_fields,j),
                            fresh_certificate=s['certify'](o,cx.nu[j]),input_sha256=ah(cx.U[j]),source=pin,
                            numerical_identity=numerical_identity,
                            note='Failed byte-equality expectation retracted after independent actual-array audit; bounded numerical identity checked without changing states or science gates. Origin of variation unknown.')
                        cx.rep.setdefault('identity_probes',{})[key]=probe
                        np.savez_compressed(cx.out/f"identity_probe_{s['name']}_c{j:03d}.npz",C=np.asarray(o['C']),W=np.asarray(o['W']),fields=np.asarray(o['fields']),
                            old_C=np.asarray(z['C']),old_W=np.asarray(z['W']),old_decoded_fields=np.asarray(old_fields),Gram=gram)
                        cx.save()
                    assert prior['common_rollout_check'] and numerical_identity['passed']
                detail=dict(prior['certificate'])
                assert ah(z['C'])==prior['coefficient_sha256']
                assert detail['tight_certificate']['anchor_ok'] and detail['tighter_certificate']['anchor_ok'] and detail['tight_tighter_distance']<=2.5e-5
                detail['production_certificate']=s['certify'](o,cx.nu[j]);detail['reused_anchor_source']=pin
                for k in ('Wtight','Ctight','gamma_tight','Wtighter','Ctighter','gamma_tighter'):detail[k]=np.asarray(z[k])
                diff=np.asarray(o['W'])[1:]-detail['Wtight'][1:]
                detail['prod_to_tight_distance']=float(np.sqrt(np.maximum(np.sum((diff@gram)*diff,axis=1),0)).max()/detail['n0'])
                assert abs(detail['n0']/float(np.linalg.norm(np.asarray(cx.U[j])))-1)<=1e-12
                detail['production_stopping_gtol']=float(s['sch']['gtol'])
                detail['source_C0_identity_distance']=c0_identity
            else:detail=s['validate'](cx.U[j],cx.nu[j],prod=o,need_tighter=True)
            saved={k:np.asarray(v) for k,v in detail.items() if k.startswith(('Wtight','Ctight','gamma_tight'))}
            saved.update(W=np.asarray(o['W']),C=np.asarray(o['C']),**{'stats__'+k:np.asarray(v) for k,v in o['stats'].items()})
            np.savez_compressed(cx.out/f"{s['name']}_c{j:03d}.npz",**saved)
            small={k:v for k,v in detail.items() if k not in saved}
            # The exact typed call signature accompanies the coefficient-bearing production output.
            pc=detail['production_certificate'];it=int(np.asarray(o['stats']['it'])[:s['steps']].sum())
            cx.rep['runs'][f"{s['name']}|case{j}"]=dict(subject=s['name'],case=j,scheme=s['scheme'],dt=s['dt'],steps=s['steps'],
                production_stopping_gtol=float(s['sch']['gtol']),input_sha256=ah(cx.U[j]),coefficient_sha256=ah(o['C']),output_sha256=ah(o['fields']),score=score(cx,o['fields'],j),
                certificate=small,valid=pc['valid'],nonstationary=pc['nonstat'],
                common_rollout_check=bool(pc['valid'] and detail['tight_certificate']['anchor_ok'] and detail['tighter_certificate']['anchor_ok']
                                          and detail['prod_to_tight_distance']<=2.5e-4 and detail['tight_tighter_distance']<=2.5e-5),
                method_specific_extra_masks=dict(full_residual=True,hyper_reduced=False,sampled_full_gradient='not applicable'),
                work=dict(nonlinear_iterations=it,residual_jacobian_pairs=s['steps']+it,
                          residual_evaluations=4*s['steps']+it,history_F_evaluations=s['steps'],count_kind='logical calls; not GPU kernels'))
            cx.save()
    rng=np.random.default_rng(20261026)
    for j in cx.cases:
        for si in rng.permutation(len(subjects)):
            s=subjects[si]
            for rep in range(cx.cfg['timing_reps']):
                trip=[]
                for role,sub in [('A1',baseline),('B',s),('A2',baseline)]:
                    C3.burn(cx.cfg['burn_seconds']);t0=time.perf_counter()
                    u=jax.block_until_ready(jnp.asarray(C3.C.initial_interior(65,cx.tab,j)))
                    t1=time.perf_counter();o=jax.block_until_ready(sub['query'](u,cx.nu[j],sub['data'],sub['sch']));t2=time.perf_counter()
                    old=cx.rep['runs'][f"{sub['name']}|case{j}"]
                    assert ah(o['C'])==old['coefficient_sha256'] and ah(o['fields'])==old['output_sha256']
                    trip.append(dict(subject=sub['name'],case=j,rep=rep,role=role,input_prep_seconds=t1-t0,
                        given_native_seconds=t2-t1,descriptor_seconds=t2-t0,score=score(cx,o['fields'],j),
                        coefficient_sha256=ah(o['C']),output_sha256=ah(o['fields'])))
                drift=trip[2]['descriptor_seconds']/trip[0]['descriptor_seconds']
                native_drift=trip[2]['given_native_seconds']/trip[0]['given_native_seconds']
                for row in trip:row.update(anchor_drift_ratio=drift,timing_admitted=bool(1/1.1<=drift<=1.1),
                    native_anchor_drift_ratio=native_drift,native_timing_admitted=bool(1/1.1<=native_drift<=1.1))
                cx.rep['timing'][f"{s['name']}|case{j}|rep{rep}"]=trip;cx.save()
    candidates=[]
    for s in subjects:
        rows=[cx.rep['runs'][f"{s['name']}|case{j}"] for j in cx.cases]
        alltimes=[r for trip in cx.rep['timing'].values() for r in trip if r['subject']==s['name'] and r['role']=='B']
        admitted_by_case={j:[r['descriptor_seconds'] for r in alltimes if r['case']==j and r['timing_admitted']] for j in cx.cases}
        coverage=all(admitted_by_case[j] for j in cx.cases)
        qualified=all(r['common_rollout_check'] for r in rows) and sum(r['nonstationary'] for r in rows)<=.01*sum(r['steps'] for r in rows)
        rec=dict(subject=s['name'],qualified_subset=qualified,
                 median_descriptor_seconds=float(np.median([np.median(admitted_by_case[j]) for j in cx.cases])) if coverage else None,
                 cost_selection_usable=coverage,cost_case_coverage=[j for j in cx.cases if admitted_by_case[j]],
                 worst_error=max(r['score']['e'] for r in rows),median_error=float(np.median([r['score']['e'] for r in rows])),
                 timing_all_admitted=all(r['timing_admitted'] for r in alltimes),eligible_cases=[r['case'] for r in rows if r['score']['eligible']])
        cx.rep['selection'][s['name']]=rec
        if qualified and len(rec['eligible_cases'])==4:candidates.append(rec)
    picks=[]
    if candidates:
        for key in ('median_descriptor_seconds','worst_error'):
            eligible=[r for r in candidates if key=='worst_error' or r['cost_selection_usable']]
            if not eligible:continue
            p=min(eligible,key=lambda r:(r[key],r['median_error'],r['subject']))['subject']
            if p not in picks:picks.append(p)
    cx.rep['frozen_picks']=picks;cx.rep['full_population_qualification']='not established by four-case tuning';cx.save()
