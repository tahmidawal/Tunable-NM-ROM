"""Fair frozen five-schedule POD development on the four predeclared validation cases."""
from pathlib import Path
import hashlib
import time
import numpy as np
import jax
import jax.numpy as jnp
import c3common as C3

HERE=Path(__file__).resolve().parent
def ah(x):return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()
def score(cx,F,j):
    R,meta,status=cx.ref(j);n0=float(meta['judge']['n0'])
    et=np.linalg.norm(np.asarray(F)-R,axis=1)/n0
    return dict(e=float(et[1:].max()),e_t=et.tolist(),epsilon=status['eps_ref'],eligible=status['eligible'],normalisation=n0)

def run(cx,POD):
    t=time.perf_counter();subjects=POD.build(HERE.parent/'b3d-pod/inputs/Phi.npz')
    cx.rep['setup']['POD']=dict(seconds=time.perf_counter()-t,metadata=subjects[0]['metadata'])
    assert cx.cases==[0,5,30,43] and len(subjects)==5
    cx.rep.update(scope='five frozen POD schedules on four exposed validation cases; developmental selection only',selection={})
    baseline=next(s for s in subjects if s['steps']==15)
    # Warm all query paths and bind each supplied production output to its exact subject/case call.
    for s in subjects:
        for j in cx.cases:
            o=jax.block_until_ready(s['query'](cx.U[j],cx.nu[j],s['data'],s['sch']))
            expected=jax.block_until_ready(s['initialization_query'](cx.U[j],s['data']))[0]
            assert np.allclose(np.asarray(o['c0']),np.asarray(expected),rtol=1e-12,atol=1e-12)
            detail=s['validate'](cx.U[j],cx.nu[j],prod=o,need_tighter=True)
            saved={k:np.asarray(v) for k,v in detail.items() if k.startswith(('Wtight','Ctight','gamma_tight'))}
            saved.update(W=np.asarray(o['W']),C=np.asarray(o['C']),**{'stats__'+k:np.asarray(v) for k,v in o['stats'].items()})
            np.savez_compressed(cx.out/f"{s['name']}_c{j:03d}.npz",**saved)
            small={k:v for k,v in detail.items() if k not in saved}
            # The exact typed call signature accompanies the coefficient-bearing production output.
            pc=detail['production_certificate'];it=int(np.asarray(o['stats']['it'])[:s['steps']].sum())
            cx.rep['runs'][f"{s['name']}|case{j}"]=dict(subject=s['name'],case=j,scheme=s['scheme'],dt=s['dt'],steps=s['steps'],
                input_sha256=ah(cx.U[j]),coefficient_sha256=ah(o['C']),output_sha256=ah(o['fields']),score=score(cx,o['fields'],j),
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
                for row in trip:row.update(anchor_drift_ratio=drift,timing_admitted=bool(1/1.1<=drift<=1.1))
                cx.rep['timing'][f"{s['name']}|case{j}|rep{rep}"]=trip;cx.save()
    candidates=[]
    for s in subjects:
        rows=[cx.rep['runs'][f"{s['name']}|case{j}"] for j in cx.cases]
        alltimes=[r for trip in cx.rep['timing'].values() for r in trip if r['subject']==s['name'] and r['role']=='B']
        admitted=[r['descriptor_seconds'] for r in alltimes if r['timing_admitted']]
        qualified=all(r['common_rollout_check'] for r in rows) and sum(r['nonstationary'] for r in rows)<=.01*sum(r['steps'] for r in rows)
        rec=dict(subject=s['name'],qualified_subset=qualified,median_descriptor_seconds=float(np.median(admitted)) if admitted else None,
                 worst_error=max(r['score']['e'] for r in rows),median_error=float(np.median([r['score']['e'] for r in rows])),
                 timing_all_admitted=all(r['timing_admitted'] for r in alltimes),eligible_cases=[r['case'] for r in rows if r['score']['eligible']])
        cx.rep['selection'][s['name']]=rec
        if qualified and admitted and len(rec['eligible_cases'])==4:candidates.append(rec)
    picks=[]
    if candidates:
        for key in ('median_descriptor_seconds','worst_error'):
            p=min(candidates,key=lambda r:(r[key],r['median_error'],r['subject']))['subject']
            if p not in picks:picks.append(p)
    cx.rep['frozen_picks']=picks;cx.rep['full_population_qualification']='not established by four-case tuning';cx.save()
