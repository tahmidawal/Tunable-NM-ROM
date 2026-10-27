"""NumPy reconstruction of the one A3 replay, with schedule-sized certificates."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from analyse import sha,safe_ratio


def certificate(z,level,n):
    C=z['C'];g=z['gamma'];rs=z['reason'];lam=z['lam']
    assert C.shape==(n+1,512) and g.shape==rs.shape==lam.shape==(n,)
    known=np.isin(rs,[0,1,2,3,4,5]).all();finite=np.isfinite(C).all() and np.isfinite(lam).all() and (lam<1e14).all()
    gt={'prod':.001,'tight':1e-7,'tighter':1e-8}[level]
    stall=(rs==3)&np.isfinite(g)&(g<=100*gt)
    return dict(valid=bool(finite and known and not np.isin(rs,[3,5]).any()),
        anchor_ok=bool(finite and known and not (rs==5).any() and ((np.isfinite(g)&(g<=gt))|stall).all()
                       and ((rs!=3)|stall).all()),
        nonstat=int((~(np.isfinite(g)&(g<=.001))).sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--basis',required=True);p.add_argument('--out',required=True)
    p.add_argument('--references',default='/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
    a=p.parse_args();rp=Path(a.result);d=rp.parent;j=json.loads(rp.read_text());assert j['complete']
    assert sha(a.basis)==j['artifacts']['files']['Phi.npz']
    Phi=np.load(a.basis)['Phi'][:,:512];gram=Phi.T@Phi;assert np.max(np.abs(gram-np.eye(512)))<=1e-10
    sys.path.insert(0,a.references);import refstore as RS
    assert RS.store_version()==j['reference']['store_version']
    assert sha(Path(a.references)/'store/STATUS.json')==j['reference']['status_sha256']
    ids=j['config']['cases'];assert ids==[0,5,30,43] and list(j['cases'])==list(map(str,ids));n=j['steps']
    fact=dict(result_path=str(rp),result_sha256=sha(rp),source_commit=j['source_commit'],job_id=j['job_id'],
        selection=j['selection'],config=j['config'],steps=n,cases={},Q1={},checks=[],whole_residual={},
        scope='four preselected diagnostic cases only; not full-population qualification')
    def distance(x,y,n0):
        diff=x[1:]-y[1:]
        return float(np.sqrt(np.maximum(np.sum((diff@gram)*diff,axis=1),0)).max()/n0)
    for i in ids:
        ref,rm=RS.load(3,'val64',i,'ST',field='finest');n0=rm['judge']['n0'];fact['cases'][str(i)]={};saved={}
        for arm in ('k16','none'):
            rr=j['cases'][str(i)][arm];z=np.load(d/'arrays'/f'{arm}_c{i:03}_prod.npz')
            t=np.load(d/'arrays'/f'{arm}_c{i:03}_tight.npz');saved[arm]=z['W']
            assert int(z['case'])==i and z['W'].shape==(6,512) and np.array_equal(z['times'],np.arange(6)*.05)
            c=certificate(z,'prod',n);ct=certificate(t,'tight',n);dd=distance(z['W'],t['W'],n0)
            F=z['W']@Phi.T
            fact['checks'].append(np.linalg.norm(F-z['fields'])/np.linalg.norm(F)<=1e-12)
            err=np.linalg.norm(F[1:]-ref[1:],axis=1)/n0;e=float(err.max())
            row=dict(error=e,certificate=c,native_tight_anchor=ct['anchor_ok'],tight_distance=dd,
                nonlinear_iterations=int(z['it'].sum()),query_seconds=rr['query_seconds'])
            fact['checks'] += [c['valid']==rr['cert']['valid'],c['nonstat']==rr['cert']['nonstat'],
                ct['anchor_ok']==rr['tight']['anchor_ok'],abs(dd-rr['tight_distance'])<=1e-10,abs(e-rr['e0']['e'])<=1e-10]
            if i in j['config']['tighter_cases']:
                tt=np.load(d/'arrays'/f'{arm}_c{i:03}_tighter.npz');ctt=certificate(tt,'tighter',n);dtt=distance(t['W'],tt['W'],n0)
                row.update(native_tighter_anchor=ctt['anchor_ok'],tight_tighter_distance=dtt)
                fact['checks'] += [ctt['anchor_ok']==rr['tighter']['anchor_ok'],abs(dtt-rr['tight_tighter_distance'])<=1e-10]
            fact['cases'][str(i)][arm]=row
            if arm=='k16':
                m=rr['whole_residual'];gf=np.asarray(m['g_full']);ge=np.asarray(m['g_sampled'])
                assert gf.shape==ge.shape==(n,512)
                den=np.asarray(m['jacobian_norm_full'])*np.asarray(m['residual_norm_full'])
                dg=safe_ratio(np.linalg.norm(ge-gf,axis=1),den);gg=safe_ratio(np.linalg.norm(gf,axis=1),den)
                finite=bool(np.isfinite(gf).all() and np.isfinite(ge).all() and np.isfinite(den).all()
                            and np.isfinite(dg).all() and np.isfinite(gg).all() and all(m['finite']))
                fact['whole_residual'][str(i)]=dict(delta_g=dg.tolist(),full_gamma=gg.tolist(),finite=finite,
                    gradient_step_mask=(np.isfinite(dg)&(dg<=.001)).tolist(),
                    full_stationarity_bad_step_mask=((~np.isfinite(gg))|(gg>.001)).tolist())
                fact['checks'] += [bool(np.allclose(dg,m['delta_g'],rtol=1e-10,atol=1e-13)),
                    bool(np.allclose(gg,m['full_gamma'],rtol=1e-10,atol=1e-13)),
                    bool(np.allclose(den,m['denominator'],rtol=1e-12,atol=0))]
        dd=distance(saved['k16'],saved['none'],n0);fact['cases'][str(i)]['k16_to_none']=dd
        fact['checks'].append(abs(dd-j['cases'][str(i)]['k16_to_none'])<=1e-10)
    for arm in ('k16','none'):
        rows=[fact['cases'][str(i)][arm] for i in ids];bad=sum(r['certificate']['nonstat'] for r in rows)
        qp=bool(all(r['certificate']['valid'] and r['native_tight_anchor'] and r['tight_distance']<=.00025 for r in rows)
            and bad<=.01*n*len(ids) and all(fact['cases'][str(i)][arm]['native_tighter_anchor']
                and fact['cases'][str(i)][arm]['tight_tighter_distance']<=.000025 for i in ids[:2]))
        fact['Q1'][arm]=dict(pass_=qp,nonstationary_steps=bad,steps=n*len(ids),
            median_error=float(np.median([r['error'] for r in rows])),worst_error=max(r['error'] for r in rows))
        fact['checks'].append(qp==j['Q1'][arm]['q1'])
    field={str(i):bool(np.isfinite(fact['cases'][str(i)]['k16_to_none']) and fact['cases'][str(i)]['k16_to_none']<=.00025) for i in ids}
    whole=fact['whole_residual'];finite=all(m['finite'] for m in whole.values())
    gp=bool(finite and all(all(m['gradient_step_mask']) for m in whole.values()))
    bad=sum(sum(m['full_stationarity_bad_step_mask']) for m in whole.values());sp=bool(finite and bad<=.01*n*len(ids))
    common=bool(all(q['pass_'] for q in fact['Q1'].values()) and all(field.values()))
    fact['admission']=dict(common_layer_pass=common,shared_field_case_mask=field,extra_gradient_pass=gp,
        extra_full_stationarity_bad_steps=bad,extra_full_stationarity_pass=sp,both_layers_pass=bool(common and gp and sp))
    for k,v in fact['admission'].items():fact['checks'].append(v==j['admission'][k])
    # Each timed output is reconstructible from its exact saved payload, including a differing rejected repeat.
    import hashlib
    def ah(x):
        x=np.ascontiguousarray(x);return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()
    for row in j['timing']:
        for v in row['invocations']:
            z=np.load(d/v['payload_path'])
            fact['checks'] += [ah(z[k])==v[k+'_sha256'] for k in ('C','W','fields')]
        fact['checks'].append(row['timing_admitted']==bool(row['drift_accepted'] and all(v['identity'] and v['e0']['eligible'] for v in row['invocations'])))
    fact.update(endpoint_reproduction_pass=j['endpoint_reproduction_pass'],timing=j['timing'],
        full64_adapter_release=j['admission']['full64_adapter_release'],offline_checks_pass=bool(all(fact['checks'])))
    assert fact['offline_checks_pass']
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True);(out/'SUMMARY.json').write_text(json.dumps(fact,indent=2)+'\n')
    print(json.dumps({k:v for k,v in fact.items() if k in ('Q1','admission','offline_checks_pass','full64_adapter_release')},indent=2))


if __name__=='__main__':main()
