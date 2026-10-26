"""NumPy-only validation of complete fallback or frozen k32 replay evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from analyse import certificate,sha,safe_ratio


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',required=True);ap.add_argument('--basis',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--references',default='/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
    a=ap.parse_args();p=Path(a.result);d=p.parent;out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    j=json.loads(p.read_text());assert j['complete']
    cfg=j['config'];ids=cfg['cases'];arms=cfg['arms'];assert list(j['cases'])==[str(i) for i in ids]
    assert sha(a.basis)==j['artifacts']['files']['Phi.npz']
    Phi=np.load(a.basis)['Phi'][:,:512];gram=Phi.T@Phi
    assert np.max(np.abs(gram-np.eye(512)))<=1e-10
    sys.path.insert(0,a.references);import refstore as RS
    assert RS.store_version()==j['reference']['store_version']
    assert sha(Path(a.references)/'store/STATUS.json')==j['reference']['status_sha256']
    fact=dict(result_path=str(p.resolve()),result_sha256=sha(p),config=cfg,source_commit=j['source_commit'],
              gpu=j['gpu'],job_id=j['job_id'],reference=j['reference'],cases={},arms={},checks=[],
              complete_population=ids==list(range(64)),qualification_scope=cfg['scope'],elapsed_seconds=j['elapsed_seconds'])
    for i in ids:
        ref,rm=RS.load(3,'val64',i,'ST',field='finest');st=RS.reference_status(3,'val64',i,'ST');assert st['eligible']
        n0=rm['judge']['n0'];fact['cases'][str(i)]={}
        for arm in arms:
            rr=j['cases'][str(i)][arm];z=np.load(d/'arrays'/f'{arm}_c{i:03}_prod.npz');zt=np.load(d/'arrays'/f'{arm}_c{i:03}_tight.npz')
            assert int(z['case'])==i and z['W'].shape==(6,512) and np.array_equal(z['times'],np.arange(6)*.05)
            cert=certificate(z,'prod');ct=certificate(zt,'tight')
            diff=z['W'][1:]-zt['W'][1:]
            dist=float(np.sqrt(np.maximum(np.sum((diff@gram)*diff,axis=1),0)).max()/n0)
            fact['checks'] += [cert['valid']==rr['cert']['valid'],cert['nonstat']==rr['cert']['nonstat'],
                               ct['anchor_ok']==rr['tight']['anchor_ok'],abs(dist-rr['tight_distance'])<=1e-10]
            fields=z['W']@Phi.T
            et=np.linalg.norm(fields[1:]-ref[1:],axis=1)/n0
            e=float(et.max());fact['checks'].append(abs(e-rr['e0']['e'])<=1e-10)
            row=dict(error=e,eps_ref=st['eps_ref'],eligible=True,query_seconds=rr['query_seconds'],
                     certificate=cert,tight_anchor_ok=ct['anchor_ok'],tight_distance=dist,
                     nonlinear_iterations=int(z['it'][:15].sum()),steps=15,
                     logical_jacobian_evaluations=15+int(z['it'][:15].sum()),
                     logical_full_residual_evaluations=60+int(z['it'][:15].sum()),
                     history_nonlinear_evaluations=15)
            if i in cfg['tighter_cases']:
                zz=np.load(d/'arrays'/f'{arm}_c{i:03}_tighter.npz');cc=certificate(zz,'tighter')
                diff=zt['W'][1:]-zz['W'][1:]
                dd=float(np.sqrt(np.maximum(np.sum((diff@gram)*diff,axis=1),0)).max()/n0)
                row.update(tighter_anchor_ok=cc['anchor_ok'],tight_tighter_distance=dd)
                fact['checks'] += [cc['anchor_ok']==rr['tighter']['anchor_ok'],abs(dd-rr['tight_tighter_distance'])<=1e-10]
            fact['cases'][str(i)][arm]=row
        print('checked_case',i,flush=True)
    for arm in arms:
        rows=[fact['cases'][str(i)][arm] for i in ids];err=np.array([r['error'] for r in rows]);times=np.array([r['query_seconds'] for r in rows])
        nonstat=sum(r['certificate']['nonstat'] for r in rows)
        qs=bool(all(r['certificate']['valid'] and r['tight_anchor_ok'] and r['tight_distance']<=.00025 for r in rows)
                and nonstat<=.01*15*len(ids)
                and all(fact['cases'][str(i)][arm]['tighter_anchor_ok'] and fact['cases'][str(i)][arm]['tight_tighter_distance']<=.000025 for i in cfg['tighter_cases']))
        fact['checks'].append(qs==j['Q1'][arm]['q1'])
        fact['arms'][arm]=dict(Q1=qs,nonstationary_steps=nonstat,steps=15*len(ids),median_error=float(np.median(err)),
                              worst_error=float(err.max()),errors_above_one_percent=int((err>.01).sum()),
                              median_query_seconds=float(np.median(times)),query_seconds=times.tolist(),
                              median_nonlinear_iterations=float(np.median([r['nonlinear_iterations'] for r in rows])),
                              maximum_tight_distance=max(r['tight_distance'] for r in rows))
    fact['timing']=j['timing'];fact['HR']=j.get('HR')
    fact['checks'].append(all(x['identity'] and x['e0']['eligible'] for t in j['timing'] for x in t['invocations']))
    fact['offline_checks_pass']=bool(all(fact['checks']));assert fact['offline_checks_pass']
    (out/'SUMMARY.json').write_text(json.dumps(fact,indent=2)+'\n')
    print(json.dumps({k:v for k,v in fact.items() if k in ('arms','complete_population','offline_checks_pass','HR')},indent=2))


if __name__=='__main__':main()
