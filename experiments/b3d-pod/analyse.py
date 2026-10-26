"""Independent NumPy checks of saved POD replay arrays and a source-generated diagnostic report."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<22),b''):h.update(b)
    return h.hexdigest()


def safe_ratio(n,d):
    n,d=np.asarray(n,float),np.asarray(d,float)
    with np.errstate(divide='ignore',invalid='ignore'):
        ans=np.where(d==0,np.where(n==0,0.,np.inf),n/d)
    return np.where(np.isfinite(n)&np.isfinite(d)&(d>=0),ans,np.inf)


def certificate(z,level):
    C,g,rs,lam=z['C'],z['gamma'],z['reason'][:15],z['lam'][:15]
    assert C.shape==(16,512) and g.shape==(15,)
    known=np.isin(rs,[0,1,2,3,4,5]).all()
    finite=np.isfinite(C).all() and np.isfinite(lam).all() and np.all(lam<1e14)
    valid=bool(finite and known and not np.isin(rs,[3,5]).any())
    gt={'prod':1e-3,'tight':1e-7,'tighter':1e-8}[level]
    stall=(rs==3)&np.isfinite(g)&(g<=100*gt)
    anchor=bool(finite and known and not (rs==5).any() and np.all((np.isfinite(g)&(g<=gt))|stall)
                and np.all((rs!=3)|stall))
    return dict(valid=valid,anchor_ok=anchor,nonstat=int(np.sum(~(np.isfinite(g)&(g<=1e-3)))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',required=True);ap.add_argument('--basis',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args()
    p=Path(a.result);d=p.parent;out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    j=json.loads(p.read_text());assert j['complete']
    assert sha(a.basis)==j['artifacts']['files']['Phi.npz']
    Phi=np.load(a.basis)['Phi'][:,:512]
    ids=j['config']['cases'];expected=[str(i) for i in ids]
    assert list(j['cases'])==expected
    E0=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/store')
    assert hashlib.sha256((E0/'MANIFEST.sha256').read_bytes()).hexdigest()==j['reference']['store_version']
    man={line.split(maxsplit=1)[1]:line.split()[0] for line in (E0/'MANIFEST.sha256').read_text().splitlines()}
    facts=dict(result_sha256=sha(p),config=j['config'],checks=[],arms={},cases={},timing_coverage=j.get('timing_coverage_complete'),
               qualification=j['qualification'],label='diagnostic subset; no full-population qualification')
    per={x:dict(prod=[],tight=[],tighter=[],nonstat=0,steps=0) for x in ('k8','k16','none')}
    for i in ids:
        refp=E0/f'3d/val64/ST/c{i:03}.json';assert sha(refp)==man[str(refp.relative_to(E0))]
        n0=json.loads(refp.read_text())['judge']['n0']
        W={};fds={};cf={}
        for arm in per:
            rec=j['cases'][str(i)][arm];z=np.load(d/'arrays'/f'{arm}_c{i:03}_prod.npz')
            W[arm]=z['W'];assert W[arm].shape==(6,512) and int(z['case'])==i
            c=certificate(z,'prod');per[arm]['nonstat']+=c['nonstat'];per[arm]['steps']+=15
            facts['checks'].append(bool(c['valid']==rec['cert']['valid'] and c['nonstat']==rec['cert']['nonstat']))
            for lev in ('tight','tighter'):
                if lev=='tighter' and i not in j['config']['tighter_cases']:continue
                zz=np.load(d/'arrays'/f'{arm}_c{i:03}_{lev}.npz');cc=certificate(zz,lev)
                facts['checks'].append(cc['anchor_ok']==rec[lev]['anchor_ok'])
                if lev=='tight':
                    dd=max(np.linalg.norm((W[arm][k]-zz['W'][k])@Phi.T)/n0 for k in range(1,6))
                    facts['checks'].append(bool(abs(dd-rec['tight_distance'])<=1e-10))
                else:
                    zt=np.load(d/'arrays'/f'{arm}_c{i:03}_tight.npz')
                    dd=max(np.linalg.norm((zt['W'][k]-zz['W'][k])@Phi.T)/n0 for k in range(1,6))
                    facts['checks'].append(bool(abs(dd-rec['tight_tighter_distance'])<=1e-10))
                per[arm][lev].append(cc['anchor_ok'])
            per[arm]['prod'].append(c['valid'])
            if arm!='none':
                m=rec['whole_residual'];gf=np.asarray(m['g_full']);ge=np.asarray(m['g_sampled']);den=np.asarray(m['denominator'])
                dg=safe_ratio(np.linalg.norm(ge-gf,axis=1),den);g=safe_ratio(np.linalg.norm(gf,axis=1),den)
                facts['checks'].extend([bool(np.allclose(dg,m['delta_g'],rtol=1e-10,atol=1e-13)),
                                       bool(np.allclose(g,m['full_gamma'],rtol=1e-10,atol=1e-13)),
                                       bool(np.allclose(den,np.asarray(m['residual_norm_full'])*np.asarray(m['jacobian_norm_full']),rtol=1e-12))])
                cf[arm]=dict(delta_g_max=float(dg.max()),full_gamma_max=float(g.max()),nonstat=int((g>.001).sum()))
        for x,y in [('k8','k16'),('k16','none'),('k8','none')]:
            name=f'{x}_to_{y}'
            fds[name]=max(float(np.linalg.norm((W[x][k]-W[y][k])@Phi.T)/n0) for k in range(1,6))
            facts['checks'].append(bool(abs(fds[name]-j['cases'][str(i)]['field_distance'][name])<=1e-10))
        facts['cases'][str(i)]=dict(field_distance=fds,whole_residual=cf)
    for arm,v in per.items():
        es=[j['cases'][str(i)][arm]['e0']['e'] for i in ids]
        facts['arms'][arm]=dict(median_error=float(np.median(es)),worst_error=float(np.max(es)),q1=j['Q1'][arm],
            recomputed_valid=all(v['prod']),recomputed_anchors=all(v['tight']) and all(v['tighter']),
            recomputed_nonstat=v['nonstat'],steps=v['steps'],outliers_above_1pct=int(np.sum(np.asarray(es)>.01)))
        facts['checks'].append(bool(v['nonstat']==j['Q1'][arm]['nonstat']))
    for row in j['timing']:
        facts['checks'].append(all(x['identity'] and x['e0']['eligible'] for x in row['invocations']))
    facts['offline_checks_pass']=all(facts['checks'])
    assert facts['offline_checks_pass'],'saved-array recomputation failed'
    (out/'SUMMARY.json').write_text(json.dumps(facts,indent=2)+'\n')
    lines=['# Frozen POD/ECSW consistency replay','',
           'Diagnostic subset evidence against approved E0 references. Numerical errors retain conditional reference uncertainty; these cases cannot qualify the whole validation population.','',
           'Both frozen HR supports FAIL the new whole-residual admission contract. Q1 below certifies the sampled objective for HR arms and the full objective for the no-HR arm. Passing sampled Q1 does not establish agreement with the full least-squares problem.','',
           f"The field-distance bar is {j['config']['admission']['field_distance']:g}; the whole-residual gradient bar is {j['config']['admission']['gradient']:g}. Full stationarity must pass on at least {1-j['config']['admission']['nonstationary_fraction']:.0%} of pooled steps.", '',
           'The job exited after its complete checkpoint and all timings were saved: the final log print could not serialize a NumPy boolean. The saved metadata string for the finite flag is quarantined; admission is recomputed numerically from arrays. No scientific replay was repeated.','',
           '| arm | median error | worst error | Q1 | cases above one-percent error |','|---|---:|---:|---|---:|']
    for arm,v in facts['arms'].items():lines.append(f"| {arm} | {v['median_error']:.6%} | {v['worst_error']:.6%} | {v['q1']['q1']} | {v['outliers_above_1pct']} |")
    lines+=['','| case | k8→k16 distance | k16→none distance | k8→none distance |','|---|---:|---:|---:|']
    for i,v in facts['cases'].items():
        f=v['field_distance'];lines.append(f"| {i} | {f['k8_to_k16']:.8g} | {f['k16_to_none']:.8g} | {f['k8_to_none']:.8g} |")
    lines+=['','| case | arm | maximum whole-residual gradient discrepancy | maximum full stationarity |','|---|---|---:|---:|']
    for i,v in facts['cases'].items():
        for arm,m in v['whole_residual'].items():lines.append(f"| {i} | {arm} | {m['delta_g_max']:.8g} | {m['full_gamma_max']:.8g} |")
    lines+=['',f"Independent saved-array checks passed: {facts['offline_checks_pass']}. Paired timing coverage complete: {facts['timing_coverage']}.",'',
            'This is a qualification failure for the frozen sampled models, not evidence that POD loses to the coordinate bank. Historical observed accuracy/cost domination remains labelled with its original scope.','',
            '## Glossary','', '- arm: one frozen numerical model; none uses every residual row.',
            '- k8 / k16: archived weighted supports with eight / sixteen points per reduced coefficient.',
            '- median / worst error: median / largest relative error among these diagnostic cases.',
            '- Q1: nonlinear validity, fresh stationarity and independently stricter-solve agreement.',
            '- distance: largest output field difference divided by the reference initial-state norm.',
            '- whole-residual gradient discrepancy: sampled versus full least-squares gradient difference, normalized by full Jacobian and residual norms.',
            '- full stationarity: full least-squares gradient norm with that same normalization.',
            '- E0: refined approved reference; its numerical uncertainty is conditional.','']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in facts.items() if k in ('arms','qualification','offline_checks_pass','timing_coverage')},indent=2))


if __name__=='__main__':main()
