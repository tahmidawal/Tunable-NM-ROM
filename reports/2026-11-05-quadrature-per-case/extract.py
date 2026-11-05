import sys, json
W='/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/'
BQ=W+'2026-11-05-burgers-quadrature-plateau/experiments/burgers-quadrature-plateau'
NQ=W+'2026-11-05-ns3d-offmesh-quadrature/experiments/ns3d-offmesh-quadrature/runs'
sys.path.insert(0, BQ)
import plateau as P
FAM={'fibonacci':'Fibonacci lattice','gauss':'Gauss','sobol':'Sobol','lattice':'Rank-1 lattice','korobov':'Korobov lattice',
     'lat':'Rank-1 lattice','gl':'Gauss','kor':'Korobov lattice','sob':'Sobol','trap':'Shifted trapezoid'}
out=[]
runs=[('p2d256b',256,'2D'),('p2d512',512,'2D'),('p2d1024',1024,'2D'),('p2d2048',2048,'2D'),('p2d4096',4096,'2D'),
      ('p3d65b',64,'3D'),('p3d129',128,'3D'),('p3d257c',256,'3D')]
for run,L,dim in runs:
    rep=json.load(open(f'{BQ}/results/{run}/result.json'))
    cases=P.eligible_cases(rep,'ST'); pl=P.plateau(rep); fs=P.fom_stats(rep)
    ba=P.by_arm(rep)
    fams=[]
    for f,v in pl['families'].items():
        arms=[r['arm'] for r in v['rungs']]
        ms=[r['m'] for r in v['rungs']]
        per={c:[round(ba[a][c]['ST']['e']*100,5) for a in arms] for c in cases}
        g=v['global_']
        fams.append(dict(key=f,name=FAM[f],m=ms,W=[r['W']*100 for r in v['rungs']],T=[r['T']*100 for r in v['rungs']],A=[sum(ba[a][c]['ST']['e'] for c in cases)/len(cases)*100 for a in arms],
            cases=per,mstar=g['m_star'] if v['tail_demonstrated'] else None,
            endpoint=None if v['tail_demonstrated'] else g['m_star'],bracket=g['bracket']))
    ub=fs.get(f'UB_L{L}')
    out.append(dict(pde='burgers',dim=dim,mesh=L,label=f'{L}{"²" if dim=="2D" else "³"}',n=len(cases),n_all=len(rep['cases']),
        ref=dict(name='Plateau (largest rule)',W=pl['conv']['W']*100,T=pl['conv']['T']*100,
                 mean=sum(ba[pl['conv']['arm']][c]['ST']['e'] for c in cases)/len(cases)*100),
        fom=None if ub is None else dict(name='Upwind full solver, same mesh',W=ub['W']*100,T=ub['T']*100,
                 mean=sum(r['ST']['e'] for r in rep['fom_rows'] if r['fom']==f'UB_L{L}' and r['case'] in cases)/len(cases)*100),
        families=fams))
for q,L in [('q32',32),('q64',64),('q96',96)]:
    s=json.load(open(f'{NQ}/{q}/output/summary.json'))
    ev=lambda errs:[max(e[1:])*100 for e in errs]
    ten=ev(s['tensor']['head']['errors']); n=len(ten)
    byf={}
    for rn,r in s['ladder'].items(): byf.setdefault(r['family'],[]).append((r['m'],rn))
    fams=[]
    for f in ['trap','gl','lat','kor','sob']:
        rs=sorted(byf[f]); ms=[m for m,_ in rs]
        E=[ev(s['ladder'][rn]['net']['rollouts']['head']['errors']) for _,rn in rs]
        per={c:[round(E[i][c],5) for i in range(len(rs))] for c in range(n)}
        import numpy as np
        ms_=s['mstar'][f'net/{f}/head']
        fams.append(dict(key=f,name=FAM[f],m=ms,A=[float(np.mean(e)) for e in E],W=[max(e) for e in E],T=[float(np.median(e)) for e in E],cases=per,
            mstar=None if ms_['endpoint_only'] else ms_['mstar'],endpoint=ms_['mstar'] if ms_['endpoint_only'] else None))
    c60=s['cnab2']['60']['stats']; c50=s['cnab2']['50']['stats']
    out.append(dict(pde='ns3d',dim='3D',mesh=L,label=f'{L}³',n=n,n_all=n,
        ref=dict(name='Exact tensor ROM',W=max(ten),T=float(np.median(ten)),mean=float(np.mean(ten))),
        fom=dict(name='CNAB2, 60 steps',W=c60['evolved_worst']*100,T=c60['evolved_median']*100,
                 mean=float(np.mean(ev(s['cnab2']['60']['errors'])))),
        families=fams))
json.dump(out,open(sys.argv[1],'w'),separators=(',',':'))
for o in out: print(o['pde'],o['label'],o['n'],[ (f['key'],f['mstar'],f['endpoint']) for f in o['families']], round(o['ref']['mean'],3), o['fom'] and round(o['fom']['mean'],3), [round(x,3) for x in o['families'][0]['A']][-3:])
