import numpy as np, json
def params_draw(seed, count):   # copy of mr-burgers2d/engines.py:20
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(.15,.85,count), r.uniform(.15,.85,count), r.uniform(.05,.20,count),
                     r.uniform(.5,2.,count), np.exp(r.uniform(np.log(.01),np.log(.1),count))], axis=1)
def sample_params(seed, m):     # copy of burgers2d_film.sample_params (non-z part)
    rng=np.random.default_rng(seed); cx=rng.uniform(.15,.85,m); cy=rng.uniform(.15,.85,m)
    w=rng.uniform(.05,.2,m); a=rng.uniform(.5,2.,m); nu=np.exp(rng.uniform(np.log(.01),np.log(.1),m))
    return np.stack([cx,cy,w,a,nu],1)
def case_seed(split, i):
    codes=dict(calibration=0,train=1,validation=2)
    return int(np.random.SeedSequence([20260914,22,codes[split],i]).generate_state(1,dtype=np.uint32)[0])
train = np.concatenate([sample_params(0,576), sample_params(1000,4032)])
assert np.array_equal(sample_params(1000,4032), params_draw(1000,4032))
print('train', train.shape, 'unique rows', len(np.unique(train,axis=0)))
vseeds=[case_seed('validation',i) for i in range(32)]
cohorts = {
 'val32': np.stack([params_draw(s,1)[0] for s in vseeds]),
 'hold64': params_draw(20260916,64),
 'dev6': np.concatenate([params_draw(7090702,4), params_draw(911702,2)]),
 'sealed6': params_draw(17092026,6),
}
print('validation case seeds in {0,1000}?', any(s in (0,1000) for s in vseeds), 'min/max', min(vseeds), max(vseeds))
lo=np.array([.15,.15,.05,.5,.01]); hi=np.array([.85,.85,.2,2.,.1])
def scaled(P): Q=P.copy(); Q[:,4]=np.log(Q[:,4]); l=lo.copy(); h=hi.copy(); l[4]=np.log(l[4]); h[4]=np.log(h[4]); return (Q-l)/(h-l)
Ts=scaled(train)
# also: any evaluation scalar equal to any training scalar (column-reuse landmine)
tr_vals=set(train.ravel().tolist())
for k,P in cohorts.items():
    d=np.sqrt(((scaled(P)[:,None,:]-Ts[None])**2).sum(-1))
    nn=d.min(1)
    shared=sum(v in tr_vals for v in P.ravel().tolist())
    print(f'{k:7s} n={len(P):3d} min unit-cube dist to train {nn.min():.4e} (median nn {np.median(nn):.3e}) exact-row-matches {int((d==0).sum())} allclose(1e-8) {int((d<1e-8).sum())} shared-scalars {shared}')
# check recorded values
sc=json.load(open('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-seeds/experiments/b-seeds/checks/sealed-cohort.json'))
print('sealed json matches regen', np.allclose(sc['physical_cases'],cohorts['sealed6'],rtol=1e-14,atol=0), 'dev json matches', np.allclose(sc['development_cases'],cohorts['dev6'],rtol=1e-14,atol=0))
# typical train nn spacing for scale
dtt=np.sqrt(((Ts[:500,None]-Ts[None])**2).sum(-1)); np.fill_diagonal(dtt[:, :500],np.inf); print('train self nn median', np.median(dtt.min(1)))
# cross-cohort overlap among eval cohorts
import itertools
for (a,A),(b,B) in itertools.combinations(cohorts.items(),2):
    print(a,b,'min dist',np.sqrt(((scaled(A)[:,None]-scaled(B)[None])**2).sum(-1)).min())
