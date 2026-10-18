import json, numpy as np, glob, re
D='/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-29-b1d-tensor/experiments/separable-decoder/runs/b1dtensor'
rng=np.random.default_rng(0)
print("=== (4) tensor vs NNLS-32 / learned-32 per trajectory ===")
allp=[]
for p in sorted(glob.glob(D+'/tensor_n*/out/*.json'), key=lambda s:int(re.search(r'_n(\d+)\.json',s).group(1))):
    d=json.load(open(p)); N=d['config']['N']; v=d['variants']
    t=np.array([r['err_fast'] for r in v['tensor']['rollout']])
    b=np.array([r['err_fast'] for r in v['base_tight']['rollout']])
    l=np.array([r['err_fast'] for r in v['nodes_tight']['rollout']])
    o=np.array([r['err_fast'] for r in v['oracle']['rollout']])
    for lab,x in (('NNLS',b),('learned',l)):
        dlt=t-x; wins=int((dlt<0).sum()); n=len(dlt)
        # exact two-sided sign test
        from math import comb
        k=min(wins,n-wins); p2=sum(comb(n,i) for i in range(0,k+1))*2/2**n
        # paired bootstrap of mean ratio
        idx=rng.integers(0,n,(20000,n)); ratios=t[idx].mean(1)/x[idx].mean(1)
        lo,hi=np.quantile(ratios,[0.025,0.975])
        print(f"N={N} tensor vs {lab}: wins {wins}/{n} sign-test p={p2:.3f}; mean ratio {t.mean()/x.mean():.3f} boot95 [{lo:.3f},{hi:.3f}]; per-traj ratio min/max {np.min(t/x):.3f}/{np.max(t/x):.3f}")
    allp.append((N,t,b,o))
    # tol vs final residual
    n_i=d['config']['n_interior']
    rn=np.array([r['rn_final_fast'] for r in v['oracle']['rollout']])
    print(f"   oracle rn_final: median {np.median(rn):.2e} min {rn.min():.2e}; mean njac {np.mean([r['mean_njac_fast'] for r in v['oracle']['rollout']]):.2f}")
# pooled across N (same 8 trajectories -> not independent, report anyway)
t=np.concatenate([a[1] for a in allp]); b=np.concatenate([a[2] for a in allp])
print(f"pooled 32 pairs: wins {(t<b).sum()}/32, mean ratio {t.mean()/b.mean():.3f}")
# per-trajectory averaged over N (8 independent-ish units)
tm=np.mean([a[1] for a in allp],0); bm=np.mean([a[2] for a in allp],0)
print("per-traj N-averaged ratio:", np.round(tm/bm,3), "wins", (tm<bm).sum(),"/8")

print("\n=== (2)/(6) audit candidate rows ===")
for p in sorted(glob.glob(D+'/audit/audit_n*.json'), key=lambda s:int(re.search(r'_n(\d+)\.json',s).group(1))):
    a=json.load(open(p)); N=a['config']['N']; rows=a['TL_candidates']['rows']
    cand=[r for r in rows if r['kind']=='cand']; neg=[r for r in rows if r['n_neg']>0]; pos=[r for r in rows if r['n_neg']==0]
    rr=np.array([r['r_rel'] for r in rows]); rrn=np.array([r['r_rel'] for r in neg]); rrp=np.array([r['r_rel'] for r in pos])
    print(f"N={N}: all rows {len(rows)} r_rel med {np.median(rr):.1e}; u<=0 rows {len(neg)} r_rel med {np.median(rrn):.1e} p95 {np.quantile(rrn,.95):.1e} max {rrn.max():.1e}; positive rows {len(pos)} r_rel max {rrp.max():.1e}")
    # last accepted candidate per (traj,step): stationarity + rn/tol
    last={}
    for r in rows:
        key=(r['traj'],r['step'])
        if r['kind']=='init' or r['accepted']: last[key]=r
    st=np.array([r['g_stationarity'] for r in last.values()]); rt=np.array([r['rn_over_tol'] for r in last.values()])
    print(f"   final accepted state per step: n={len(st)} stationarity |J^T r|/(|J||r|) median {np.median(st):.2e} p90 {np.quantile(st,.9):.2e} max {st.max():.2e}; frac >1e-2: {np.mean(st>1e-2):.2f}; rn/tol median {np.median(rt):.1e} min {rt.min():.1e}")
    # undershoot depth vs mismatch
    mu=np.array([-r['min_u'] for r in neg]); q=np.array([r['q_rel'] for r in neg])
    m=(mu>1e-5)&(q>1e-12)
    sl=np.polyfit(np.log(mu[m]),np.log(q[m]),1)[0]
    print(f"   log-log slope q_rel vs undershoot depth (u<=0 rows): {sl:.2f}  (corr {np.corrcoef(np.log(mu[m]),np.log(q[m]))[0,1]:.2f}); n_neg med {np.median([r['n_neg'] for r in neg]):.0f} max {max(r['n_neg'] for r in neg)}")
    # rejected candidates: margin of rn_new vs rn?  not stored. accepted rel_dec not stored.
    tj=a['TJ_unperturbed']; print(f"   TJ unpert J_rel max {tj['J_rel']['max']:.1e} g_cos_min {tj['g_cos_min']:.6f}; TC per_entry max {a['TC_contraction']['per_entry_max']:.1e} med {a['TC_contraction']['per_entry_median']:.1e} vs_qmax max {a['TC_contraction']['vs_qmax_max']:.1e}")
