import pickle, numpy as np, os
D="/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ckpt_poisson"
for k in [2,4,6,8,12,16,24,32]:
    d=pickle.load(open(os.path.join(D,f"autodec_K{k}_N64_hbc_stages.pkl"),"rb"))
    Z=np.asarray(d["z_tr"],dtype=np.float64)
    s=Z.std(0); mu=Z.mean(0)
    # SVD of centred Z
    sv=np.linalg.svd(Z-mu,compute_uv=False)
    print(f"k={k:2d} n={Z.shape[0]} rms={np.sqrt((Z**2).mean()):.4f}")
    print(f"   per-dim std  min {s.min():.4g} max {s.max():.4g} ratio {s.max()/s.min():.3g}")
    print(f"   per-dim |mean| min {np.abs(mu).min():.4g} max {np.abs(mu).max():.4g}")
    print(f"   sv ratio (centred) {sv[0]/sv[-1]:.4g}   sv: {np.array2string(sv,precision=3,max_line_width=200)}")
    print(f"   std: {np.array2string(np.sort(s)[::-1],precision=4,max_line_width=200)}")
    print(f"   cfg keys: seed={d['config'].get('train_seed')} ")
