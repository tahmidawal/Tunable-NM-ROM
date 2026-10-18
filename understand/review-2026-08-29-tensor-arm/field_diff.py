import os, sys, numpy as np
sys.path.insert(0,'/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-29-b1d-tensor/experiments/separable-decoder')
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import b1d_common as b1, b1d_fast_common as fc, b1d_tensor_common as tc
N=int(os.environ['N'])
ck=f'/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-29-b1d-tensor/experiments/separable-decoder/runs/b1dqf/b1ds_n{N}/out/sep_b1d_scale_n{N}.pkl'
su=fc.Setup(ck,N); G=np.asarray(su.G_int); Phi=np.asarray(su.Phi_j)
Q=tc.symmetrize(tc.build_T(Phi,G,su.dx))
U_test,nu_test=fc.gen_test(N); interior=su.interior
ic=fc.make_ic_ref(su)
ops={'or':fc.make_device_ref(su,su.make_full_rw()),'T':fc.make_device_ref(su,su.make_tensor_rw(Q))}
dec=su.decode_all
for ti in range(fc.N_TEST):
    nu=float(nu_test[ti]); u0=U_test[ti,0]; u0i=jnp.asarray(u0[interior])
    tol=fc.STEP_TOL*float(np.sqrt(np.mean(u0[interior]**2)))*float(np.sqrt(su.n_i))
    z0,_=ic(u0i); out={}
    for k,o in ops.items():
        Zk,rn,nJ,rs=o['rollout'](z0,nu,tol,fc.GN_BUDGET); out[k]=(np.asarray(Zk),np.asarray(rn),np.asarray(nJ),np.asarray(rs))
    Zo,ZT=out['or'][0],out['T'][0]
    Uo=np.asarray(dec(jnp.asarray(Zo))); UT=np.asarray(dec(jnp.asarray(ZT))); tru=U_test[ti,1:][:,interior]
    fd=np.linalg.norm(UT-Uo,axis=1)/np.linalg.norm(Uo,axis=1)
    eo=np.linalg.norm(Uo-tru,axis=1)/np.linalg.norm(tru,axis=1); eT=np.linalg.norm(UT-tru,axis=1)/np.linalg.norm(tru,axis=1)
    rd=out['or'][1]; rT=out['T'][1]
    print(f"N={N} traj {ti} nu={nu:.4f}: field diff T-vs-or rel L2: max {fd.max():.2e} mean {fd.mean():.2e} | err-metric diff {abs(eT.mean()-eo.mean()):.2e} | latdev {np.max(np.abs(ZT-Zo)):.2e} | reasons equal {np.array_equal(out['or'][3],out['T'][3])} njac equal {np.array_equal(out['or'][2],out['T'][2])} | rn_final rel diff max {np.max(np.abs(rT-rd)/rd):.1e} | stall-rel-dec margin n/a")
