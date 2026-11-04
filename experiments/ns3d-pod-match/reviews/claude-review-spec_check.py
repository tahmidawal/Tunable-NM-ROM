import sys, time, numpy as np
W="/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-ns3d-pod-match/experiments"
for e in ("ns3d","ns3d-shift","ns3d-grok","ns2d","separable-decoder"): sys.path.insert(0, f"{W}/{e}")
import jax, jax.numpy as jnp
import ns3d_fom as F
n=64; par=F.parameters(202609201, 4)
solver=F.make_solver(0.001,200,40); g=F.geometry(n)
k=np.abs(np.fft.fftfreq(n)*n); kk=np.maximum(np.maximum(k[:,None,None],k[None,:,None]),k[None,None,:])
t=time.time()
for row in par[:3]:
    fr=np.asarray(solver(jnp.asarray(F.initial(n,row)),float(row[-1]),g))
    for f in (fr[0],fr[-1]):
        s=(np.abs(np.fft.fftn(f,axes=(1,2,3)))**2).sum(0)
        print("nu=%.4f"%row[-1], " >10: %.2e  >=16: %.2e"%(s[kk>10].sum()/s.sum(), s[kk>=16].sum()/s.sum()))
print(time.time()-t)
