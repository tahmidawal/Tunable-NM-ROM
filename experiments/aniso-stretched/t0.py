import numpy as np, scipy.sparse.linalg as spla, time, sys, resource
import aniso as an
fam=sys.argv[1]; N=int(sys.argv[2])
mu=np.array([33.0,-4.0,.4,.6])
t=time.time(); x,y,A,b=an.problem(fam,mu,N,2.5,2.5); print("assemble",time.time()-t, A.nnz)
t=time.time(); lu=spla.splu(A.tocsc(),permc_spec="MMD_AT_PLUS_A",diag_pivot_thresh=0.0,options=dict(SymmetricMode=True)); tf=time.time()-t
t=time.time(); u=lu.solve(b); ts=time.time()-t
print(fam,N,"factor %.2f solve %.3f"%(tf,ts),"nnzL",lu.L.nnz,"res",np.linalg.norm(A@u-b)/np.linalg.norm(b),"maxrss GB",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, "umax",u.max())
