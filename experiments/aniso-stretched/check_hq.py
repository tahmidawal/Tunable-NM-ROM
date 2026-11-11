import numpy as np, scipy.sparse.linalg as spla, hq, aniso as an
mu=np.array([33.,-3.,.4,.6])
# p=1 must reproduce the Q1 code up to K sampling (constant K family A -> identical matrix)
A1,F1,n1=hq.problem("A",mu,16,1.5,0,1); x,y,A,b=an.problem("A",mu,16,1.5,0)
print("p=1 vs Q1 matrix diff",abs(A1-A).max())
# self-convergence against a fine Q1 solve, family B
x,y,A,b=an.problem("B",mu,1024,1.5,1.5); Uf=an.to_full(spla.spsolve(A.tocsc(),b),1024); xe=x
for p in (1,2,4):
    for N in (8,16,32,64):
        Ap,Fp,n1=hq.problem("B",mu,N,1.5,1.5,p); u=spla.spsolve(Ap.tocsc(),Fp)
        P=hq.interp1d(N,1.5,p,xe); U=hq.to_eval(u,n1,P,P)
        print(p,N,n1,"rel l2 %.2e"%(np.linalg.norm(U-Uf)/np.linalg.norm(Uf)), "asym",abs(Ap-Ap.T).max())
