import numpy as np, time, sys
sys.argv=[sys.argv[0],'32']; exec(open('podfloor.py').read().split("n=int")[0])
n=32; T=[0,.1,.2,.3,.4,.5]
Fd=fam(920399,64); Fi=fields(n,Fd,T); V=fields(n,fam(921777,256),T)
for cnt in [512,2048,4096]:
    X=fields(n,fam(921000,cnt),T).reshape(-1,31**3)
    # randomized range: use Gram on subsample-free: X^T X is 29791^2 too big; use X X^T if small else svd via QR of random proj
    t0=time.time(); 
    if len(X)<=12288:
        G=X@X.T; w,v=np.linalg.eigh(G); w=w[::-1]; v=v[:,::-1]; 
        get=lambda R:(X.T@v[:,:R])/np.sqrt(w[:R])
    else:
        C=X.T@X; w,v=np.linalg.eigh(C); v=v[:,::-1]; get=lambda R:v[:,:R]
    for R in [256,320,384]:
        Q=get(R); out=[]
        for C_ in (Fi,V):
            c=C_.reshape(-1,C_.shape[-1]); e=np.sqrt(np.maximum(1-np.sum((c@Q)**2,1)/np.sum(c*c,1),0)).reshape(C_.shape[0],6)
            out.append((round(100*e.max(),3),int(np.argmax(e[:,0]))))
        print(cnt,R,'final(t0 worst, case)',out[0],'val256',out[1],round(time.time()-t0,1),flush=True)
    del X
print('worst final draw', Fd[np.argmax(1)] if False else '')
