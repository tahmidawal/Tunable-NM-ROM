import numpy as np, scipy.fft as F, time, sys
def fam(seed,count):
    a=np.random.default_rng(seed).random((count,5)); return a*np.array([.3,.3,.3,.05,.4])+np.array([.35,.35,.35,.10,.8])
def fields(n,draws,times,nu=.02):
    a=np.arange(1,n)/n; k=np.arange(1,n); l=4*n*n*np.sin(np.pi*k/(2*n))**2
    lam=l[:,None,None]+l[None,:,None]+l[None,None,:]; out=[]
    for p in draws:
        f=[4*a*(1-a)*np.exp(-(a-p[i])**2/(2*p[3]**2)) for i in range(3)]
        u=p[4]*f[0][:,None,None]*f[1][None,:,None]*f[2][None,None,:]
        c=F.dstn(u,type=1,norm='ortho')
        out.append(np.stack([F.idstn(c*np.exp(-nu*t*lam),type=1,norm='ortho').ravel() for t in times]))
    return np.stack(out)
n=int(sys.argv[1]); T=[0,.1,.2,.3,.4,.5]; t0=time.time()
U=fields(n,fam(921000,512),T); V=fields(n,fam(921001,16),T); D=fields(n,fam(920311,16),T); Fi=fields(n,fam(920399,64),T)
print('gen',time.time()-t0,U.shape)
X=U.reshape(-1,U.shape[-1]); Xn=X/np.linalg.norm(X,axis=1,keepdims=True)
for name,M in [('plain',X),('normalized',Xn)]:
    G=M@M.T; w,v=np.linalg.eigh(G); w=w[::-1]; v=v[:,::-1]
    for R in [64,128,192,256,320,384,512]:
        Q=(M.T@v[:,:R])/np.sqrt(w[:R]); 
        res={}
        for cn,C in [('val',V),('dev',D),('final',Fi)]:
            c=C.reshape(-1,C.shape[-1]); e=np.sqrt(np.maximum(1-np.sum((c@Q)**2,1)/np.sum(c*c,1),0)).reshape(C.shape[0],6)
            res[cn]=(round(100*e.max(),3), round(100*e[:,0].max(),3), round(100*e[:,1:].max(),3))
        print(name,R,'(all,t0,evolved)%',res,flush=True)
