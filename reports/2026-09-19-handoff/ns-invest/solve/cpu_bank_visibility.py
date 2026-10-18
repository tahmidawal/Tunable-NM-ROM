"""CPU: bank on the 256^2 grid in NumPy (silu MLP), its singular values, and the test-space
projected bank Phi^T G via FFT (only |k|<=26.2 modes): visibility conditioning; head Jacobian conditioning."""
import numpy as np, pickle, json
SP='/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/6e5fe858-5d6b-4b34-a1bd-df699b1d4032/scratchpad/ns-invest/solve'
ck=pickle.load(open('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d/checkpoints/ckpt_K32_R512.pkl','rb'))
p=ck['params']; N=256
silu=lambda x: x/(1+np.exp(-x))
def mlp(ps,x):
    for w,b in ps[:-1]: x=silu(x@np.asarray(w)+np.asarray(b))
    w,b=ps[-1]; return x@np.asarray(w)+np.asarray(b)
x=np.arange(N)/N; X,Y=np.meshgrid(x,x,indexing='ij'); xy=np.stack([X.ravel(),Y.ravel()],1)
ang=2*np.pi*(xy@np.asarray(p['B'])); ff=np.concatenate([np.sin(ang),np.cos(ang)],-1)
G=float(p['out_scale'])*mlp(p['g'],ff); G=G-G.mean(0,keepdims=True)
print('G',G.shape)
sG=np.linalg.svd(G,compute_uv=False); print('sv(G) max/min',sG[0],sG[-1],'cond',sG[0]/sG[-1])
k=np.fft.fftfreq(N,1/N); KX,KY=np.meshgrid(k,k,indexing='ij'); K2=KX**2+KY**2
inside=K2<=689
# energy of each bank column inside the test space
Fk=np.fft.fft2(G.T.reshape(512,N,N)); P=np.abs(Fk)**2
fin=P[:,inside].sum(1)/P.sum((1,2))
print('bank columns: fraction of energy inside |k|<=26.2: min %.6f median %.6f'%(fin.min(),np.median(fin)))
# the projected bank A = Phi^T G restricted to the test space: singular values via the low-pass filtered bank (Parseval)
Gl=np.real(np.fft.ifft2(Fk*inside[None])).reshape(512,-1).T
sA=np.linalg.svd(Gl,compute_uv=False); print('sv(Phi^T G) max/min',sA[0],sA[-1],'cond',sA[0]/sA[-1],'  ratio min sv(A)/min sv(G)',sA[-1]/sG[-1])
# helmholtz-scaled at nu=1e-3 and 1e-2, dt=2e-3
lam=(2-2*np.cos(2*np.pi*KX/N)+2-2*np.cos(2*np.pi*KY/N))*N*N
out=dict(cond_G=float(sG[0]/sG[-1]),cond_PhiG=float(sA[0]/sA[-1]),bank_frac_inside_min=float(fin.min()),bank_frac_inside_median=float(np.median(fin)))
for nu in [1e-3,1e-2]:
    sc=1/(1+0.5*2e-3*nu*lam)
    Gs=np.real(np.fft.ifft2(Fk*inside[None]*sc[None])).reshape(512,-1).T
    s=np.linalg.svd(Gs,compute_uv=False); print(f'nu={nu}: Helmholtz-scaled projected bank cond {s[0]/s[-1]:.1f}, min scale factor {sc[inside].min():.3f}')
    out[f'cond_scaled_nu{nu}']=float(s[0]/s[-1])
# head Jacobian at a few training codes: dh/dz (512 x 32), in the whitened (field) metric R_b dh/dz
Z=np.asarray(ck['Z_tr']); Rb=np.linalg.qr(G,mode='r')
def head(z):
    return mlp(p['h'],z)+z@np.asarray(p['h_lin'])
rng=np.random.default_rng(0); conds=[]; svmin=[]; svmax=[]
for i in rng.choice(len(Z),16,replace=False):
    z=Z[i]; eps=1e-6; J=np.stack([(head(z+eps*np.eye(32)[j])-head(z-eps*np.eye(32)[j]))/(2*eps) for j in range(32)],1)
    s=np.linalg.svd(Rb@J,compute_uv=False); conds.append(s[0]/s[-1]); svmin.append(s[-1]); svmax.append(s[0])
print('field-metric head Jacobian R_b dh/dz: cond median %.1f max %.1f; sv max median %.3g min median %.3g'%(np.median(conds),max(conds),np.median(svmax),np.median(svmin)))
out.update(head_jac_cond_median=float(np.median(conds)),head_jac_cond_max=float(max(conds)))
print('code radius (max ||z - mean||):',float(np.max(np.linalg.norm(Z-Z.mean(0),axis=1))),' median ||z||',float(np.median(np.linalg.norm(Z,axis=1))))
json.dump(out,open(f'{SP}/cpu_bank_visibility.json','w'),indent=1)
