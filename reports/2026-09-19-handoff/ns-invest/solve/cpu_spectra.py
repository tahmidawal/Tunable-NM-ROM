"""CPU: spectra of truth / ROM / error, energy+enstrophy budgets, test-space cutoff fractions."""
import numpy as np, json
SP='/tmp/claude-1002/-home-tahmid-Dev-pod-ae-nmrom-Tunable-NM-ROM-Claude/6e5fe858-5d6b-4b34-a1bd-df699b1d4032/scratchpad/ns-invest/solve'
ref=np.load(f'{SP}/ns304/experiments/ns2d/output/reference_N256.npz'); rom=np.load(f'{SP}/ns304/experiments/ns2d/output/rom_fields_N256.npz')
U=ref['U']; N=256; print('U',U.shape, 'eval_idx',ref['eval_idx'])
k=np.fft.fftfreq(N,1/N); KX,KY=np.meshgrid(k,k,indexing='ij'); K2=KX**2+KY**2; KA=np.sqrt(K2)
lam=(2-2*np.cos(2*np.pi*KX/N)+2-2*np.cos(2*np.pi*KY/N))*N*N; lam[0,0]=1
# M=2176 test space: 1088 wavevectors in the half-plane ordered by |k|^2 -> k2 cutoff
cands=sorted((kx*kx+ky*ky,kx,ky) for kx in range(-127,128) for ky in range(-127,128) if (kx,ky)!=(0,0) and not (kx<0 or (kx==0 and ky<0)))
k2cut=cands[1087][0]; print('M=2176 -> 1088 wavevectors, |k|^2 cutoff',k2cut,'|k|<=',np.sqrt(k2cut), ' (next |k|^2', cands[1088][0],')')
inside=(K2<=k2cut)
def spec_frac(w):
    F=np.fft.fft2(w); P=np.abs(F)**2; P[0,0]=0
    return P[inside].sum()/P.sum()
def energy(w):
    F=np.fft.fft2(w); psiF=F/lam; psiF[0,0]=0; psi=np.real(np.fft.ifft2(psiF)); return 0.5*np.sum(psi*w)/N**2
def enst(w): return 0.5*np.sum(w*w)/N**2
out={}
print('\nTRUTH: fraction of enstrophy inside the test space (|k|<=%.1f), per case per time'%np.sqrt(k2cut))
for c in range(8):
    fr=[spec_frac(U[c,t].reshape(N,N)) for t in range(6)]
    print(c, np.round(fr,5))
    out[f'truth_frac_inside_case{c}']=fr
# time-derivative proxy: difference between consecutive eval states (coarse) -- and the residual-like quantity: J_A + nu*lap of the truth would need nu; skip.
print('\nERROR spectra: fraction of the ROM error energy inside the test space, and at |k|<=4 (the IC band), q0 and q512, per case (t=0.2..1.0)')
for s in ['neural_q0','neural_q512','pod_k32','pod_k544']:
    for c in [0,1,2,4]:
        R=rom[f'{s}__case{c}']
        e=R-U[c].reshape(6,N,N)
        fi=[spec_frac(e[t]) for t in range(1,6)]
        lo=[]
        for t in range(1,6):
            F=np.fft.fft2(e[t]); P=np.abs(F)**2; P[0,0]=0; lo.append(P[K2<=4].sum()/P.sum())
        print(f'{s:12s} case{c}: inside-testspace {np.round(fi,4)}  |k|<=2 band {np.round(lo,3)}')
print('\nENERGY / ENSTROPHY of ROM vs truth (ratio rom/truth) at t=0.2..1.0')
for s in ['neural_q0','neural_q256','neural_q512','pod_k32','pod_k544']:
    for c in [0,1,2,4]:
        R=rom[f'{s}__case{c}']; T=U[c].reshape(6,N,N)
        er=[energy(R[t])/energy(T[t]) for t in range(1,6)]; zr=[enst(R[t])/enst(T[t]) for t in range(1,6)]
        print(f'{s:12s} case{c}: E ratio {np.round(er,4)}  Z ratio {np.round(zr,4)}')
print('\nTRUTH energy and enstrophy decay (case 1, Re 890 and case 0, Re 118): E(t)/E(0), Z(t)/Z(0)')
for c in [0,1]:
    T=U[c].reshape(6,N,N); print(c, np.round([energy(T[t])/energy(T[0]) for t in range(6)],4), np.round([enst(T[t])/enst(T[0]) for t in range(6)],4))
print('\nCURRENT-relative error (||e||/||w(t)||) q0 median over cases:')
import json
d=json.load(open('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d/artifacts/ns304/result.json'))
inv=[i for i in d['invocations'] if i['timed'] and i['rep']==3]
for s in ['neural_q0','neural_q512','pod_k32']:
    rows=sorted([i for i in inv if i['subject']==s],key=lambda r:r['case'])
    print(s, np.round(np.median([r['current_per_time'] for r in rows],0),4))
json.dump(out,open(f'{SP}/cpu_spectra.json','w'),indent=1)
