"""gen_e1.py -- generate the E1 cohorts at 64^2 with the extended certified FOM (GPU, vmapped in chunks).
env: FAMS=A,B,C,D,E  NTR=128 NDEV=32 CHUNK=32 SMOKE=0 OUTDIR=.  N=64
Saves <OUTDIR>/data_<fam>.npz with U_tr (ntr, NOUT, N^2), U_dev, P_tr, P_dev, newton iters and
residuals, and timings.  Newton ntol 1e-11, BiCGStab ltol 1e-9 (the lane's settings)."""
import os, sys, time, json
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, jax, jax.numpy as jnp
import fom_ext, families as FA
F = fom_ext.F
N = int(os.environ.get('N', '64'))
NTR, NDEV = int(os.environ.get('NTR', '128')), int(os.environ.get('NDEV', '32'))
CHUNK = int(os.environ.get('CHUNK', '32'))
SMOKE = int(os.environ.get('SMOKE', '0'))
OUTDIR = os.environ.get('OUTDIR', HERE)
FAMS = os.environ.get('FAMS', 'A,B,C,D,E').split(',')
NTOL, LTOL = 1e-11, 1e-9
SEED_TR, SEED_DEV = 20260917, 20260918
t_begin = time.time()
print('backend', jax.default_backend(), 'x64', jax.config.jax_enable_x64, flush=True)
assert jax.default_backend() == 'gpu'


def gen_family(fam, P):
    m = FA.META[fam]
    run1, runv = fom_ext.make_fom_ext(N, m['dt'], m['nsteps'], m['out_every'], forced=m['forced'])
    nout = m['nsteps'] // m['out_every'] + 1
    ics = [FA.initial(fam, N, p) for p in P]
    W0 = np.stack([w for w, _, _, _ in ics]); NU = np.array([nu for _, nu, _, _ in ics])
    UV = np.stack([uv for _, _, uv, _ in ics]); FP = np.stack([fp for _, _, _, fp in ics])
    U, IT, RN, cfl = [], [], [], 0.0
    for s in range(0, len(P), CHUNK):
        t0 = time.time()
        st, it, rn = runv(jnp.asarray(W0[s:s + CHUNK]), jnp.asarray(NU[s:s + CHUNK]), jnp.asarray(UV[s:s + CHUNK]),
                          jnp.asarray(FP[s:s + CHUNK]), NTOL, LTOL)
        st = np.asarray(st); it = np.asarray(it); rn = np.asarray(rn)
        assert np.isfinite(st).all(), f'{fam}: non-finite states in chunk {s}'
        # CFL with the FOM's discrete velocity plus the background flow
        c_int = max(F.cfl_of(st[i], N, m['dt'])[0] for i in range(min(4, st.shape[0])))
        c_bg = float(np.abs(UV[s:s + CHUNK]).max() * m['dt'] * N)
        cfl = max(cfl, c_int + c_bg)
        U.append(st.reshape(st.shape[0], nout, -1)); IT.append(it); RN.append(rn)
        print(f'  {fam} chunk {s}-{s+st.shape[0]}: {time.time()-t0:.1f}s  worst_res {rn.max():.2e}  '
              f'newton mean {it.mean():.2f} max {it.max()}  cfl<={cfl:.2f}  max|w| {np.abs(st).max():.1f}  [{time.time()-t_begin:.0f}s]', flush=True)
    return np.concatenate(U), np.concatenate(IT), np.concatenate(RN), cfl, W0, NU, UV, FP


if SMOKE:
    # A: 4 dev trajectories must reproduce the head investigation's data64.npz (same seeds, lane FOM)
    ref = np.load(os.path.join(HERE, '..', 'head', 'data64.npz'))
    P = FA.draw('A', SEED_DEV, 64)[:4]
    U, IT, RN, cfl, *_ = gen_family('A', P)
    diff = np.abs(U - ref['U_dev'][:4]).max() / np.abs(ref['U_dev'][:4]).max()
    print(f'SMOKE A: max rel diff vs head data64 U_dev[:4] = {diff:.3e}  (bitwise-level expected)', flush=True)
    for fam in ('B', 'C', 'D', 'E'):
        P = FA.draw(fam, SEED_DEV, 64)[:2]
        U, IT, RN, cfl, *_ = gen_family(fam, P)
        E, Z = fom_ext.energy_enstrophy_np(U.reshape(-1, N * N), N)
        E, Z = E.reshape(2, -1), Z.reshape(2, -1)
        print(f'SMOKE {fam}: E(T)/E(0) {E[:, -1] / E[:, 0]}  Z(T)/Z(0) {Z[:, -1] / Z[:, 0]}  params {P.round(3).tolist()}', flush=True)
    print('SMOKE-DONE', time.time() - t_begin, flush=True)
    sys.exit(0)

for fam in FAMS:
    m = FA.META[fam]
    Ptr, Pdev = FA.draw(fam, SEED_TR, 512)[:NTR], FA.draw(fam, SEED_DEV, 64)[:NDEV]
    t0 = time.time()
    Udev, ITd, RNd, cfld, *_ = gen_family(fam, Pdev)
    Utr, ITt, RNt, cflt, *_ = gen_family(fam, Ptr)
    np.savez_compressed(os.path.join(OUTDIR, f'data_{fam}.npz'), U_tr=Utr, U_dev=Udev, P_tr=Ptr, P_dev=Pdev,
                        it_tr=ITt, it_dev=ITd, rn_tr=RNt, rn_dev=RNd, N=N, cfl=max(cfld, cflt),
                        meta=json.dumps({k: v for k, v in m.items()}), seconds=time.time() - t0,
                        seeds=np.array([SEED_TR, SEED_DEV]), ntol=NTOL, ltol=LTOL)
    print(f'{fam} DONE  {Utr.shape} {Udev.shape}  worst_res {max(RNt.max(), RNd.max()):.2e}  '
          f'newton/step mean {np.concatenate([ITt.ravel(), ITd.ravel()]).mean():.3f} max {max(ITt.max(), ITd.max())}  '
          f'cfl {max(cfld, cflt):.2f}  {time.time()-t0:.0f}s  [{time.time()-t_begin:.0f}s]', flush=True)
print('ALL-DONE', time.time() - t_begin, flush=True)
