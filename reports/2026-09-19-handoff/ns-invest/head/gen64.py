"""gen64.py -- regenerate the ns2d dev cohort (64 traj) and the first n train trajectories at 64^2
with the lane's certified FOM (read-only import), for CPU analysis. Time-capped; saves progressively."""
import os, sys, time
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
LANE = '/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d'
sys.path.insert(0, LANE)
import numpy as np, jax, jax.numpy as jnp
import ns2d_fom as F
N, DT, NSTEPS, OUT_EVERY, NTOL, LTOL = 64, 2e-3, 500, 20, 1e-11, 1e-9
CAP = float(os.environ.get('CAP', '780'))   # seconds of wall clock for the whole script
OUT = os.environ.get('OUT', 'data64.npz')
t_begin = time.time()
print('backend', jax.default_backend(), flush=True)
run, _ = F.make_fom(N, DT, NSTEPS, OUT_EVERY)
NOUT = NSTEPS // OUT_EVERY + 1
phys = dict(dev=F.params_draw(20260918, 64), train=F.params_draw(20260917, 512))
def gen(P, cap_at):
    U = []; worst = 0.0; mit = 0
    for i, p in enumerate(P):
        st, it, rn = run(jnp.asarray(F.initial(N, p)), float(p[-1]), NTOL, LTOL)
        st = np.asarray(st); assert np.isfinite(st).all()
        U.append(st.reshape(NOUT, -1)); worst = max(worst, float(np.max(rn))); mit = max(mit, int(np.max(it)))
        if i == 0 or (i + 1) % 16 == 0:
            print(f'  traj {i+1}/{len(P)}  worst_res {worst:.2e} max_it {mit}  [{time.time()-t_begin:.0f}s]', flush=True)
        if time.time() - t_begin > cap_at:
            print(f'  time cap reached after {i+1} trajectories', flush=True); break
    return np.stack(U), worst, mit
Udev, wd, md = gen(phys['dev'], CAP * 0.45)
print('dev done', Udev.shape, wd, md, flush=True)
np.savez_compressed(OUT, U_dev=Udev, phys_dev=phys['dev'], phys_train=phys['train'], N=N)
Utr, wt, mt = gen(phys['train'], CAP)
print('train done', Utr.shape, wt, mt, flush=True)
np.savez_compressed(OUT, U_dev=Udev, U_tr=Utr, phys_dev=phys['dev'], phys_train=phys['train'], N=N,
                    worst_res=np.array([wd, wt]), max_it=np.array([md, mt]))
print('ALL-DONE', time.time() - t_begin, flush=True)
