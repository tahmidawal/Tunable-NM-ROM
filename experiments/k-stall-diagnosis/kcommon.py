"""Shared setup for the k-stall diagnostic.  READ-ONLY against the worktrees."""
import os, sys, pickle
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

WT = "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees"
PRO = f"{WT}/2026-08-17-cost-to-tolerance/experiments/poisson2d-rom-objective"
MS  = f"{WT}/2026-08-14-multistage-precision/experiments/multistage-precision"
CTOL= f"{WT}/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance"
CKPT= f"{CTOL}/ckpt_poisson"
for p in (MS, PRO, os.path.join(PRO,"followup"), CTOL):
    sys.path.insert(0, p)
import ms_parametric as mp
import pro_common as pc

F64 = jnp.float64
N_TEST = 16
N_TRAIN = mp.N_TRAIN

def load(k):
    d, cfg, stages, Z_tr, hb = pc.load_pkl(os.path.join(CKPT, f"autodec_K{k}_N64_hbc_stages.pkl"))
    return pc.make_decoder(stages[:1], hard_bc=bool(hb)), np.asarray(Z_tr)

def testbed(n=64):
    """(grid, Fs (16,ni,ni), U_int truth, tn, nn_idx, params)"""
    cx, cy, w, a, z = mp.sample_params()
    grid = pc.Grid(n)
    Fs = np.stack([mp.source_interior(n, cx[N_TRAIN+i], cy[N_TRAIN+i],
                                      w[N_TRAIN+i], a[N_TRAIN+i]) for i in range(N_TEST)])
    op = lambda v: mp.neg_lap_interior(v, n)
    solve = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(op, F, tol=mp.CG_TOL,
                                                         maxiter=mp.CG_MAXITER)[0])
    U = np.asarray(jax.lax.map(solve, jnp.asarray(Fs)))
    tn = np.array([np.linalg.norm(U[i]) for i in range(N_TEST)])
    zt = np.asarray(z)
    nn = np.argmin(((zt[N_TRAIN:N_TRAIN+N_TEST,None,:]-zt[None,:N_TRAIN,:])**2).sum(-1), axis=1)
    par = dict(cx=cx[N_TRAIN:N_TRAIN+N_TEST], cy=cy[N_TRAIN:N_TRAIN+N_TEST],
               w=w[N_TRAIN:N_TRAIN+N_TEST], a=a[N_TRAIN:N_TRAIN+N_TEST],
               ztheta=zt[N_TRAIN:N_TRAIN+N_TEST], ztheta_tr=zt[:N_TRAIN])
    return grid, Fs, U, tn, nn, par

def weak_full(grid, M):
    """FULL-GRID weak-form objective pieces: r(z) = Wl*(PhiT@(wq*u(z,pts))) - f_m
    with pts = all interior nodes and wq = 1 (exact discrete projection).  This is
    the m -> full limit of the hyper-reduced objective, so it removes NNLS-EQ from
    the diagnostic entirely."""
    spec = dict(kind="weak", alpha=1.0, M=M)
    pts = np.asarray(grid.coords_int)
    PhiT, Wl = pc.colloc_mode_table(grid, spec, "grid", pts)
    wq = jnp.ones(pts.shape[0], F64)
    return spec, jnp.asarray(pts), wq, PhiT, Wl

def fm(grid, spec, F):
    return pc.weak_source_term(grid, spec, "grid", F)
