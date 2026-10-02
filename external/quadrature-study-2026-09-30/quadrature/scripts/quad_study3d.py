"""3D quadrature-error study (Burgers 3D): rho of every rule vs a converged continuum reference on
projected training states, rho vs the dense mesh target at each N, EQ/lattice mesh rules, timing.
usage: PYGPU scripts/quad_study3d.py [--meshes 64,128] [--q_acc 64]"""
import argparse, sys, os, time, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom, jax, jax.numpy as jnp
from nmrom.dim3 import *
from nmrom.rom import ReducedProblem, scalar_params
from nmrom.quadrature import fit_eq_nnls
ap = argparse.ArgumentParser()
ap.add_argument("--meshes", default="64,128"); ap.add_argument("--q_acc", type=int, default=64)
ap.add_argument("--n_states", type=int, default=96); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
ap.add_argument("--eq_nfit", type=int, default=16); ap.add_argument("--eq_cand", type=int, default=8192)
args = ap.parse_args()
pde = BURGERS3D
ckpt = pickle.load(open(f"{args.data}/burgers3d_model.pkl", "rb"))
ckpt["train_params"] = pickle.load(open(f"{args.data}/burgers3d_train_params.pkl", "rb"))
evals = pickle.load(open(f"{args.data}/burgers3d_eval.pkl", "rb"))
k, R = ckpt["k"], ckpt["R"]; M_acc, M_fast = 4 * (k + args.q_acc), 4 * k
meshes = [int(m) for m in args.meshes.split(",")]
rows = []; log = open(f"{args.out}/burgers3d_quad_study.log", "w")
def say(*a): print(*a, flush=True); print(*a, file=log, flush=True)
rng = np.random.default_rng(0)
sel = rng.choice(len(ckpt["codes"]), min(args.n_states, len(ckpt["codes"])), replace=False)
states = [ckpt["codes"][i] for i in sel]; state_p = [scalar_params(ckpt["train_params"][int(ckpt["case"][i])]) for i in sel]
say(f"== burgers3d: {len(states)} projected training states, M_acc={M_acc} M_fast={M_fast}")
N0 = meshes[0]
mm0 = MeshModel3(ckpt, pde, N0, M_acc); say(f"  offline N={N0}: {mm0.offline_time:.1f}s")

def rho_over_states(ev, ev_ref):
    nl, nlr = jax.jit(ev["nl"]), jax.jit(ev_ref["nl"]); out = []
    for c, ps in zip(states, state_p):
        a, b = np.asarray(nl(jnp.asarray(c), ps)), np.asarray(nlr(jnp.asarray(c), ps))
        out.append((np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300), np.linalg.norm(a[:M_fast] - b[:M_fast]) / max(np.linalg.norm(b[:M_fast]), 1e-300)))
    return np.array(out)

def time_eval(rp, n_rep=10):
    zy = jnp.concatenate([jnp.asarray(ckpt["Z"][0]), jnp.zeros(rp.q)]); c_n = rp.coeffs(zy); b = jnp.zeros(rp.mm.M); ps = state_p[0]
    p0 = evals[0]["p"]; nu, r = float(pde.nu(p0)), float(pde.react(p0))
    f = lambda: (rp.residual_jit(zy, c_n, b, nu, r, pde.dt, ps), rp.jac_jit(zy, c_n, b, nu, r, pde.dt, ps))
    out = f(); jax.block_until_ready(out); t = time.perf_counter()
    for _ in range(n_rep): out = f()
    jax.block_until_ready(out); return (time.perf_counter() - t) / n_rep * 1e3

def add(target, N, rule, par, m, form, r, extra=None):
    row = dict(pde="burgers3d", target=target, N=N, rule=rule, param=par, m=int(m), form=form,
               rho_acc_med=float(np.median(r[:, 0])), rho_acc_max=float(r[:, 0].max()), rho_acc_p90=float(np.quantile(r[:, 0], .9)),
               rho_fast_med=float(np.median(r[:, 1])), rho_fast_max=float(r[:, 1].max()))
    if extra: row.update(extra)
    rows.append(row); return row

offmesh = []
for p_ in [6, 8, 12, 16, 20, 24, 32]: offmesh.append(("gauss_tensor", p_, gauss_tensor3(p_)))
for lv in range(3, 11): offmesh.append(("smolyak_cc", lv, smolyak3(lv)))
for e in range(10, 16):
    offmesh.append(("sobol", 2**e, sobol3(2**e))); offmesh.append(("mc", 2**e, mc3(2**e)))
    offmesh.append(("kuo_lattice", 2**e, kuo_lattice3(2**e))); offmesh.append(("kuo_lattice_tent", 2**e, kuo_lattice3(2**e, tent=True)))
for n in [4093, 16381]:
    z, P = korobov_search3(n); say(f"  korobov n={n}: z={z.tolist()} P2={P:.3e}"); offmesh.append(("korobov", n, rank1_lattice3(n, z)))
say("  building continuum reference (Gauss 64^3) ...")
ref = make_offmesh_evaluator3(mm0, *gauss_tensor3(64), "ref64"); chk = make_offmesh_evaluator3(mm0, *gauss_tensor3(48), "ref48")
r = rho_over_states(chk, ref); say(f"  reference check GL48 vs GL64: rho max {r[:,0].max():.2e}")
chk2 = make_offmesh_evaluator3(mm0, *gauss_tensor3(48), "ref48f", form="flux"); r = rho_over_states(chk2, ref); say(f"  flux vs point (GL48 vs GL64 point): rho max {r[:,0].max():.2e}")
say("== off-mesh rules vs continuum reference")
for form in ["point", "flux"]:
    for name, par, (X, w) in offmesh:
        if form == "flux" and name not in ("gauss_tensor", "sobol", "kuo_lattice", "smolyak_cc"): continue
        ev = make_offmesh_evaluator3(mm0, X, w, f"{name}:{par}", form=form); r = rho_over_states(ev, ref)
        row = add("continuum", 0, name, par, len(w), form, r, dict(wmin=float(w.min()), wsum=float(w.sum())))
        say(f"  [{form}] {name:17s} {str(par):6s} m={len(w):7d}  rho(M={M_acc}) med {row['rho_acc_med']:.2e} max {row['rho_acc_max']:.2e} | rho(M={M_fast}) med {row['rho_fast_med']:.2e} max {row['rho_fast_max']:.2e}")
    json.dump(rows, open(f"{args.out}/burgers3d_quad_study.json", "w"), indent=1)
del chk, chk2
fit_codes = ckpt["codes"][np.random.default_rng(1).choice(len(ckpt["codes"]), args.eq_nfit, replace=False)]
for N in meshes:
    say(f"== mesh N={N}")
    mm = mm0 if N == N0 else MeshModel3(ckpt, pde, N, M_acc)
    if N != N0: say(f"  offline N={N}: {mm.offline_time:.1f}s")
    dense = make_dense_evaluator3(mm)
    ref_here = ref if N == N0 else make_offmesh_evaluator3(mm, *gauss_tensor3(64), "ref64")
    gap = rho_over_states(ref_here, dense)
    row = add("mesh", N, "continuum_gap", 64, 262144, "point", gap); say(f"  mesh-vs-continuum gap: med {row['rho_acc_med']:.2e} max {row['rho_acc_max']:.2e}")
    for q in [0, args.q_acc]:
        rp = ReducedProblem(mm, q, dense); ms = time_eval(rp, n_rep=3 if N >= 128 else 10)
        rows.append(dict(pde="burgers3d", target="time", N=N, rule="dense", param=0, m=(N - 1) ** 3, form="point", q=q, ms_eval=ms)); say(f"  dense q={q}: {ms:.2f} ms per residual+Jacobian")
    for stride in [N // 8, N // 16, N // 32]:
        if stride < 1: continue
        idx, w = mesh_lattice3(N, stride); ev = make_mesh_evaluator3(mm, idx, w, f"lattice:{stride}"); r = rho_over_states(ev, dense)
        row = add("mesh", N, "mesh_lattice", stride, len(idx), "point", r); say(f"  mesh_lattice stride {stride} m={len(idx)}: rho max {row['rho_acc_max']:.2e} | fast max {row['rho_fast_max']:.2e}")
    n_all = (N - 1) ** 3
    cand = np.sort(np.random.default_rng(2).choice(n_all, min(args.eq_cand, n_all), replace=False))
    nl_d = jax.jit(dense["nl"]); ps0 = state_p[0]; n1 = N - 1
    i_ = cand // (n1 * n1) + 1; j_ = (cand // n1) % n1 + 1; k_ = cand % n1 + 1
    def gather(ii, jj, kk):
        inside = (ii >= 1) & (ii <= n1) & (jj >= 1) & (jj <= n1) & (kk >= 1) & (kk <= n1)
        lin = np.clip(((ii - 1) * n1 + (jj - 1)) * n1 + (kk - 1), 0, n1**3 - 1); return mm.G[lin] * inside[:, None]
    block = np.stack([gather(i_, j_, k_), gather(i_ - 1, j_, k_), gather(i_ + 1, j_, k_), gather(i_, j_ - 1, k_), gather(i_, j_ + 1, k_), gather(i_, j_, k_ - 1), gather(i_, j_, k_ + 1)], 1)
    a_, b_, c_ = mm.modes[:, 0], mm.modes[:, 1], mm.modes[:, 2]
    psi = (2.0 / N) ** 1.5 * np.sin(a_[None] * np.pi * i_[:, None] / N) * np.sin(b_[None] * np.pi * j_[:, None] / N) * np.sin(c_[None] * np.pi * k_[:, None] / N)
    stencil = jax.jit(lambda U: pde.nonlinear_stencil(U[:, 0], U[:, 1], U[:, 2], U[:, 3], U[:, 4], U[:, 5], U[:, 6], N, ps0))
    A_rows, beta_rows = [], []
    for c in fit_codes:
        Fi = np.asarray(stencil(jnp.asarray(block @ c))); beta = np.asarray(nl_d(jnp.asarray(c), ps0)); nb = np.linalg.norm(beta)
        A_rows.append((psi * Fi[:, None]).T / nb); beta_rows.append(beta / nb)
    A = np.concatenate(A_rows, 0); beta = np.concatenate(beta_rows); say(f"  EQ fit matrix {A.shape}")
    for m in [1024, 2048]:
        t0 = time.time(); w, supp = fit_eq_nnls(A, beta, m); sel_ = np.flatnonzero(w > 0)
        ev = make_mesh_evaluator3(mm, cand[sel_], w[sel_], f"eq:{m}"); r = rho_over_states(ev, dense)
        row = add("mesh", N, "eq_nnls", m, supp, "point", r, dict(fit_res=float(np.linalg.norm(A @ w - beta) / np.linalg.norm(beta)), fit_time=time.time() - t0))
        say(f"  eq_nnls m={m} support {supp} ({time.time()-t0:.0f}s): rho max {row['rho_acc_max']:.2e} | fast max {row['rho_fast_max']:.2e}")
        np.savez(f"{args.out}/burgers3d_eq_N{N}_m{m}.npz", idx=cand[sel_], w=w[sel_])
    for name, par, (X, w) in offmesh:
        if name in ("mc", "kuo_lattice_tent"): continue
        ev = make_offmesh_evaluator3(mm, X, w, f"{name}:{par}"); r = rho_over_states(ev, dense); add("mesh", N, name, par, len(w), "point", r)
    for name, par, (X, w) in [("gauss_tensor", 16, gauss_tensor3(16)), ("gauss_tensor", 24, gauss_tensor3(24)), ("smolyak_cc", 8, smolyak3(8)),
                              ("sobol", 4096, sobol3(4096)), ("sobol", 32768, sobol3(32768)), ("kuo_lattice", 4096, kuo_lattice3(4096)), ("kuo_lattice", 32768, kuo_lattice3(32768))]:
        for q in [0, args.q_acc]:
            rp = ReducedProblem(mm, q, make_offmesh_evaluator3(mm, X, w, f"{name}:{par}")); ms = time_eval(rp)
            rows.append(dict(pde="burgers3d", target="time", N=N, rule=name, param=par, m=len(w), form="point", q=q, ms_eval=ms))
            say(f"  time {name}:{par} m={len(w)} q={q}: {ms:.2f} ms per residual+Jacobian")
    json.dump(rows, open(f"{args.out}/burgers3d_quad_study.json", "w"), indent=1)
say("done")
