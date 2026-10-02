"""Quadrature-error study: relative error rho of the tested nonlinear term for every
rule against (a) the continuum reference (tensor Gauss-Legendre 300x300 on the same
decoded state) and (b) the dense mesh evaluation on each mesh N, on states reached by
the dense reduced solver.  Also the mesh-vs-continuum gap and evaluation timing.

usage: python scripts/quad_study.py burgers [--meshes 128,256,512,1024] [--q_acc 64]
"""
import argparse, sys, os, time, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom, jax, jax.numpy as jnp
from nmrom.pdes import PDES
from nmrom import quadrature as Q
from nmrom.rom import (MeshModel, ReducedProblem, make_dense_evaluator, make_mesh_evaluator,
                       make_offmesh_evaluator, rollout_errors, rho_study, scalar_params)

ap = argparse.ArgumentParser()
ap.add_argument("pde"); ap.add_argument("--meshes", default="128,256,512,1024")
ap.add_argument("--q_acc", type=int, default=64); ap.add_argument("--N_states", type=int, default=256)
ap.add_argument("--n_states", type=int, default=96); ap.add_argument("--data", default="data")
ap.add_argument("--out", default="results"); ap.add_argument("--eq_nfit", type=int, default=24)
ap.add_argument("--eq_cand", type=int, default=8192)
ap.add_argument("--states", default="reached", choices=["reached", "train"])
ap.add_argument("--tag", default="")
ap.add_argument("--model_suffix", default="")
ap.add_argument("--skip_eq", action="store_true")
ap.add_argument("--skip_time", action="store_true")
args = ap.parse_args()
pde = PDES[args.pde]
meshes = [int(m) for m in args.meshes.split(",")]
ckpt = pickle.load(open(f"{args.data}/{args.pde}_model{args.model_suffix}.pkl", "rb"))
ckpt["train_params"] = pickle.load(open(f"{args.data}/{args.pde}_train_params.pkl", "rb"))
evals = pickle.load(open(f"{args.data}/{args.pde}_eval.pkl", "rb")) if args.states == "reached" else None
train_params = ckpt["train_params"]
k, R = ckpt["k"], ckpt["R"]
M_acc, M_fast = 4 * (k + args.q_acc), 4 * k
os.makedirs(args.out, exist_ok=True)
rows = []
log = open(f"{args.out}/{args.pde}_quad_study{args.tag}.log", "w")
def say(*a):
    print(*a, flush=True); print(*a, file=log, flush=True)

# ---------------------------------------------------------------- reached states
say(f"== {args.pde}: reached states from dense reduced rollouts at N={args.N_states}")
mm0 = MeshModel(ckpt, pde, args.N_states, M_acc)
states, state_p = [], []
for q, M in ([(0, M_fast), (args.q_acc, M_acc)] if args.states == "reached" else []):
    mmq = MeshModel(ckpt, pde, args.N_states, M) if M != M_acc else mm0
    rp = ReducedProblem(mmq, q, make_dense_evaluator(mmq))
    for ci, case in enumerate(evals):
        roll = rp.rollout(case)
        err = rollout_errors(mmq, roll, case)
        say(f"  dense q={q} M={M} case {ci}: err same-grid {err['err_same_grid'].max()*100:.3f}% "
            f"refined {err['err_refined'].max()*100:.3f}% (FOM {err['fom_err_refined'].max()*100:.3f}%) "
            f"its/step {np.mean(roll['its']):.1f} exits {np.bincount(roll['exits'], minlength=4).tolist()} wall {roll['wall']:.1f}s")
        for c in roll["coeffs"]:
            states.append(c); state_p.append(scalar_params(case["p"]))
if args.states == "train":
    say("  using projected training snapshots as states")
    for i, c in enumerate(ckpt["codes"]):
        states.append(c); state_p.append(scalar_params(train_params[int(ckpt["case"][i])]))
rng = np.random.default_rng(0)
sel = rng.choice(len(states), min(args.n_states, len(states)), replace=False)
states = [states[i] for i in sel]; state_p = [state_p[i] for i in sel]
say(f"  {len(states)} reached states kept")

def rho_over_states(ev, ev_ref):
    nl, nlr = jax.jit(ev["nl"]), jax.jit(ev_ref["nl"])
    out = []
    for c, ps in zip(states, state_p):
        a, b = np.asarray(nl(jnp.asarray(c), ps)), np.asarray(nlr(jnp.asarray(c), ps))
        out.append((np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300),
                    np.linalg.norm(a[:M_fast] - b[:M_fast]) / max(np.linalg.norm(b[:M_fast]), 1e-300)))
    return np.array(out)

def time_eval(rp, n_rep=20):
    """ms for one residual + Jacobian evaluation (the per-iteration cost of LM)."""
    zy = jnp.concatenate([jnp.asarray(ckpt["Z"][0]), jnp.zeros(rp.q)])
    c_n = rp.coeffs(zy); b = jnp.zeros(rp.mm.M); ps = state_p[0]
    p0 = evals[0]["p"] if evals is not None else train_params[0]
    nu, r = float(pde.nu(p0)), float(pde.react(p0))
    f = lambda: (rp.residual_jit(zy, c_n, b, nu, r, pde.dt, ps), rp.jac_jit(zy, c_n, b, nu, r, pde.dt, ps))
    out = f(); jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(n_rep):
        out = f()
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / n_rep * 1e3

# ------------------------------------------------------------- off-mesh rules
offmesh = []
for p_ in [6, 8, 12, 16, 24, 32, 48, 64]:
    offmesh.append(("gauss_tensor", p_, Q.gauss_tensor(p_)))
for lv in range(3, 11):
    offmesh.append(("smolyak_cc", lv, Q.smolyak(lv, "cc")))
for lv in range(3, 10):
    offmesh.append(("smolyak_gl_exp", lv, Q.smolyak(lv, "gl_exp")))
for lv in range(6, 40, 4):
    offmesh.append(("smolyak_gl_lin", lv, Q.smolyak(lv, "gl_lin")))
for m in [256, 512, 1024, 2048, 4096, 8192]:
    offmesh.append(("sobol", m, Q.sobol(m, True, 0)))
    offmesh.append(("halton", m, Q.halton(m, True, 0)))
    offmesh.append(("mc", m, Q.uniform_mc(m, 0)))
for kk in [12, 13, 14, 15, 16, 17, 18, 19, 20]:
    offmesh.append(("fibonacci", kk, Q.fibonacci_lattice(kk, seed=0)))
    offmesh.append(("fibonacci_tent", kk, Q.fibonacci_lattice(kk, tent=True, seed=0)))
forms = ["point"] + (["flux"] if pde.nonlinear_flux is not None else [])

# continuum reference and its self-consistency check
ref = make_offmesh_evaluator(mm0, *Q.gauss_tensor(300), label="ref300")
chk = make_offmesh_evaluator(mm0, *Q.gauss_tensor(200), label="ref200")
r_chk = rho_over_states(chk, ref)
say(f"  reference check: GL200 vs GL300 rho max {r_chk[:,0].max():.2e}")
if pde.nonlinear_flux is not None:
    chk2 = make_offmesh_evaluator(mm0, *Q.gauss_tensor(300), label="ref300flux", form="flux")
    r2 = rho_over_states(chk2, ref)
    say(f"  point vs flux form at GL300: rho max {r2[:,0].max():.2e}")

say("== off-mesh rules vs continuum reference (mesh-independent)")
for form in forms:
    for name, par, (X, w) in offmesh:
        ev = make_offmesh_evaluator(mm0, X, w, label=f"{name}:{par}", form=form)
        r = rho_over_states(ev, ref)
        rows.append(dict(pde=args.pde, target="continuum", N=0, rule=name, param=par, m=len(w), form=form,
                         rho_acc_med=float(np.median(r[:, 0])), rho_acc_max=float(r[:, 0].max()), rho_acc_p90=float(np.quantile(r[:, 0], 0.9)),
                         rho_fast_med=float(np.median(r[:, 1])), rho_fast_max=float(r[:, 1].max()),
                         wmin=float(w.min()), wsum=float(w.sum())))
        say(f"  [{form}] {name:16s} {str(par):5s} m={len(w):6d}  rho(M={M_acc}) med {np.median(r[:,0]):.2e} max {r[:,0].max():.2e} | rho(M={M_fast}) med {np.median(r[:,1]):.2e} max {r[:,1].max():.2e}")

json.dump(rows, open(f"{args.out}/{args.pde}_quad_study{args.tag}.json", "w"), indent=1)
# --------------------------------------------------------- mesh-dependent part
fit_codes = ckpt["codes"][np.random.default_rng(1).choice(len(ckpt["codes"]), args.eq_nfit, replace=False)]
for N in meshes:
    say(f"== mesh N={N}")
    mm = mm0 if N == args.N_states else MeshModel(ckpt, pde, N, M_acc)
    dense = make_dense_evaluator(mm)
    ref_here = ref if N == args.N_states else make_offmesh_evaluator(mm, *Q.gauss_tensor(300), label="ref300")
    gap = rho_over_states(ref_here, dense)
    say(f"  mesh-vs-continuum gap (continuum ref vs dense upwind at N): med {np.median(gap[:,0]):.2e} max {gap[:,0].max():.2e}")
    rows.append(dict(pde=args.pde, target="mesh", N=N, rule="continuum_gap", param=300, m=90000, form="point",
                     rho_acc_med=float(np.median(gap[:, 0])), rho_acc_max=float(gap[:, 0].max()), rho_acc_p90=float(np.quantile(gap[:, 0], .9)),
                     rho_fast_med=float(np.median(gap[:, 1])), rho_fast_max=float(gap[:, 1].max()), wmin=0.0, wsum=1.0))
    # timing of dense residual+Jacobian
    for q in [0, args.q_acc]:
        rp = ReducedProblem(mm, q, dense)
        rows.append(dict(pde=args.pde, target="time", N=N, rule="dense", param=0, m=(N - 1) ** 2, form="point", q=q,
                         ms_eval=time_eval(rp)))
        say(f"  dense q={q}: {rows[-1]['ms_eval']:.2f} ms per residual+Jacobian")
    # mesh lattice rules
    for stride in [N // 8, N // 16, N // 32, N // 64]:
        if stride < 1: continue
        idx, w = Q.mesh_lattice(N, stride)
        ev = make_mesh_evaluator(mm, idx, w, f"lattice:{stride}")
        r = rho_over_states(ev, dense)
        rows.append(dict(pde=args.pde, target="mesh", N=N, rule="mesh_lattice", param=stride, m=len(idx), form="point",
                         rho_acc_med=float(np.median(r[:, 0])), rho_acc_max=float(r[:, 0].max()), rho_acc_p90=float(np.quantile(r[:, 0], .9)),
                         rho_fast_med=float(np.median(r[:, 1])), rho_fast_max=float(r[:, 1].max()), wmin=float(w.min()), wsum=float(w.sum())))
        say(f"  mesh_lattice stride {stride} m={len(idx)}: rho(M={M_acc}) med {np.median(r[:,0]):.2e} max {r[:,0].max():.2e} | fast max {r[:,1].max():.2e}")
    # NNLS empirical quadrature (paper's baseline), fitted on training coefficient snapshots
    n_all = (N - 1) ** 2
    eq_ms = [] if args.skip_eq else [256, 512, 1024, 2048]
    cand = np.sort(np.random.default_rng(2).choice(n_all, min(args.eq_cand, n_all), replace=False))
    nl_d = jax.jit(dense["nl"])
    ps0 = state_p[0]
    # per-candidate contributions A[(l,ab), i] = psi_ab(i) F_i(c_l); target beta_l = dense tested term
    n1 = N - 1
    i_ = cand // n1 + 1; j_ = cand % n1 + 1
    def gather(ii, jj):
        inside = (ii >= 1) & (ii <= n1) & (jj >= 1) & (jj <= n1)
        lin = np.clip((ii - 1) * n1 + (jj - 1), 0, n1 * n1 - 1)
        return mm.G[lin] * inside[:, None]
    block = np.stack([gather(i_, j_), gather(i_ - 1, j_), gather(i_ + 1, j_), gather(i_, j_ - 1), gather(i_, j_ + 1)], 1)
    a_, b_ = mm.modes[:, 0], mm.modes[:, 1]
    psi = (2.0 / N) * np.sin(a_[None] * np.pi * i_[:, None] / N) * np.sin(b_[None] * np.pi * j_[:, None] / N)   # (m_cand, M)
    stencil = jax.jit(lambda U: pde.nonlinear_stencil(U[:, 0], U[:, 1], U[:, 2], U[:, 3], U[:, 4], N, ps0))
    A_rows, beta_rows = [], []
    for c in fit_codes:
        Fi = np.asarray(stencil(jnp.asarray(block @ c)))                  # (m_cand,)
        beta = np.asarray(nl_d(jnp.asarray(c), ps0))                      # (M,)
        nb = np.linalg.norm(beta)
        A_rows.append((psi * Fi[:, None]).T / nb); beta_rows.append(beta / nb)
    A = np.concatenate(A_rows, 0); beta = np.concatenate(beta_rows)
    say(f"  EQ fit matrix {A.shape}")
    for m in eq_ms:
        t0 = time.time()
        w, supp = Q.fit_eq_nnls(A, beta, m)
        sel_ = np.flatnonzero(w > 0)
        ev = make_mesh_evaluator(mm, cand[sel_], w[sel_], f"eq:{m}")
        r = rho_over_states(ev, dense)
        fit_res = np.linalg.norm(A @ w - beta) / np.linalg.norm(beta)
        rows.append(dict(pde=args.pde, target="mesh", N=N, rule="eq_nnls", param=m, m=int(supp), form="point",
                         rho_acc_med=float(np.median(r[:, 0])), rho_acc_max=float(r[:, 0].max()), rho_acc_p90=float(np.quantile(r[:, 0], .9)),
                         rho_fast_med=float(np.median(r[:, 1])), rho_fast_max=float(r[:, 1].max()), wmin=float(w[sel_].min()), wsum=float(w.sum()),
                         fit_res=float(fit_res), fit_time=time.time() - t0))
        say(f"  eq_nnls m={m} support {supp} (fit res {fit_res:.2e}, {time.time()-t0:.0f}s): rho(M={M_acc}) med {np.median(r[:,0]):.2e} max {r[:,0].max():.2e} | fast max {r[:,1].max():.2e}")
        np.savez(f"{args.out}/{args.pde}{args.model_suffix}_eq_N{N}_m{m}.npz", idx=cand[sel_], w=w[sel_])
    # off-mesh rules against the mesh target at this N (what the same-grid error sees)
    for name, par, (X, w) in offmesh:
        if name in ("mc", "halton") or (name == "smolyak_gl_lin"):
            continue
        ev = make_offmesh_evaluator(mm, X, w, label=f"{name}:{par}")
        r = rho_over_states(ev, dense)
        rows.append(dict(pde=args.pde, target="mesh", N=N, rule=name, param=par, m=len(w), form="point",
                         rho_acc_med=float(np.median(r[:, 0])), rho_acc_max=float(r[:, 0].max()), rho_acc_p90=float(np.quantile(r[:, 0], .9)),
                         rho_fast_med=float(np.median(r[:, 1])), rho_fast_max=float(r[:, 1].max()), wmin=float(w.min()), wsum=float(w.sum())))
    # timing of off-mesh / mesh rules at this N (should not depend on N)
    for name, par, (X, w) in ([] if args.skip_time else [("sobol", 1024, Q.sobol(1024)), ("sobol", 4096, Q.sobol(4096)),
                              ("smolyak_cc", 7, Q.smolyak(7, "cc")), ("gauss_tensor", 32, Q.gauss_tensor(32))]):
        for q in [0, args.q_acc]:
            rp = ReducedProblem(mm, q, make_offmesh_evaluator(mm, X, w, label=f"{name}:{par}"))
            rows.append(dict(pde=args.pde, target="time", N=N, rule=name, param=par, m=len(w), form="point", q=q, ms_eval=time_eval(rp)))
            say(f"  time {name}:{par} m={len(w)} q={q}: {rows[-1]['ms_eval']:.2f} ms per residual+Jacobian")
    json.dump(rows, open(f"{args.out}/{args.pde}_quad_study{args.tag}.json", "w"), indent=1)
say("done")
