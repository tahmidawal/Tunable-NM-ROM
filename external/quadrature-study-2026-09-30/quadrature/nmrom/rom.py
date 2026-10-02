"""Reduced problem: tested residual, Levenberg--Marquardt with block damping, rollout.

Reduced residual for one backward-Euler step (mesh-normalised sine tests):
    r(z,y) = [B0 (c - c_n) + dt ( Nhat(c) + (nu Lam - r) B0 c - b )] / (1 + dt (nu Lam - r)),
    c = h(z) + C_q y,
and for steady problems  r = [nu Lam B0 c + Nhat(c) - b] / (nu Lam).
Only Nhat(c) (the tested nonlinear term) depends on the quadrature rule.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from .grid import (mode_list, eigenvalues, interior_coords, mesh_test, mesh_test_batch,
                   continuum_tests)
from .bank import bank_eval_chunked
from .head import head

EXIT = {0: "budget", 1: "stationary", 2: "tiny_step", 3: "damping_limit"}


class MeshModel:
    """Offline objects of one frozen model on one mesh: bank on the mesh, tested bank B0."""

    def __init__(self, ckpt, pde, N, M):
        self.ckpt, self.pde, self.N, self.M = ckpt, pde, N, M
        self.R, self.k = ckpt["R"], ckpt["k"]
        self.modes = mode_list(M, N)
        self.lam = eigenvalues(self.modes, N)
        self.X = interior_coords(N)
        t0 = time.time()
        self.G = bank_eval_chunked(ckpt["bank"], self.X)              # (n, R)
        G3 = self.G.reshape(N - 1, N - 1, self.R)
        cols = []
        for i in range(0, self.R, 8):
            cols.append(np.asarray(mesh_test_batch(jnp.moveaxis(jnp.asarray(G3[..., i:i + 8]), -1, 0), self.modes)))
        self.B0 = np.concatenate(cols, 0).T                            # (M, R)
        self.gram = self.G.T @ self.G
        self.offline_time = time.time() - t0

    def project_field(self, u):
        """Least-squares bank coefficients of a mesh field (initial fit)."""
        return np.linalg.solve(self.gram, self.G.T @ np.asarray(u).ravel())

    def tested_field(self, u):
        return np.asarray(mesh_test(jnp.asarray(np.asarray(u).reshape(self.N - 1, self.N - 1)), self.modes))

    def reconstruct(self, c):
        return (self.G @ c).reshape(self.N - 1, self.N - 1)


def scalar_params(p):
    return {k: float(v) for k, v in p.items() if np.isscalar(v) or np.ndim(v) == 0}


# ------------------------------------------------------------ evaluators of Nhat
def make_dense_evaluator(mm):
    """Exact mesh evaluation: decode every node, apply the FOM stencil, sine transform. O(n R)."""
    pde, N, modes = mm.pde, mm.N, mm.modes
    G = jnp.asarray(mm.G)

    def nl(c, p):
        u = (G @ c).reshape(N - 1, N - 1)
        return mesh_test(pde.nonlinear_grid(u, N, p), modes)
    return dict(nl=nl, m=(N - 1) ** 2, kind="dense", label="dense")


def make_mesh_evaluator(mm, idx, w, label):
    """Mesh-sampled rule (EQ, lattice): cached (m, 5, R) stencil block, FOM stencil at the nodes."""
    pde, N, modes, R = mm.pde, mm.N, mm.modes, mm.R
    n1 = N - 1
    i = idx // n1 + 1
    j = idx % n1 + 1

    def gather(ii, jj):
        inside = (ii >= 1) & (ii <= n1) & (jj >= 1) & (jj <= n1)
        lin = np.clip((ii - 1) * n1 + (jj - 1), 0, n1 * n1 - 1)
        return mm.G[lin] * inside[:, None]
    block = np.stack([gather(i, j), gather(i - 1, j), gather(i + 1, j), gather(i, j - 1), gather(i, j + 1)], 1)
    a, b = modes[:, 0], modes[:, 1]
    psi = (2.0 / N) * np.sin(a[None] * np.pi * i[:, None] / N) * np.sin(b[None] * np.pi * j[:, None] / N)
    Psi = jnp.asarray(w[:, None] * psi)
    blk = jnp.asarray(block)

    def nl(c, p):
        U = blk @ c                                     # (m, 5)
        F = pde.nonlinear_stencil(U[:, 0], U[:, 1], U[:, 2], U[:, 3], U[:, 4], N, p)
        return Psi.T @ F
    return dict(nl=nl, m=len(idx), kind="mesh", label=label, idx=idx, w=w)


def make_offmesh_evaluator(mm, Xq, wq, label, form="point"):
    """Off-mesh rule: partial decoding of the coordinate-network bank at the m points.
    form='point': int psi F(u, grad u);  form='flux': -int grad psi . Flux(u) (no grad u needed)."""
    pde, N, modes = mm.pde, mm.N, mm.modes
    Gq, Gx, Gy = bank_eval_chunked(mm.ckpt["bank"], Xq, grad=True)
    psi, psix, psiy = continuum_tests(Xq, modes)
    scale = N * wq[:, None]
    Gq, Gx, Gy = jnp.asarray(Gq), jnp.asarray(Gx), jnp.asarray(Gy)
    Psi, Psix, Psiy = jnp.asarray(scale * psi), jnp.asarray(scale * psix), jnp.asarray(scale * psiy)
    if form == "flux":
        assert pde.nonlinear_flux is not None
        def nl(c, p):
            u = Gq @ c
            F1, F2 = pde.nonlinear_flux(u, p)
            return -(Psix.T @ F1 + Psiy.T @ F2)
    else:
        if pde.needs_grad:
            def nl(c, p):
                u, ux, uy = Gq @ c, Gx @ c, Gy @ c
                return Psi.T @ pde.nonlinear_point(u, ux, uy, p)
        else:
            def nl(c, p):
                u = Gq @ c
                return Psi.T @ pde.nonlinear_point(u, None, None, p)
    return dict(nl=nl, m=len(wq), kind="offmesh", label=label, X=Xq, w=wq, form=form)


# ------------------------------------------------------------ reduced problem
class ReducedProblem:
    def __init__(self, mm, q, evaluator, eta_tol=1e-5, max_it=100, lam0=1e-6, ridge=1e-8):
        self.mm, self.q, self.ev = mm, q, evaluator
        self.k, self.R = mm.k, mm.R
        self.C = jnp.asarray(mm.ckpt["C"][:, :q])
        self.hp = {kk: jnp.asarray(v) for kk, v in mm.ckpt["head"].items()}
        self.B0 = jnp.asarray(mm.B0); self.lam = jnp.asarray(mm.lam)
        self.steady = mm.pde.steady
        self.eta_tol, self.max_it, self.lam0, self.ridge = eta_tol, max_it, lam0, ridge
        Z = mm.ckpt["Z"]
        self.trust = 3.0 * float(np.sqrt(np.mean(np.sum(Z**2, 1))))
        nl = evaluator["nl"]
        k = self.k

        def coeffs(zy):
            return head(self.hp, zy[:k]) + self.C @ zy[k:]
        self.coeffs = jax.jit(coeffs)

        def residual(zy, c_n, b, nu, r, dt, p):
            c = coeffs(zy)
            B0c = self.B0 @ c
            if self.steady:
                return (nu * self.lam * B0c + nl(c, p) - b) / (nu * self.lam)
            d = nu * self.lam - r
            return (B0c - self.B0 @ c_n + dt * (nl(c, p) + d * B0c - b)) / (1.0 + dt * d)
        self.residual = residual
        self.residual_jit = jax.jit(residual)
        self.jac_jit = jax.jit(lambda zy, *a: jax.jacfwd(lambda v: residual(v, *a))(zy))

        def fit_residual(zy, c_target):
            return coeffs(zy) - c_target
        self.lm_step = jax.jit(self._make_lm(residual, k))
        self.lm_fit = jax.jit(self._make_lm(fit_residual, k))
        self.nl_jit = jax.jit(nl)

    def _make_lm(self, resfn, k):
        eta_tol, max_it, ridge, trust = self.eta_tol, self.max_it, self.ridge, self.trust

        def solve(zy0, lam_init, *args):
            def jac(zy):
                return jax.jacfwd(lambda v: resfn(v, *args))(zy)
            nq = zy0.shape[0] - k

            def eta_of(J, r, nr):
                return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * nr + 1e-300)

            def body(st):
                zy, r, J, nr, lam, it, ex, nacc = st
                H = J.T @ J
                g = J.T @ r
                dg = jnp.diag(H)
                damp = jnp.concatenate([lam * dg[:k] + 1e-300, ridge * jnp.ones(nq)])
                delta = -jnp.linalg.solve(H + jnp.diag(damp), g)
                dz = delta[:k]
                ndz = jnp.linalg.norm(dz)
                delta = delta.at[:k].set(dz * jnp.minimum(1.0, trust / (ndz + 1e-300)))
                zt = zy + delta
                rt = resfn(zt, *args)
                nrt = jnp.linalg.norm(rt)
                acc = nrt < nr
                zy_n = jnp.where(acc, zt, zy)
                r_n = jnp.where(acc, rt, r)
                nr_n = jnp.where(acc, nrt, nr)
                J_n = jax.lax.cond(acc, lambda: jac(zt), lambda: J)
                lam_n = jnp.where(acc, jnp.maximum(lam / 3.0, 1e-12), jnp.minimum(10.0 * lam, 1e14))
                ex = jnp.where(eta_of(J_n, r_n, nr_n) <= eta_tol, 1, 0)
                ex = jnp.where(acc & (jnp.linalg.norm(delta) <= 1e-12 * (1.0 + jnp.linalg.norm(zy))), 2, ex)
                ex = jnp.where((~acc) & (lam >= 1e14), 3, ex)
                return zy_n, r_n, J_n, nr_n, lam_n, it + 1, ex, nacc + acc.astype(jnp.int32)

            def cond(st):
                return (st[6] == 0) & (st[5] < max_it)

            r0 = resfn(zy0, *args)
            J0 = jac(zy0)
            nr0 = jnp.linalg.norm(r0)
            ex0 = jnp.where(eta_of(J0, r0, nr0) <= eta_tol, 1, 0)
            st = (zy0, r0, J0, nr0, lam_init, jnp.int32(0), ex0, jnp.int32(0))
            zy, r, J, nr, lam, it, ex, nacc = jax.lax.while_loop(cond, body, st)
            return zy, nr, it, nacc, ex, lam, nr0
        return solve

    # ---------------------------------------------------------------- queries
    def initial_state(self, u0_mesh):
        """Nearest stored training code, then LM fit of (z, y) to the projected initial field."""
        mm = self.mm
        c0 = mm.project_field(u0_mesh)
        ck = mm.ckpt
        t0_idx = np.flatnonzero(ck["t"] == 0.0)
        Zs = ck["Z"][t0_idx]
        preds = np.asarray(jax.vmap(lambda z: head(self.hp, z))(jnp.asarray(Zs)))
        s = np.argmin(np.linalg.norm(preds - c0[None], axis=1))
        zy0 = jnp.concatenate([jnp.asarray(Zs[s]), jnp.zeros(self.q)])
        zy, nr, it, nacc, ex, _, nr0 = self.lm_fit(zy0, jnp.float64(self.lam0), jnp.asarray(c0))
        return zy, dict(fit_res=float(nr), fit_its=int(it), fit_exit=int(ex), c0=c0)

    def rollout(self, case, dt=None, verbose=False):
        """Solve one evaluation case; returns coefficients per step, outputs and stats."""
        mm, pde = self.mm, self.mm.pde
        p = case["p"]
        ps = scalar_params(p)
        nu, r = float(pde.nu(p)), float(pde.react(p))
        X = mm.X
        b = jnp.asarray(mm.tested_field(pde.source(X, p)))
        t_wall = time.time()
        if pde.steady:
            # start from the stored code of the training case nearest in (normalised) parameter space
            tp = mm.ckpt["train_params"]
            keys = sorted(k_ for k_ in tp[0] if np.isscalar(tp[0][k_]))
            P = np.array([[float(t[k_]) for k_ in keys] + list(np.ravel(t["c"])) for t in tp])
            pv = np.array([float(p[k_]) for k_ in keys] + list(np.ravel(p["c"])))
            sd = P.std(0).clip(1e-12)
            s = int(np.argmin(np.linalg.norm((P - pv) / sd, axis=1)))
            snap = int(np.flatnonzero(mm.ckpt["case"] == s)[0])
            zy0 = jnp.concatenate([jnp.asarray(mm.ckpt["Z"][snap]), jnp.zeros(self.q)])
            zy, nr, it, nacc, ex, _, nr0 = self.lm_step(zy0, jnp.float64(self.lam0), jnp.zeros(self.R), b, nu, r, 1.0, ps)
            zy.block_until_ready()
            c = np.asarray(self.coeffs(zy))
            return dict(coeffs=[c], its=[int(it)], acc=[int(nacc)], exits=[int(ex)], out_coeffs=[c],
                        t_out=np.array([0.0]), wall=time.time() - t_wall, init=dict(fit_its=0))
        dt = pde.dt if dt is None else dt
        nsteps = int(round(pde.T / dt))
        out_steps = {int(round(t / dt)): t for t in pde.out_times}
        u0 = pde.u0(X, p)
        zy, info = self.initial_state(u0)
        lam = jnp.float64(self.lam0)
        zy_prev = None
        coeffs, its, accs, exits, out_c, t_out = [np.asarray(self.coeffs(zy))], [], [], [], [], []
        for n in range(1, nsteps + 1):
            c_n = self.coeffs(zy)
            zy_start = zy
            if zy_prev is not None:
                zy_ext = 2 * zy - zy_prev
                r_a = jnp.linalg.norm(self.residual_jit(zy, c_n, b, nu, r, dt, ps))
                r_e = jnp.linalg.norm(self.residual_jit(zy_ext, c_n, b, nu, r, dt, ps))
                zy_start = jnp.where(r_e < r_a, zy_ext, zy)
            zy_new, nr, it, nacc, ex, lam, nr0 = self.lm_step(zy_start, lam, c_n, b, nu, r, dt, ps)
            zy_prev, zy = zy, zy_new
            its.append(int(it)); accs.append(int(nacc)); exits.append(int(ex))
            c = np.asarray(self.coeffs(zy))
            coeffs.append(c)
            if n in out_steps:
                out_c.append(c); t_out.append(out_steps[n])
            if verbose and n % 10 == 0:
                print(f"    step {n} its {it} acc {nacc} exit {EXIT[int(ex)]} |r| {float(nr):.2e}", flush=True)
        wall = time.time() - t_wall
        return dict(coeffs=coeffs, its=its, acc=accs, exits=exits, out_coeffs=out_c, t_out=np.array(t_out),
                    wall=wall, init=info)


def rollout_errors(mm, roll, case, N_ref_sub=True):
    """Relative L2 errors of the reconstructed outputs against the same-grid FOM and the
    refined reference (subsampled to this mesh); normalised by the initial-field norm
    (steady: by the reference norm)."""
    N = mm.N
    sg = case["same_grid"][N]["u_out"]
    ref = case["refined"]["u_out"]
    Nr = case["refined"]["N"]
    s = Nr // N
    ref_sub = ref[:, s - 1::s, s - 1::s] if s > 1 else ref
    U = np.stack([mm.reconstruct(c) for c in roll["out_coeffs"]])
    if mm.pde.steady:
        norm = np.linalg.norm(sg[0])
    else:
        norm = np.linalg.norm(mm.pde.u0(mm.X, case["p"]))
    e_sg = np.linalg.norm((U - sg).reshape(len(U), -1), axis=1) / norm
    e_ref = np.linalg.norm((U - ref_sub).reshape(len(U), -1), axis=1) / norm
    e_fom_ref = np.linalg.norm((sg - ref_sub).reshape(len(U), -1), axis=1) / norm
    return dict(err_same_grid=e_sg, err_refined=e_ref, fom_err_refined=e_fom_ref)


def rho_study(ev, ev_ref, states, ps):
    """Relative error of the tested nonlinear term of a rule against a reference evaluator."""
    nl, nlr = jax.jit(ev["nl"]), jax.jit(ev_ref["nl"])
    rhos = []
    for c in states:
        c = jnp.asarray(c)
        a, b = np.asarray(nl(c, ps)), np.asarray(nlr(c, ps))
        rhos.append(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
    return np.array(rhos)
