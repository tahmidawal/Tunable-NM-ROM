"""PHASE 1 driver: the staggered MAC steady-Stokes FOM and its correctness gates.

Revision 4 (2026-08-30), rewritten against `STOKES-PHASE1-VERIFY-codex.md`
(Codex gpt-5.6-sol), which confirmed the operators, the solver and the
closed-form claim, and required three additions:

  A. a SECOND, GENERIC manufactured solution, because the frozen one is
     degenerate (its discrete error is a uniform scalar amplitude factor);
  B. a REPAIRED S3 control -- deterministic aligned pressures p = chi_kl
     instead of grid-white random noise;
  C. every diagnostic turned into a real ASSERTION with a real threshold.

Revision 4 closes STOKES-PHASE1C-VERIFY-codex.md: S-BACKERR gains blockwise
residuals and full 17-solve coverage, PRECOND gains a frozen-config equality
check, an expected-gate manifest with exact row counts, a non-finite sweep, and
the rule that a SMOKE run never sets complete=true; and the tautological
sqrt(M)*control-metric assertion is replaced by an independently constructed
control.

Every result is written as a NUMBER, never a boolean, into one JSON.

  S0        backend / f64 / matmul precision -- now ASSERTED, not just recorded.
  gate REF  the operators reproduce the archived auditor reference
            (STOKES-AUDIT-mac_check.py) entry-for-entry at small N.
  gate MF   the sparse matrices and the INDEPENDENT pad-and-slice matrix-free
            implementations agree on random inputs.
  gate SYM  ||L - L^T||_max (the vector Laplacian must be symmetric).
  gate MMSF each manufactured family's ANALYTIC Laplacian, pressure gradient,
            divergence and wall trace checked against high-accuracy finite
            differences of its own u, p.  Guards the hand-derived algebra.
  S-ADJ     ||M_u Grad + D^T M_p|| / (||M_u Grad|| + ||D^T M_p||) <= 1e-14,
            and the test-projected defect normalized the same way <= 1e-14.
            A sign-flipped-v-block NEGATIVE CONTROL must be >= 1e-2.
  S-STRUCT  ||D + Grad^T||_inf, ||D C||_inf, ||D C||_max: all exactly 0.
  S-RANK    rank D = N^2-1, dim ker D = rank C = (N-1)^2 by dense SVD at the
            RANK_NS meshes, and by an exact sparse-LU witness at every mesh.
  S-PRESS   REPAIRED S3.  Deterministic p = chi_kl, gated against an
            INDEPENDENTLY constructed control (the analytic gradient of the
            same chi, sampled on the face lattices, never touching Grad):
            matched cosine >= 0.99, control Frobenius >= 1e-2, solenoidal
            projection and cosine <= 1e-13.  The self-normalised control and
            the random-pressure numbers are labelled DIAGNOSTICS ONLY.
  S-FOM     the frozen manufactured solution on N = 32,64,128,256 (8 and 16
            added because the audit tabulated anchors there too).  Order
            2.00 +/- 0.05 in BOTH variables, AND agreement with the audit
            anchors.
  S-EXACT   the closed-form discrete solution.  The GATE is field agreement
            <= 1e-8.  pred_dev is RECORDED ONLY; two attempts to gate it were
            wrong and are retracted (STOKES-NOTES.md retractions 6-7).
  S-BACKERR global normalised backward error AND the three blockwise residuals
            (momentum, continuity, mean-zero gauge) over ALL 17 solves.
  MANIFEST  every expected gate present, exact expected row counts, and no
            non-finite float anywhere in the report.
  S-FOMGEN  the GENERIC manufactured solution on N = 32,64,128.  Order
            2.00 +/- 0.05, agreement with the verifier's tabulated values, and
            error/solution cosine < 0.99 -- i.e. it must NOT be degenerate.
  S-NU      exact 1/nu velocity scaling at fixed f.
  S-FREESLIP the same ladder with EVEN tangential ghosts: the deliberate wrong
            answer, ASSERTED to fail S-FOM (O(1) error, order <= 0.5).

Env: NS="8,16,32,64,128,256", LADDER="32,64,128,256" (frozen order gate),
     GEN_NS="32,64,128" (generic MMS ladder), ADJ_NS (default = NS),
     RANK_NS="32,64", FREESLIP_NS="8,16,32,64,128", M_MODES=64, NU=1.0,
     SEED=20260830, OUT_TAG, OUT_PREFIX, ALLOW_CPU=0.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import stk2d_common as stk

HERE = os.path.dirname(os.path.abspath(__file__))

NS = [int(v) for v in os.environ.get("NS", "8,16,32,64,128,256").split(",")]
LADDER = [int(v) for v in os.environ.get("LADDER", "32,64,128,256").split(",")]
GEN_NS = [int(v) for v in os.environ.get("GEN_NS", "32,64,128").split(",") if v]
ADJ_NS = [int(v) for v in os.environ.get("ADJ_NS", ",".join(map(str, NS))).split(",")]
RANK_NS = [int(v) for v in os.environ.get("RANK_NS", "32,64").split(",") if v]
FREESLIP_NS = [int(v) for v in
               os.environ.get("FREESLIP_NS", "8,16,32,64,128").split(",") if v]
M_MODES = int(os.environ.get("M_MODES", "64"))
NU = float(os.environ.get("NU", "1.0"))
SEED = int(os.environ.get("SEED", "20260830"))
OUT_TAG = os.environ.get("OUT_TAG", "")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "runs/stk2d/")
ALLOW_CPU = int(os.environ.get("ALLOW_CPU", "0"))
SMOKE = int(os.environ.get("SMOKE", "0"))   # relaxes PRECOND only; recorded

ORDER_TARGET, ORDER_BAND = 2.00, 0.05
ADJ_TOL = 1e-14
NEG_CTL_FLOOR = 1e-2          # S-ADJ negative control must be this large
S3_FLOOR = 1e-2               # STOKES-DESIGN.md S3 control floor (kept)
S3_MATCHED_COS = 0.99         # verifier's repaired-control requirement
S3_SOL_TOL = 1e-13            # verifier's solenoidal requirement
EXACT_FIELD_TOL = 1e-8
BACKERR_TOL = 1e-13           # frozen engineering threshold (~450 eps)
CONT_TOL = 1e-12              # blockwise continuity residual
GAUGE_TOL = 1e-12             # |1^T p| / (sqrt(n_p) ||p||)
GAUGE_RAW_TOL = 1e-8          # raw |1^T p|, the verifier's requested form
# (a sqrt(M)*metric floor was gated in rev 3; retracted as tautological)
MMSF_TOL = 1e-6               # analytic-vs-FD forcing consistency
ANCHOR_TOL = 1e-3             # anchors are quoted to 4 significant figures
GEN_ANCHOR_TOL = 1e-5         # verifier quoted 7 significant figures
FREESLIP_ERR_FLOOR = 0.5      # the wrong BVP must be O(1) wrong
FREESLIP_ORDER_CEIL = 0.5     # ... and must not converge

# ---- the frozen contract, and the manifest a certified run must satisfy ----
FROZEN_CONFIG = dict(ns=[8, 16, 32, 64, 128, 256], ladder=[32, 64, 128, 256],
                     generic_ns=[32, 64, 128],
                     adj_ns=[8, 16, 32, 64, 128, 256], rank_ns=[32, 64],
                     freeslip_ns=[8, 16, 32, 64, 128], M_modes=64, nu=1.0,
                     allow_cpu=0)
EXPECTED_GATES = frozenset((
    "PRECOND", "S0", "REF", "MMSF", "MF", "S_ADJ", "S_STRUCT", "S_PRESS",
    "SYM", "S_RANK", "S_FOM", "S_EXACT", "S_FOMGEN", "S_NU", "S_FREESLIP",
    "S_BACKERR"))
# Exact row counts implied by FROZEN_CONFIG.  "Non-empty" is not sufficient.
EXPECTED_ROWS = dict(REF=3, MMSF=2, MF=6, S_ADJ=6, S_STRUCT=6, S_PRESS=6,
                     SYM=6, S_FOM=6, S_EXACT=6, S_FOMGEN=3, S_FREESLIP=5,
                     S_RANK_dense=2, S_RANK_indirect=6, S_BACKERR=17)

# Anchors transcribed from the audits / verification, with their source.
ANCHORS_FROZEN = {8: (5.303e-2, 2.617e-2), 16: (1.295e-2, 6.455e-3),
                  32: (3.219e-3, 1.608e-3), 64: (8.036e-4, 4.017e-4)}
ANCHORS_GENERIC = {32: (1.541713e-2, 1.833939e-1),
                   64: (3.820960e-3, 4.577326e-2),
                   128: (9.531800e-4, 1.144216e-2)}


# Every call to stk.solve_stokes in this driver goes through track_solve, so
# S-BACKERR covers ALL of them.  Revision 3 covered only the 14 that produced
# report rows, silently excluding the S0 probe and the two S-NU solves
# (STOKES-PHASE1C-VERIFY-codex.md).
SOLVES = []


def track_solve(tag, g, f, **kw):
    u, p, info = stk.solve_stokes(g, f, **kw)
    SOLVES.append(dict(tag=tag, N=int(g.N),
                       ghost=kw.get("ghost", "odd"), nu=float(kw.get("nu", 1.0)),
                       backward_err=info["backward_err"],
                       mom_resid=info["mom_resid"],
                       cont_resid=info["cont_resid"],
                       gauge_resid=info["gauge_resid"],
                       gauge_raw=info["gauge_raw"],
                       lin_resid_rel=info["lin_resid_rel"],
                       K_fro=info["K_fro"], lam=info["lam"]))
    return u, p, info


def finite(label, xs):
    """Assert every value in an aggregate is finite BEFORE reducing it.

    Python's max([finite, nan]) returns the finite value, so a NaN from a
    failed solve could pass an aggregate assertion and turn a hard failure into
    a green run.  Every aggregate in this driver is funnelled through here.
    """
    a = np.asarray([float(x) for x in xs], dtype=float)
    bad = ~np.isfinite(a)
    assert not bad.any(), (f"non-finite value(s) in {label}: "
                           f"{a[bad].tolist()} (indices {np.nonzero(bad)[0].tolist()})")
    return a


def log(*a):
    print(*a, flush=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"],
                                       text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def jax_provenance():
    """The FOM is scipy/CPU f64 by design (phase 1 is a small sparse direct
    solve).  S0 nevertheless ASSERTS the JAX environment, because phase 2 runs
    in it and a silent x64=False or matmul!=highest there would invalidate
    everything downstream."""
    out = dict(imported=False)
    try:
        import jax
        dev = jax.devices()[0]
        out = dict(imported=True, backend=dev.platform,
                   device=str(dev), device_kind=getattr(dev, "device_kind", ""),
                   x64=bool(jax.config.jax_enable_x64),
                   matmul_precision=os.environ.get(
                       "JAX_DEFAULT_MATMUL_PRECISION"),
                   jax_version=jax.__version__)
    except Exception as e:                                    # pragma: no cover
        out["error"] = repr(e)
    return out


# ------------------------------------------------------------------ gates ----

def gate_ref(report):
    src = open(os.path.join(HERE, "STOKES-AUDIT-mac_check.py")).read()
    src = src.split("def report(")[0]
    ns = {}
    exec(src, ns)                                             # noqa: S102
    rows = []
    for N in (4, 8, 16):
        g = stk.MacGrid(N)
        D = stk.divergence_matrix(g).toarray()
        Gr = stk.gradient_matrix(g).toarray()
        L = stk.laplacian_matrix(g, "odd").toarray()
        C = stk.curl_matrix(g).toarray()
        _, _, _, _, _, Dr, Grr, Cr, Lr = ns["ops"](N)
        rows.append(dict(N=N,
                         D_maxdiff=float(np.abs(D - Dr).max()),
                         Grad_maxdiff=float(np.abs(Gr - Grr).max()),
                         L_maxdiff=float(np.abs(L - Lr).max()),
                         C_maxdiff=float(np.abs(C - Cr).max())))
        log(f"  gate REF N={N:3d}: D {rows[-1]['D_maxdiff']:.1e}  "
            f"Grad {rows[-1]['Grad_maxdiff']:.1e}  "
            f"L {rows[-1]['L_maxdiff']:.1e}  C {rows[-1]['C_maxdiff']:.1e}")
    worst = float(finite("REF", [r[k] for r in rows for k in r
                                 if k != "N"]).max())
    report["gates"]["REF"] = dict(
        rows=rows, worst_maxdiff=float(worst), tol=0.0,
        rule="operators identical (exact 0) to STOKES-AUDIT-mac_check.py ops()")
    assert worst == 0.0, f"gate REF failed: worst entrywise diff {worst}"


def gate_mf(N, rng):
    g = stk.MacGrid(N)
    D = stk.divergence_matrix(g)
    Gr = stk.gradient_matrix(g)
    Lo = stk.laplacian_matrix(g, "odd")
    Le = stk.laplacian_matrix(g, "even")
    C = stk.curl_matrix(g)
    u = rng.standard_normal(g.n_u)
    p = rng.standard_normal(g.n_p)
    s = rng.standard_normal(g.n_psi)
    U, V = g.unpack(u)
    P = p.reshape(g.shape_p)
    S = s.reshape(g.shape_psi)

    def r(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))

    out = dict(
        N=N,
        L_odd=r(g.pack(*stk.apply_laplacian(g, U, V, "odd")), Lo @ u),
        L_even=r(g.pack(*stk.apply_laplacian(g, U, V, "even")), Le @ u),
        D=r(stk.apply_divergence(g, U, V).ravel(), D @ u),
        Grad=r(g.pack(*stk.apply_gradient(g, P)), Gr @ p),
        C=r(g.pack(*stk.apply_curl(g, S)), C @ s))
    out["worst"] = float(finite(f"MF N={N}",
                                [v for k, v in out.items() if k != "N"]).max())
    return out


def adjoint_gates(N, rng):
    g = stk.MacGrid(N)
    D = stk.divergence_matrix(g)
    Gr = stk.gradient_matrix(g)
    Mu = stk.mass_u(g)
    Mp = stk.mass_p(g)
    A1 = (Mu @ Gr).tocsr()
    A2 = (D.T @ Mp).tocsr()
    Dfr = (A1 + A2).tocsr()
    n1, n2 = stk.spnorm_fro(A1), stk.spnorm_fro(A2)
    primary = stk.spnorm_fro(Dfr) / (n1 + n2)

    Phi, lams, modes = stk.test_modes(g, min(M_MODES, g.n_psi))
    M = Phi.shape[1]
    T1 = Phi.T @ A1
    T2 = Phi.T @ A2
    fro = lambda X: float(np.linalg.norm(np.asarray(X), "fro"))   # noqa: E731
    proj_self = fro(T1 + T2) / (fro(T1) + fro(T2) + 1e-300)
    proj_op = fro(T1 + T2) / (n1 + n2)

    # negative control: a gradient with the v-block sign flipped.
    sgn = np.ones(g.n_u)
    sgn[g.n_ux:] = -1.0
    Gb = (sp.diags(sgn) @ Gr).tocsr()
    Ab = (Mu @ Gb).tocsr()
    neg = stk.spnorm_fro((Ab + A2).tocsr()) / (stk.spnorm_fro(Ab) + n2)

    C = stk.curl_matrix(g)
    DC = (D @ C).tocsr()
    dgt = (D + Gr.T).tocsr()

    # ---- matched NON-solenoidal control bases -------------------------------
    # chi_j = cos(k pi x) cos(l pi y) at cell centres; X_j = Grad chi_j is the
    # DISCRETE gradient of it, and is the pressure field the S3 probe uses.
    xp, yp = g.coords_p()
    xu, yu = g.coords_u()
    xv, yv = g.coords_v()
    Chi = np.column_stack([(np.cos(k * np.pi * xp)
                            * np.cos(l * np.pi * yp)).ravel()
                           for (k, l) in modes])                 # (n_p, M)
    X = Gr @ Chi                                                 # (n_u, M)

    # SELF-NORMALISED control, Psi_j = X_j / (h ||X_j||).  RETAINED ONLY AS A
    # DIAGNOSTIC: it is a normalised copy of X itself, so its matched cosine is
    # identically 1 and sqrt(M)*ctl_fro is identically 1 for ANY nonzero X --
    # even if Grad is wrong.  Revision 3 asserted on it; that assertion was
    # tautological and is retracted (STOKES-PHASE1C-VERIFY-codex.md).
    Psi = X / (g.h * np.linalg.norm(X, axis=0))[None, :]

    # INDEPENDENTLY CONSTRUCTED control: the ANALYTIC gradient of the same
    # chi_j, evaluated directly on the two face lattices.  It never touches the
    # Grad operator, so aligning with X_j is a real measurement.  Note
    # Grad_h chi has component factors sinc(k pi h/2) and sinc(l pi h/2) that
    # DIFFER between components, so the matched cosine is < 1 at coarse h and
    # rises to 1 -- it is not an identity.
    Ana = np.column_stack([
        g.pack(-k * np.pi * np.sin(k * np.pi * xu) * np.cos(l * np.pi * yu),
               -l * np.pi * np.cos(k * np.pi * xv) * np.sin(l * np.pi * yv))
        for (k, l) in modes])
    PsiA = Ana / (g.h * np.linalg.norm(Ana, axis=0))[None, :]

    # ---- REPAIRED S3: deterministic aligned pressures p = chi_j ------------
    MuX = X * (g.h ** 2)
    A_ctl = Psi.T @ MuX                                          # (M, M)
    A_ctlA = PsiA.T @ MuX                                        # (M, M)
    A_sol = Phi.T @ MuX                                          # (M, M)
    nX = np.linalg.norm(MuX, axis=0)
    nPsiF = np.linalg.norm(Psi, "fro")
    nPsiAF = np.linalg.norm(PsiA, "fro")
    nPhiF = np.linalg.norm(Phi, "fro")
    ctl_fro_j = np.linalg.norm(A_ctl, axis=0) / (nPsiF * nX)
    ctlA_fro_j = np.linalg.norm(A_ctlA, axis=0) / (nPsiAF * nX)
    sol_fro_j = np.linalg.norm(A_sol, axis=0) / (nPhiF * nX)
    PsiN = Psi / np.linalg.norm(Psi, axis=0)[None, :]
    PsiAN = PsiA / np.linalg.norm(PsiA, axis=0)[None, :]
    PhiN = Phi / np.linalg.norm(Phi, axis=0)[None, :]
    XN = X / np.linalg.norm(X, axis=0)[None, :]
    Cc = np.abs(PsiN.T @ XN)
    CcA = np.abs(PsiAN.T @ XN)
    Cs = np.abs(PhiN.T @ XN)
    matched_cos_j = np.diag(Cc)
    matchedA_cos_j = np.diag(CcA)
    offA = CcA.copy()
    np.fill_diagonal(offA, 0.0)

    # ---- retained DIAGNOSTIC: the original grid-white random pressure ------
    p = rng.standard_normal(g.n_p)
    p = p - p.mean()
    gp = Gr @ p
    ann_abs = float(np.linalg.norm(Phi.T @ (Mu @ gp)))
    ann_norm = ann_abs / (nPhiF * np.linalg.norm(Mu @ gp) + 1e-300)
    ctl_abs = float(np.linalg.norm(Psi.T @ (Mu @ gp)))
    ctl_norm = ctl_abs / (nPsiF * np.linalg.norm(Mu @ gp) + 1e-300)
    gpn = np.linalg.norm(gp) + 1e-300
    cos_sol = float(np.abs(Phi.T @ gp).max() / (gpn / g.h))
    cos_ctl = float(np.abs(Psi.T @ gp).max() / (gpn / g.h))

    dphi = float(np.linalg.norm(D @ Phi))
    dphi_norm = dphi / (stk.spnorm_fro(D) * np.linalg.norm(Phi) + 1e-300)

    return dict(
        N=N, n_u=g.n_u, n_p=g.n_p, n_psi=g.n_psi, M=M,
        adj_primary=primary,
        adj_norm_MuGrad_fro=n1, adj_norm_DtMp_fro=n2,
        adj_defect_fro=stk.spnorm_fro(Dfr),
        adj_defect_max=stk.spnorm_max(Dfr),
        adj_test_projected=proj_self,
        adj_test_projected_opnorm=proj_op,
        adj_test_defect_fro=fro(T1 + T2),
        adj_negative_control=neg,
        struct_D_plus_GradT_inf=stk.spnorm_inf(dgt),
        struct_D_plus_GradT_max=stk.spnorm_max(dgt),
        struct_DC_inf=stk.spnorm_inf(DC),
        struct_DC_max=stk.spnorm_max(DC),
        struct_DC_nnz=int(DC.nnz),
        # repaired S3 (deterministic, aligned)
        # --- GATED: independently constructed analytic-gradient control ---
        s3ind_ctl_fro_min=float(ctlA_fro_j.min()),
        s3ind_ctl_fro_max=float(ctlA_fro_j.max()),
        s3ind_matched_cos_min=float(matchedA_cos_j.min()),
        s3ind_offdiag_cos_max=float(offA.max()),
        # --- DIAGNOSTIC ONLY: the self-normalised control (tautological) ---
        s3_ctl_fro_min=float(ctl_fro_j.min()),
        s3_ctl_fro_max=float(ctl_fro_j.max()),
        s3_ctl_fro_ideal=float(1.0 / np.sqrt(M)),
        s3_ctl_fro_scaled_min=float(ctl_fro_j.min() * np.sqrt(M)),
        s3_matched_cos_min=float(matched_cos_j.min()),
        s3_sol_fro_max=float(sol_fro_j.max()),
        s3_sol_cos_max=float(Cs.max()),
        s3_n_pressures=int(M),
        # retained random-pressure diagnostic
        rand_press_annih_norm=ann_norm, rand_press_control_norm=ctl_norm,
        rand_press_cos_max_solenoidal=cos_sol,
        rand_press_cos_max_control=cos_ctl,
        D_Phi_abs=dphi, D_Phi_norm=dphi_norm,
        L_sym_max=stk.spnorm_max((stk.laplacian_matrix(g, "odd")
                                  - stk.laplacian_matrix(g, "odd").T).tocsr()))


def rank_gates_dense(N):
    g = stk.MacGrid(N)
    D = stk.divergence_matrix(g).toarray()
    C = stk.curl_matrix(g).toarray()
    t = time.time()
    rD = int(np.linalg.matrix_rank(D))
    rC = int(np.linalg.matrix_rank(C))
    return dict(N=N, rank_D=rD, dim_ker_D=int(g.n_u - rD), rank_C=rC,
                expect_rank_D=int(g.n_p - 1),
                expect_dim_ker_D=int((N - 1) ** 2),
                expect_rank_C=int((N - 1) ** 2),
                seconds=float(time.time() - t))


def rank_witness_indirect(N):
    """Cheap EXACT witness valid at every N, no dense SVD.

    rank(D) = n_p - dim ker(D^T) and D^T = -Grad, so rank D = n_p - dim ker Grad.
    ker Grad contains the constants (||Grad 1|| = 0).  If the bordered pressure
    Laplacian [[D Grad, 1],[1^T, 0]] is nonsingular then dim ker Grad <= 1, so
    it is exactly 1 and rank D = n_p - 1.  C is injective iff C^T C is
    nonsingular.  Both witnessed by a sparse LU with nonzero min |U_ii|.
    """
    g = stk.MacGrid(N)
    D = stk.divergence_matrix(g)
    Gr = stk.gradient_matrix(g)
    C = stk.curl_matrix(g)
    one = np.ones(g.n_p)
    grad_one = float(np.linalg.norm(Gr @ one))
    Lp = (D @ Gr).tocsr()
    Kb = sp.bmat([[Lp, sp.csr_matrix(one.reshape(-1, 1))],
                  [sp.csr_matrix(one.reshape(1, -1)), None]], format="csc")
    dU = np.abs(spla.splu(Kb).U.diagonal())
    dU2 = np.abs(spla.splu((C.T @ C).tocsc()).U.diagonal())
    return dict(N=N, grad_one_norm=grad_one,
                bordered_plap_min_absdiagU=float(dU.min()),
                bordered_plap_max_absdiagU=float(dU.max()),
                CtC_min_absdiagU=float(dU2.min()),
                CtC_max_absdiagU=float(dU2.max()),
                implied_rank_D=int(g.n_p - 1),
                implied_dim_ker_D=int(g.n_u - (g.n_p - 1)),
                implied_rank_C=int(g.n_psi))


def fom_run(N, ghost, nu, family="frozen"):
    g = stk.MacGrid(N)
    mf = stk.MMS_FAMILIES[family]["build"](g, nu=nu)
    D = stk.divergence_matrix(g)
    Gr = stk.gradient_matrix(g)
    L = stk.laplacian_matrix(g, ghost)
    t = time.time()
    u, p, info = track_solve(f"{family}/{ghost}", g, mf["f"], nu=nu,
                             ghost=ghost, ops=(D, Gr, L))
    dt = time.time() - t
    uex = mf["u"]
    pex = mf["p"] - mf["p"].mean()
    p = p - p.mean()
    err_u = stk.mass_rel(g, u, uex)
    err_p = stk.mass_rel(g, p, pex)
    U, V = g.unpack(u)
    Ue, Ve = g.unpack(uex)
    bu = np.concatenate([U[:, 0], U[:, -1], V[0, :], V[-1, :]])
    bue = np.concatenate([Ue[:, 0], Ue[:, -1], Ve[0, :], Ve[-1, :]])
    err_bnd = float(np.linalg.norm(bu - bue) / (np.linalg.norm(bue) + 1e-300))

    if ghost == "odd" and family == "frozen":
        ud, pd, a_, b_ = stk.exact_discrete(g)
        exact_u = stk.mass_rel(g, u, ud)
        exact_p = stk.mass_rel(g, p, pd)
        pred_u, pred_p = float(a_ - 1.0), float(b_ - 1.0)
        dev_u = float(abs(err_u - pred_u) / pred_u)
        dev_p = float(abs(err_p - pred_p) / pred_p)
    else:
        # None, not NaN: the final non-finite sweep must be unambiguous.
        exact_u = exact_p = pred_u = pred_p = dev_u = dev_p = None

    cos_eu = float(np.abs((u - uex) @ uex)
                   / (np.linalg.norm(u - uex) * np.linalg.norm(uex) + 1e-300))
    du = D @ u
    div_norm = float(np.linalg.norm(du)
                     / (stk.spnorm_fro(D) * np.linalg.norm(u) + 1e-300))
    return dict(N=N, ghost=ghost, nu=nu, family=family, n_u=g.n_u, n_p=g.n_p,
                dtype=str(u.dtype),
                err_u_mass_rel=err_u, err_p_mass_rel=err_p,
                err_u_l2_rel=float(np.linalg.norm(u - uex)
                                   / np.linalg.norm(uex)),
                err_p_l2_rel=float(np.linalg.norm(p - pex)
                                   / np.linalg.norm(pex)),
                err_u_bnd_rel=err_bnd, err_u_cos_with_uex=cos_eu,
                exact_discrete_u_rel=exact_u, exact_discrete_p_rel=exact_p,
                predicted_err_u=pred_u, predicted_err_p=pred_p,
                pred_dev_u=dev_u, pred_dev_p=dev_p,
                err_u_mass_abs=stk.mass_norm(g, u - uex),
                err_p_mass_abs=stk.mass_norm(g, p - pex),
                div_u_abs=float(np.linalg.norm(du)), div_u_norm=div_norm,
                lam=info["lam"], lin_resid_rel=info["lin_resid_rel"],
                backward_err=info["backward_err"], K_fro=info["K_fro"],
                p_mean_raw=info["p_mean"],
                saddle_dim=info["saddle_dim"], saddle_nnz=info["saddle_nnz"],
                solve_seconds=float(dt))


def orders(rows, key):
    return [dict(coarse=a["N"], fine=b["N"],
                 order=float(np.log2(a[key] / b[key])))
            for a, b in zip(rows[:-1], rows[1:])]


def anchor_rows(rows, anchors):
    out = []
    for r in rows:
        if r["N"] in anchors:
            au, ap = anchors[r["N"]]
            out.append(dict(N=r["N"], anchor_err_u=au, anchor_err_p=ap,
                            err_u=r["err_u_mass_rel"], err_p=r["err_p_mass_rel"],
                            rel_dev_u=float(abs(r["err_u_mass_rel"] - au) / au),
                            rel_dev_p=float(abs(r["err_p_mass_rel"] - ap) / ap)))
    return out


# ------------------------------------------------------------------- main ----

def main():
    t_all = time.time()
    rng = np.random.default_rng(SEED)
    tag = OUT_TAG or f"nu{NU:g}_M{M_MODES}"
    out = os.path.join(OUT_PREFIX, f"stk2d_fom_gates_{tag}.json")
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)

    jp = jax_provenance()
    report = dict(config=dict(
        pde="stokes2d", kind="staggered_MAC_FOM_phase1", driver_revision=4,
        discretization="MAC, N cells, h=1/N, p at N^2 centres (mean-zero "
                       "gauge), u_x on N(N-1) interior vertical faces, u_y on "
                       "N(N-1) interior horizontal faces, boundary-normal "
                       "velocities eliminated, no-slip tangential via ODD "
                       "ghosts (wall diagonal -5/h^2)",
        ns=NS, order_ladder=LADDER, generic_ns=GEN_NS, adj_ns=ADJ_NS,
        rank_ns=RANK_NS, freeslip_ns=FREESLIP_NS, M_modes=M_MODES, nu=NU,
        seed=SEED,
        mms_families={k: v["label"] for k, v in stk.MMS_FAMILIES.items()},
        norms="mass-weighted: ||x||_M = h ||x||_2 (M_u=M_p=h^2 I), so the "
              "mass-weighted RELATIVE error equals the plain relative "
              "2-norm on this uniform layout; both are recorded",
        solver="scipy.sparse.linalg.spsolve (SuperLU) on the bordered saddle "
               "system [[-nu L, Grad, 0],[D, 0, 1],[0, 1^T, 0]]",
        arithmetic="float64 throughout (numpy/scipy CPU); JAX is imported for "
                   "provenance and is ASSERTED by S0 because phase 2 runs in it",
        thresholds=dict(adj_tol=ADJ_TOL, order_target=ORDER_TARGET,
                        order_band=ORDER_BAND, neg_ctl_floor=NEG_CTL_FLOOR,
                        s3_floor=S3_FLOOR, s3_matched_cos=S3_MATCHED_COS,
                        s3_sol_tol=S3_SOL_TOL,
                        exact_field_tol=EXACT_FIELD_TOL, mmsf_tol=MMSF_TOL,
                        anchor_tol=ANCHOR_TOL, gen_anchor_tol=GEN_ANCHOR_TOL,
                        freeslip_err_floor=FREESLIP_ERR_FLOOR,
                        freeslip_order_ceil=FREESLIP_ORDER_CEIL),
        numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), jax=jp, allow_cpu=bool(ALLOW_CPU),
        smoke=bool(SMOKE),
        git_commit=git_commit(), hostname=os.uname().nodename),
        gates=dict(), rows=[], complete=False)

    def save():
        json.dump(report, open(out, "w"), indent=1, default=float)

    # Write complete=false IMMEDIATELY, before anything can fail, so a crash
    # can never leave an older complete=true artifact at this path untouched.
    save()
    log(f"stk2d FOM gates (driver rev 4) -> {out}")
    log(f"  numpy {np.__version__} scipy {scipy.__version__}  jax {jp}")

    # ---- S0: ASSERTED, not merely recorded --------------------------------
    probe = track_solve("S0_probe", stk.MacGrid(8),
                        stk.manufactured(stk.MacGrid(8))["f"])[0]
    s0 = dict(jax=jp, numpy_float64=str(probe.dtype),
              numpy_is_f64=bool(probe.dtype == np.float64),
              allow_cpu=bool(ALLOW_CPU),
              rule="solver output dtype must be float64; JAX must report "
                   "x64=True and matmul_precision='highest', and backend "
                   "'gpu' unless ALLOW_CPU=1.  Phase 1's numerics are CPU "
                   "scipy by design, but phase 2 inherits this JAX env")
    report["gates"]["S0"] = s0
    save()
    assert probe.dtype == np.float64, f"S0: solver dtype {probe.dtype} != f64"
    assert jp.get("imported"), f"S0: JAX did not import: {jp}"
    assert jp.get("x64") is True, "S0: JAX_ENABLE_X64 is not active"
    assert jp.get("matmul_precision") == "highest", \
        f"S0: JAX_DEFAULT_MATMUL_PRECISION={jp.get('matmul_precision')}"
    if not ALLOW_CPU:
        assert jp.get("backend") == "gpu", \
            f"S0: jax backend is {jp.get('backend')}, not gpu"
    log(f"  S0 asserted: dtype {probe.dtype}, jax x64 {jp.get('x64')}, "
        f"matmul {jp.get('matmul_precision')}, backend {jp.get('backend')}")

    # ---- PRECOND: the harness's own preconditions, ENFORCED ---------------
    # STOKES-PHASE1B-VERIFY-codex.md: the "complete=true implies every gate
    # passed" guarantee holds only under these.  They are now asserted rather
    # than assumed, so a JSON produced with assertions disabled or with an
    # emptied ladder cannot masquerade as a certified artifact.
    observed_cfg = dict(ns=NS, ladder=LADDER, generic_ns=GEN_NS, adj_ns=ADJ_NS,
                        rank_ns=RANK_NS, freeslip_ns=FREESLIP_NS,
                        M_modes=M_MODES, nu=NU, allow_cpu=int(ALLOW_CPU))
    cfg_mismatch = {k: [FROZEN_CONFIG[k], v] for k, v in observed_cfg.items()
                    if FROZEN_CONFIG[k] != v}
    pre = dict(debug_asserts_active=bool(__debug__), smoke=int(SMOKE),
               frozen_config=FROZEN_CONFIG, observed_config=observed_cfg,
               config_mismatch=cfg_mismatch,
               expected_gates=sorted(EXPECTED_GATES),
               expected_row_counts=EXPECTED_ROWS,
               rule="ASSERTED unless SMOKE=1: the ENTIRE configuration must "
                    "equal the frozen contract -- not merely be non-empty, "
                    "since env overrides could otherwise shorten a ladder and "
                    "still reach complete=true; the expected gate manifest "
                    "must be present with the EXACT expected row counts; and "
                    "no aggregated array may contain a non-finite value "
                    "(python's max([finite, nan]) returns the finite value, so "
                    "a NaN could otherwise turn a failed solve into a green "
                    "run).  Python must run WITHOUT -O, checked by a raise "
                    "rather than an assert.  A SMOKE=1 run NEVER sets "
                    "complete=true")
    report["gates"]["PRECOND"] = pre
    save()
    # NOT an assert: an assert cannot detect its own disablement under -O.
    if not __debug__:
        raise RuntimeError("PRECOND: python is running with -O, so every "
                           "assert in this harness is dead.  Refusing to "
                           "produce a JSON that would claim complete=true "
                           "without having checked anything.")
    if not SMOKE:
        assert not cfg_mismatch, ("PRECOND: configuration differs from the "
                                  f"frozen contract: {cfg_mismatch}")
    log(f"  PRECOND: asserts_active={__debug__} smoke={SMOKE} "
        f"config_mismatch={cfg_mismatch or 'none'}")

    # ---- gate REF ---------------------------------------------------------
    log(" gate REF: operators vs archived auditor reference")
    gate_ref(report)

    # ---- gate MMSF: the hand-derived analytic forcing ----------------------
    log(" gate MMSF: analytic Laplacian / gradient vs high-accuracy FD")
    mmsf = [stk.mms_forcing_consistency(f) for f in stk.MMS_FAMILIES]
    for r in mmsf:
        r["worst"] = float(finite(f"MMSF {r['family']}",
                                  [r[k] for k in ("lap_u_rel", "lap_v_rel",
                                                  "grad_px_rel", "grad_py_rel",
                                                  "div_rel", "wall_trace_max")]
                                  ).max())
        log(f"  {r['family']:8s}: lap_u {r['lap_u_rel']:.2e} "
            f"lap_v {r['lap_v_rel']:.2e} grad {r['grad_px_rel']:.2e}/"
            f"{r['grad_py_rel']:.2e} div {r['div_rel']:.2e} "
            f"wall {r['wall_trace_max']:.2e}")
    report["gates"]["MMSF"] = dict(
        rows=mmsf,
        worst=float(finite("MMSF", [r["worst"] for r in mmsf]).max()),
        tol=MMSF_TOL,
        rule="each family's analytic Laplacian, pressure gradient, continuous "
             "divergence and wall trace must match 4th-order finite "
             "differences of its own u,p to <= 1e-6 relative.  A sign or "
             "coefficient slip in the hand-derived forcing shows as O(1)")
    save()
    assert report["gates"]["MMSF"]["worst"] <= MMSF_TOL, "gate MMSF failed"

    # ---- gate MF ----------------------------------------------------------
    log(" gate MF: sparse vs independent matrix-free")
    mfr = [gate_mf(N, rng) for N in ADJ_NS]
    worst_mf = float(finite("MF", [r["worst"] for r in mfr]).max())
    report["gates"]["MF"] = dict(rows=mfr, worst_rel=float(worst_mf), tol=1e-13,
                                 rule="max relative disagreement over "
                                      "{L_odd,L_even,D,Grad,C} <= 1e-13")
    for r in mfr:
        log(f"  gate MF N={r['N']:4d}: worst rel {r['worst']:.2e}")
    assert worst_mf <= 1e-13, f"gate MF failed: {worst_mf}"

    # ---- S-ADJ + structure + S3 -------------------------------------------
    log(" S-ADJ / S-STRUCT / S-PRESS (repaired S3)")
    adj = [adjoint_gates(N, rng) for N in ADJ_NS]
    for r in adj:
        log(f"  N={r['N']:4d}  S-ADJ {r['adj_primary']:.3e} / "
            f"{r['adj_test_projected']:.3e}  neg-ctl "
            f"{r['adj_negative_control']:.3e}  ||DC||_inf "
            f"{r['struct_DC_inf']:.3e}  S3ind ctl_fro "
            f"{r['s3ind_ctl_fro_min']:.6f}  matched_cos "
            f"{r['s3ind_matched_cos_min']:.6f}  sol_fro "
            f"{r['s3_sol_fro_max']:.2e}  sol_cos {r['s3_sol_cos_max']:.2e}")
    report["gates"]["S_ADJ"] = dict(
        rows=adj,
        worst_primary=float(finite("adj_primary",
                                   [r["adj_primary"] for r in adj]).max()),
        worst_test_projected=float(finite(
            "adj_test_projected", [r["adj_test_projected"] for r in adj]).max()),
        worst_test_projected_opnorm=float(
            max(r["adj_test_projected_opnorm"] for r in adj)),
        min_negative_control=float(finite(
            "adj_neg_ctl", [r["adj_negative_control"] for r in adj]).min()),
        tol=ADJ_TOL, neg_ctl_floor=NEG_CTL_FLOOR,
        rule="||M_u Grad + D^T M_p||_F/(||M_u Grad||_F+||D^T M_p||_F) <= 1e-14 "
             "AND ||Phi^T(M_u Grad + D^T M_p)||_F normalized the same way "
             "<= 1e-14; the sign-flipped-v-block negative control must be "
             ">= 1e-2 (ASSERTED, so the gate is demonstrably falsifiable)")
    report["gates"]["S_STRUCT"] = dict(
        rows=[{k: r[k] for k in r if k.startswith("struct_")
               or k in ("N", "n_u", "n_p", "n_psi")} for r in adj],
        worst_DC_inf=float(finite("struct_DC_inf",
                                  [r["struct_DC_inf"] for r in adj]).max()),
        worst_D_plus_GradT_inf=float(max(r["struct_D_plus_GradT_inf"]
                                         for r in adj)),
        rule="||D + Grad^T||_inf and ||D C||_inf must be exactly 0")
    report["gates"]["S_PRESS"] = dict(
        rows=[{k: r[k] for k in r
               if k.startswith("s3_") or k.startswith("rand_press_")
               or k.startswith("D_Phi") or k in ("N", "M")} for r in adj],
        min_ctl_fro_independent=float(
            finite("s3ind_ctl_fro", [r["s3ind_ctl_fro_min"] for r in adj]).min()),
        min_matched_cos_independent=float(
            finite("s3ind_matched_cos",
                   [r["s3ind_matched_cos_min"] for r in adj]).min()),
        max_offdiag_cos_independent=float(
            finite("s3ind_offdiag", [r["s3ind_offdiag_cos_max"] for r in adj]).max()),
        min_ctl_fro_selfnorm_diagnostic=float(
            finite("s3_ctl_fro", [r["s3_ctl_fro_min"] for r in adj]).min()),
        min_ctl_fro_scaled_selfnorm_diagnostic=float(
            finite("s3_scaled", [r["s3_ctl_fro_scaled_min"] for r in adj]).min()),
        min_matched_cos_selfnorm_diagnostic=float(
            finite("s3_matched", [r["s3_matched_cos_min"] for r in adj]).min()),
        max_sol_fro=float(finite("s3_sol_fro",
                                 [r["s3_sol_fro_max"] for r in adj]).max()),
        max_sol_cos=float(finite("s3_sol_cos",
                                 [r["s3_sol_cos_max"] for r in adj]).max()),
        worst_D_Phi_norm=float(finite("D_Phi_norm",
                                      [r["D_Phi_norm"] for r in adj]).max()),
        floor=S3_FLOOR, matched_cos=S3_MATCHED_COS, sol_tol=S3_SOL_TOL,
        selfnorm_scaled_gated=False,
        rule="REPAIRED S3 (STOKES-PHASE1-VERIFY-codex.md).  Deterministic "
             "p = chi_kl aligned with each of the M control columns: "
             "matched-control cosine >= 0.99, control Frobenius metric "
             ">= 1e-2 (the design's own floor, = 1/sqrt(M) exactly for a "
             "mass-orthonormal control), solenoidal Frobenius metric and "
             "cosine <= 1e-13.  The rand_press_* fields are the SUPERSEDED "
             "grid-white-random-pressure diagnostic, retained as evidence and "
             "NOT gated.  THE GATED CONTROL IS THE INDEPENDENT ONE "
             "(s3ind_*): the ANALYTIC gradient of the same chi_j sampled on "
             "the two face lattices, which never touches the Grad operator.  "
             "Its matched cosine is a real measurement (0.9939 at N=8 rising "
             "to 1 as h -> 0, because Grad_h chi carries component factors "
             "sinc(k pi h/2) != sinc(l pi h/2)) and it collapses to 0 under a "
             "v-block sign flip in Grad.  The SELF-NORMALISED control (s3_*, "
             "Psi_j = X_j/(h||X_j||)) is retained as a DIAGNOSTIC ONLY: it is "
             "a normalised copy of X, so its matched cosine and "
             "sqrt(M)*ctl_fro are identically 1 for any nonzero X even if "
             "Grad is wrong.  Revisions 2-3 asserted on it; that assertion was "
             "TAUTOLOGICAL and is retracted -- see STOKES-NOTES.md retraction "
             "9.  Known blind spot of the cosine form: it is invariant to a "
             "global scale or global sign of Grad; those are covered by S-ADJ "
             "and S-FOM")
    report["gates"]["SYM"] = dict(
        rows=[dict(N=r["N"], L_sym_max=r["L_sym_max"]) for r in adj],
        worst=float(finite("L_sym", [r["L_sym_max"] for r in adj]).max()),
        rule="||L - L^T||_max exactly 0")
    save()

    GA = report["gates"]["S_ADJ"]
    GP = report["gates"]["S_PRESS"]
    assert GA["worst_primary"] <= ADJ_TOL, f"S-ADJ primary: {GA['worst_primary']}"
    assert GA["worst_test_projected"] <= ADJ_TOL, "S-ADJ test-projected"
    assert GA["min_negative_control"] >= NEG_CTL_FLOOR, \
        f"S-ADJ negative control too small: {GA['min_negative_control']}"
    assert report["gates"]["S_STRUCT"]["worst_DC_inf"] == 0.0, "||DC|| != 0"
    assert report["gates"]["S_STRUCT"]["worst_D_plus_GradT_inf"] == 0.0
    assert report["gates"]["SYM"]["worst"] == 0.0, "L not symmetric"
    assert GP["min_ctl_fro_independent"] >= S3_FLOOR, \
        f"S3 control floor failed: {GP['min_ctl_fro_independent']}"
    assert GP["min_matched_cos_independent"] >= S3_MATCHED_COS, \
        f"S3 matched-control cosine failed: {GP['min_matched_cos_independent']}"
    assert GP["max_offdiag_cos_independent"] <= 1e-12, \
        f"S3 control columns not mutually distinct: {GP['max_offdiag_cos_independent']}"
    assert GP["max_sol_fro"] <= S3_SOL_TOL, \
        f"S3 solenoidal projection failed: {GP['max_sol_fro']}"
    assert GP["max_sol_cos"] <= S3_SOL_TOL, \
        f"S3 solenoidal cosine failed: {GP['max_sol_cos']}"

    # ---- ranks: ASSERTED ---------------------------------------------------
    log(" S-RANK (dense SVD + exact sparse-LU witness)")
    dense = [rank_gates_dense(N) for N in RANK_NS]
    for r in dense:
        log(f"  N={r['N']:4d}: rank D {r['rank_D']} (expect {r['expect_rank_D']})"
            f"  dim ker D {r['dim_ker_D']} (expect {r['expect_dim_ker_D']})"
            f"  rank C {r['rank_C']} (expect {r['expect_rank_C']})"
            f"  [{r['seconds']:.0f}s]")
    ind = [rank_witness_indirect(N) for N in ADJ_NS]
    report["gates"]["S_RANK"] = dict(
        dense=dense, indirect=ind,
        dense_n=len(dense),
        dense_all_match=int(sum(
            (r["rank_D"] == r["expect_rank_D"])
            and (r["dim_ker_D"] == r["expect_dim_ker_D"])
            and (r["rank_C"] == r["expect_rank_C"]) for r in dense)),
        min_grad_one_norm=float(max(r["grad_one_norm"] for r in ind)),
        min_plap_absdiagU=float(min(r["bordered_plap_min_absdiagU"]
                                    for r in ind)),
        min_CtC_absdiagU=float(min(r["CtC_min_absdiagU"] for r in ind)),
        rule="ASSERTED.  dense: rank D = N^2-1, dim ker D = rank C = (N-1)^2. "
             "indirect: ||Grad 1|| exactly 0 and both bordered-LU witnesses "
             "have min |U_ii| > 0, which forces the same values at every mesh")
    save()
    for r in dense:
        assert r["rank_D"] == r["expect_rank_D"], f"rank D at N={r['N']}"
        assert r["dim_ker_D"] == r["expect_dim_ker_D"], f"dim ker D at N={r['N']}"
        assert r["rank_C"] == r["expect_rank_C"], f"rank C at N={r['N']}"
    for r in ind:
        assert r["grad_one_norm"] == 0.0, f"||Grad 1|| != 0 at N={r['N']}"
        assert r["bordered_plap_min_absdiagU"] > 0.0, f"plap LU at N={r['N']}"
        assert r["CtC_min_absdiagU"] > 0.0, f"CtC LU at N={r['N']}"

    # ---- S-FOM (frozen family): ASSERTED order AND anchors ------------------
    log(f" S-FOM (frozen MMS, odd ghosts = no-slip), nu={NU}")
    rows = []
    for N in NS:
        r = fom_run(N, "odd", NU, "frozen")
        rows.append(r)
        report["rows"].append(r)
        log(f"  N={N:4d}  err_u {r['err_u_mass_rel']:.4e}  "
            f"err_p {r['err_p_mass_rel']:.4e}  cos {r['err_u_cos_with_uex']:.4f}"
            f"  div {r['div_u_norm']:.2e}  lam {r['lam']:.2e}  "
            f"[{r['solve_seconds']:.1f}s]")
        save()
    lad = [r for r in rows if r["N"] in LADDER]
    ou, op = orders(lad, "err_u_mass_rel"), orders(lad, "err_p_mass_rel")
    worst_dev = float(np.max(np.abs(
        finite("S_FOM orders", [o["order"] for o in ou + op]) - ORDER_TARGET)))
    amatch = anchor_rows(rows, ANCHORS_FROZEN)
    worst_anchor = float(finite("S_FOM anchors",
                                [v for a in amatch
                                 for v in (a["rel_dev_u"], a["rel_dev_p"])]).max())
    report["gates"]["S_FOM"] = dict(
        rows=rows, ladder=LADDER, orders_u=ou, orders_p=op,
        worst_order_deviation=float(worst_dev),
        min_order=float(min(o["order"] for o in ou + op)),
        max_order=float(max(o["order"] for o in ou + op)),
        anchors=amatch, worst_anchor_rel_dev=float(worst_anchor),
        target=ORDER_TARGET, band=ORDER_BAND, anchor_tol=ANCHOR_TOL,
        rule="ASSERTED: observed order 2.00 +/- 0.05 in BOTH variables over "
             "the frozen ladder, mass-weighted relative norms, AND agreement "
             "with the r1/r2 audit anchors to <= 1e-3 relative (they are "
             "quoted to 4 significant figures)")
    log(f"  orders u: {[round(o['order'], 4) for o in ou]}")
    log(f"  orders p: {[round(o['order'], 4) for o in op]}")
    log(f"  worst order deviation {worst_dev:.4f} (band {ORDER_BAND}); "
        f"worst anchor rel dev {worst_anchor:.2e} (tol {ANCHOR_TOL})")
    save()
    assert worst_dev <= ORDER_BAND, f"S-FOM order failed: dev {worst_dev}"
    assert worst_anchor <= ANCHOR_TOL, f"S-FOM anchors failed: {worst_anchor}"

    # ---- S-EXACT: field agreement is the gate; pred_dev is a DIAGNOSTIC ----
    ex = []
    for r in rows:
        # What the FIELD gate already implies for pred_dev, via the reverse
        # triangle inequality: | ||x-z||/||z|| - eps | / eps <= a * rho / eps.
        # Recorded to show pred_dev is a CONSEQUENCE of the field gate, not an
        # independent test.  It is NOT used as a threshold -- see the rule.
        a_u = 1.0 + r["predicted_err_u"]
        a_p = 1.0 + r["predicted_err_p"]
        ex.append(dict(N=r["N"], exact_u_rel=r["exact_discrete_u_rel"],
                       exact_p_rel=r["exact_discrete_p_rel"],
                       predicted_err_u=r["predicted_err_u"],
                       predicted_err_p=r["predicted_err_p"],
                       observed_err_u=r["err_u_mass_rel"],
                       observed_err_p=r["err_p_mass_rel"],
                       pred_dev_u_diagnostic=r["pred_dev_u"],
                       pred_dev_p_diagnostic=r["pred_dev_p"],
                       implied_by_field_gate_u=float(
                           a_u * EXACT_FIELD_TOL / r["predicted_err_u"]),
                       implied_by_field_gate_p=float(
                           a_p * EXACT_FIELD_TOL / r["predicted_err_p"])))
    report["gates"]["S_EXACT"] = dict(
        rows=ex,
        worst_exact_u_rel=float(finite("S_EXACT u",
                                       [r["exact_u_rel"] for r in ex]).max()),
        worst_exact_p_rel=float(finite("S_EXACT p",
                                       [r["exact_p_rel"] for r in ex]).max()),
        worst_pred_dev_u_diagnostic=float(max(r["pred_dev_u_diagnostic"]
                                              for r in ex)),
        worst_pred_dev_p_diagnostic=float(max(r["pred_dev_p_diagnostic"]
                                              for r in ex)),
        field_tol=EXACT_FIELD_TOL, pred_dev_gated=False,
        rule="The discrete saddle system has a CLOSED-FORM solution for this "
             "manufactured data (stk2d_common.exact_discrete): "
             "u_h = (t/sin t)^2 u_ex, p_h = (t/sin t) p_ex, t = pi h.  THE "
             "GATE IS THE FIELD ASSERTION: ||u_h - u_h^exact||/||.|| <= 1e-8 "
             "(and likewise for p), a frozen a-priori tolerance.  "
             "pred_dev_*_diagnostic (observed vs closed-form-predicted ERROR) "
             "is RECORDED ONLY and is NOT a gate.  Two earlier attempts to "
             "gate it were both wrong and are retracted in STOKES-NOTES.md: "
             "revision 1 used the flat field tolerance, which its own recorded "
             "2.331e-8 exceeded; revision 2 used pred_dev <= "
             "exact_rel/predicted_err, which omits a factor a = 1+eps AND is "
             "circular -- the bound derives from the very error it tests.  "
             "STOKES-PHASE1B-VERIFY-codex.md showed it is direction-dependent "
             "and blind: a parallel perturbation INSIDE the field tolerance "
             "(field error 9.99e-9) gives margin 1.0000251 and FAILS, while a "
             "10% ORTHOGONAL perturbation gives margin 0.999774 and PASSES.  "
             "implied_by_field_gate_* is what the field gate alone already "
             "forces on pred_dev, showing the quantity is redundant as a gate")
    for r in ex:
        log(f"  N={r['N']:4d}  field u {r['exact_u_rel']:.3e} p "
            f"{r['exact_p_rel']:.3e}   [diagnostic] pred_dev u "
            f"{r['pred_dev_u_diagnostic']:.2e} p "
            f"{r['pred_dev_p_diagnostic']:.2e}")
    save()
    assert report["gates"]["S_EXACT"]["worst_exact_u_rel"] <= EXACT_FIELD_TOL
    assert report["gates"]["S_EXACT"]["worst_exact_p_rel"] <= EXACT_FIELD_TOL

    # ---- S-FOMGEN: the generic manufactured solution -----------------------
    log(" S-FOMGEN (generic MMS, odd ghosts)")
    grows = []
    for N in GEN_NS:
        r = fom_run(N, "odd", NU, "generic")
        grows.append(r)
        report["rows"].append(r)
        log(f"  N={N:4d}  err_u {r['err_u_mass_rel']:.6e}  "
            f"err_p {r['err_p_mass_rel']:.6e}  cos "
            f"{r['err_u_cos_with_uex']:.4f}  div {r['div_u_norm']:.2e}  "
            f"[{r['solve_seconds']:.1f}s]")
        save()
    gou, gop = orders(grows, "err_u_mass_rel"), orders(grows, "err_p_mass_rel")
    gdev = float(np.max(np.abs(
        finite("S_FOMGEN orders", [o["order"] for o in gou + gop])
        - ORDER_TARGET)))
    gamatch = anchor_rows(grows, ANCHORS_GENERIC)
    gworst = float(finite("S_FOMGEN anchors",
                          [v for a in gamatch
                           for v in (a["rel_dev_u"], a["rel_dev_p"])]).max())
    max_cos = float(finite("S_FOMGEN cosine",
                           [r["err_u_cos_with_uex"] for r in grows]).max())
    report["gates"]["S_FOMGEN"] = dict(
        rows=grows, orders_u=gou, orders_p=gop,
        worst_order_deviation=float(gdev), anchors=gamatch,
        worst_anchor_rel_dev=float(gworst),
        max_err_solution_cosine=float(max_cos),
        min_err_solution_cosine=float(min(r["err_u_cos_with_uex"]
                                          for r in grows)),
        target=ORDER_TARGET, band=ORDER_BAND, anchor_tol=GEN_ANCHOR_TOL,
        nondegeneracy_ceil=0.99,
        rule="ASSERTED: order 2.00 +/- 0.05 in both variables; agreement with "
             "the values tabulated in STOKES-PHASE1-VERIFY-codex.md to "
             "<= 1e-5 relative (quoted to 7 significant figures); AND "
             "error/solution cosine < 0.99, i.e. this family must NOT be "
             "degenerate the way the frozen one is (verifier measured ~0.912)")
    log(f"  generic orders u: {[round(o['order'], 4) for o in gou]}")
    log(f"  generic orders p: {[round(o['order'], 4) for o in gop]}")
    log(f"  worst order dev {gdev:.4f}; worst verifier-value rel dev "
        f"{gworst:.2e}; err/solution cosine max {max_cos:.4f}")
    save()
    assert gdev <= ORDER_BAND, f"S-FOMGEN order failed: {gdev}"
    assert gworst <= GEN_ANCHOR_TOL, f"S-FOMGEN vs verifier failed: {gworst}"
    assert max_cos < 0.99, \
        f"S-FOMGEN is degenerate too (cosine {max_cos}); it adds nothing"

    # ---- S-NU: exact 1/nu scaling -----------------------------------------
    log(" S-NU: exact 1/nu velocity scaling at fixed f")
    Nn = LADDER[0]
    gn = stk.MacGrid(Nn)
    mfn = stk.manufactured(gn, nu=1.0)
    ops_n = (stk.divergence_matrix(gn), stk.gradient_matrix(gn),
             stk.laplacian_matrix(gn, "odd"))
    u1, p1, _ = track_solve("S_NU/nu=1", gn, mfn["f"], nu=1.0, ghost="odd",
                            ops=ops_n)
    nu2 = 7.0
    u2, p2, _ = track_solve("S_NU/nu=7", gn, mfn["f"], nu=nu2, ghost="odd",
                            ops=ops_n)
    du_ = float(np.linalg.norm(nu2 * u2 - u1) / np.linalg.norm(u1))
    p1c, p2c = p1 - p1.mean(), p2 - p2.mean()
    dp_ = float(np.linalg.norm(p2c - p1c) / np.linalg.norm(p1c))
    report["gates"]["S_NU"] = dict(
        N=Nn, nu_a=1.0, nu_b=nu2, u_scaling_rel=du_, p_invariance_rel=dp_,
        tol=1e-9,
        rule="with f fixed, (u/nu, p) solves at viscosity nu exactly: "
             "||nu*u_nu - u_1||/||u_1|| and ||p_nu - p_1||/||p_1|| <= 1e-9.  "
             "RETRACTED THRESHOLD: this gate was first written with 1e-11 and "
             "FAILED at p_invariance_rel = 1.077e-11 (N=32).  The identity is "
             "exact in exact arithmetic; the verifier confirmed roundoff by "
             "changing only the SuperLU permutation, which moved the same "
             "number to 9.06e-14 (MMD_ATA) and 5.12e-14 (NATURAL).  A real "
             "defect does not vanish under a reordering.  The number, not the "
             "threshold, is the result")
    log(f"  N={Nn}: ||nu u_nu - u_1||/||u_1|| = {du_:.3e}; "
        f"||p_nu - p_1||/||p_1|| = {dp_:.3e}")
    save()
    assert du_ <= 1e-9 and dp_ <= 1e-9, "S-NU failed"

    # ---- S-FREESLIP: ASSERTED to fail S-FOM --------------------------------
    if FREESLIP_NS:
        log(" S-FREESLIP (even ghosts = the bug S-FOM exists to catch)")
        fs = []
        for N in FREESLIP_NS:
            r = fom_run(N, "even", NU, "frozen")
            fs.append(r)
            log(f"  N={N:4d}  err_u {r['err_u_mass_rel']:.4e}  "
                f"err_p {r['err_p_mass_rel']:.4e}  "
                f"bnd {r['err_u_bnd_rel']:.3e}  [{r['solve_seconds']:.1f}s]")
        fou, fop = orders(fs, "err_u_mass_rel"), orders(fs, "err_p_mass_rel")
        min_err = float(finite("S_FREESLIP err",
                               [r["err_u_mass_rel"] for r in fs]).min())
        max_ord = float(np.max(np.abs(finite(
            "S_FREESLIP orders", [o["order"] for o in fou + fop]))))
        report["gates"]["S_FREESLIP"] = dict(
            rows=fs, orders_u=fou, orders_p=fop,
            min_err_u=float(min_err), max_abs_order=float(max_ord),
            finest_order_u=float(fou[-1]["order"]) if fou else None,
            finest_order_p=float(fop[-1]["order"]) if fop else None,
            err_ratio_vs_noslip_finest=float(
                fs[-1]["err_u_mass_rel"]
                / [r for r in rows if r["N"] == fs[-1]["N"]][0]
                ["err_u_mass_rel"]),
            err_floor=FREESLIP_ERR_FLOOR, order_ceil=FREESLIP_ORDER_CEIL,
            rule="ASSERTED TO FAIL S-FOM.  Even (free-slip) tangential ghosts "
                 "solve a DIFFERENT boundary-value problem, so the error must "
                 "plateau at O(1) (>= 0.5 at every mesh) and must not "
                 "converge (|observed order| <= 0.5).  If this arm ever "
                 "looked second-order, S-FOM would be blind to the failure "
                 "mode both audits single out")
        log(f"  freeslip orders u: {[round(o['order'], 4) for o in fou]}")
        log(f"  freeslip orders p: {[round(o['order'], 4) for o in fop]}")
        log(f"  min err_u {min_err:.4f} (floor {FREESLIP_ERR_FLOOR}); "
            f"max |order| {max_ord:.4f} (ceil {FREESLIP_ORDER_CEIL})")
        save()
        assert min_err >= FREESLIP_ERR_FLOOR, \
            f"S-FREESLIP too accurate ({min_err}): S-FOM may be blind"
        assert max_ord <= FREESLIP_ORDER_CEIL, \
            f"S-FREESLIP converged (order {max_ord}): S-FOM may be blind"

    # ---- S-BACKERR: independent, pre-frozen checks on the linear algebra ---
    # Replaces the retracted S-EXACT prediction gate.  FOUR asserted numbers,
    # not one: the global normalised backward error PLUS the three blockwise
    # residuals.  The blocks are separate because the global metric cannot see
    # the bordered rows -- ||K||_F is dominated by the O(h^-2) momentum block,
    # so a violated continuity or mean-zero-gauge row hides inside it.  Proven
    # negative control (STOKES-PHASE1C-VERIFY-codex.md, reproduced here): a
    # constant 1e-8 pressure offset at N=128 leaves gauge_raw = 1.6384e-4 while
    # the global metric reads 4.46e-14 and PASSES.  Covers all 17 solves the
    # driver performs, including the S0 probe and both S-NU solves, which
    # revision 3 silently excluded.
    be_rows = list(SOLVES)
    worst_be = float(finite("S_BACKERR backward_err",
                            [r["backward_err"] for r in be_rows]).max())
    worst_mom = float(finite("S_BACKERR mom",
                             [r["mom_resid"] for r in be_rows]).max())
    worst_cont = float(finite("S_BACKERR cont",
                              [r["cont_resid"] for r in be_rows]).max())
    worst_gauge = float(finite("S_BACKERR gauge",
                               [r["gauge_resid"] for r in be_rows]).max())
    worst_gauge_raw = float(np.max(np.abs(finite(
        "S_BACKERR gauge_raw", [r["gauge_raw"] for r in be_rows]))))
    report["gates"]["S_BACKERR"] = dict(
        rows=be_rows, n_solves=len(be_rows),
        worst=worst_be, worst_mom_resid=worst_mom,
        worst_cont_resid=worst_cont, worst_gauge_resid=worst_gauge,
        worst_gauge_raw=worst_gauge_raw,
        tol=BACKERR_TOL, mom_tol=BACKERR_TOL, cont_tol=CONT_TOL,
        gauge_tol=GAUGE_TOL, gauge_raw_tol=GAUGE_RAW_TOL,
        expected_solves=EXPECTED_ROWS["S_BACKERR"],
        rule="ASSERTED over EVERY solve_stokes call in the run (all "
             "17: S0 probe, 6 frozen, 3 generic, 2 S-NU, 5 free-slip).  "
             "(a) global normalised backward error "
             "||K x - b|| / (||K||_F ||x|| + ||b||) <= 1e-13; (b) momentum "
             "block ||r_mom||/(nu||L||_F||u|| + ||Grad||_F||p|| + ||f||) "
             "<= 1e-13; (c) continuity block "
             "||r_cont||/(||D||_F||u|| + |lam| sqrt(n_p)) <= 1e-12; (d) gauge "
             "row |1^T p|/(sqrt(n_p)||p||) <= 1e-12 AND the RAW |1^T p| "
             "<= 1e-8.  (b)-(d) exist because (a) alone is blind to the "
             "bordered rows.  All thresholds are FROZEN ENGINEERING "
             "thresholds -- 1e-13 is about 450 machine epsilons and predates "
             "the certified artifact -- NOT rigorously derived from "
             "O(sqrt(nnz) u): that expression gives 1.27e-13 at N=256 and "
             "sparse-LU pivot growth prevents it being a hard bound.  "
             "S-BACKERR is REFERENCE-direction independent (it has no "
             "pathology aligned with the manufactured solution, unlike the "
             "retracted pred_dev) but not literally perturbation-direction "
             "independent: it measures ||K dx||, and at N=32 equal 1e-11 "
             "relative perturbations give 1.99976e-13 (random), 2.78032e-13 "
             "(alternating high-frequency) and 2.38238e-15 (parallel to "
             "velocity).  It cannot detect a WRONG K or b that is solved "
             "accurately.  ||res||/||rhs|| is also recorded and grows like "
             "h^-2 because ||K|| does; that unnormalised form is a "
             "diagnostic, not a gate")
    log(f" S-BACKERR over {len(be_rows)} solves: global {worst_be:.3e} "
        f"(tol {BACKERR_TOL})  mom {worst_mom:.3e}  cont {worst_cont:.3e}  "
        f"gauge {worst_gauge:.3e} (raw {worst_gauge_raw:.3e})")
    save()
    assert worst_be <= BACKERR_TOL, f"S-BACKERR global failed: {worst_be}"
    assert worst_mom <= BACKERR_TOL, f"S-BACKERR momentum failed: {worst_mom}"
    assert worst_cont <= CONT_TOL, f"S-BACKERR continuity failed: {worst_cont}"
    assert worst_gauge <= GAUGE_TOL, f"S-BACKERR gauge failed: {worst_gauge}"
    assert worst_gauge_raw <= GAUGE_RAW_TOL, \
        f"S-BACKERR raw mean-zero gauge failed: {worst_gauge_raw}"

    # ---- final manifest + non-finite sweep --------------------------------
    counts = dict(REF=len(report["gates"]["REF"]["rows"]),
                  MMSF=len(report["gates"]["MMSF"]["rows"]),
                  MF=len(report["gates"]["MF"]["rows"]),
                  S_ADJ=len(report["gates"]["S_ADJ"]["rows"]),
                  S_STRUCT=len(report["gates"]["S_STRUCT"]["rows"]),
                  S_PRESS=len(report["gates"]["S_PRESS"]["rows"]),
                  SYM=len(report["gates"]["SYM"]["rows"]),
                  S_FOM=len(report["gates"]["S_FOM"]["rows"]),
                  S_EXACT=len(report["gates"]["S_EXACT"]["rows"]),
                  S_FOMGEN=len(report["gates"]["S_FOMGEN"]["rows"]),
                  S_FREESLIP=len(report["gates"]["S_FREESLIP"]["rows"]),
                  S_RANK_dense=len(report["gates"]["S_RANK"]["dense"]),
                  S_RANK_indirect=len(report["gates"]["S_RANK"]["indirect"]),
                  S_BACKERR=len(be_rows))
    missing = sorted(EXPECTED_GATES - set(report["gates"]))
    bad_counts = {k: [EXPECTED_ROWS[k], counts[k]] for k in EXPECTED_ROWS
                  if counts[k] != EXPECTED_ROWS[k]}
    nonfinite = []

    def _sweep(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                _sweep(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                _sweep(v, f"{path}[{i}]")
        elif isinstance(node, float) and not np.isfinite(node):
            nonfinite.append(path)

    _sweep(report["gates"], "gates")
    _sweep(report["rows"], "rows")
    report["gates"]["MANIFEST"] = dict(
        expected_gates=sorted(EXPECTED_GATES), missing_gates=missing,
        expected_row_counts=EXPECTED_ROWS, observed_row_counts=counts,
        row_count_mismatch=bad_counts, nonfinite_fields=nonfinite,
        rule="ASSERTED unless SMOKE=1: every expected gate present, EXACT "
             "expected row counts (non-empty is not sufficient), and no "
             "non-finite float anywhere in gates/ or rows/.  Fields that do "
             "not apply (the closed form on the generic and free-slip arms) "
             "are recorded as null, never NaN, so this sweep is unambiguous")
    log(f" MANIFEST: missing {missing or 'none'}  row-count mismatches "
        f"{bad_counts or 'none'}  non-finite {nonfinite or 'none'}")
    save()
    if not SMOKE:
        assert not missing, f"MANIFEST: missing gates {missing}"
        assert not bad_counts, f"MANIFEST: row-count mismatch {bad_counts}"
    assert not nonfinite, f"MANIFEST: non-finite values at {nonfinite}"

    # A SMOKE run NEVER produces a certified artifact: it is not merely
    # labelled, complete stays false.
    report["complete"] = not bool(SMOKE)
    report["certified"] = not bool(SMOKE)
    if SMOKE:
        report["incomplete_reason"] = (
            "SMOKE=1: PRECOND config equality and the MANIFEST row counts "
            "were not enforced, so this run is not a certified artifact and "
            "complete stays false by construction")
    report["total_seconds"] = float(time.time() - t_all)
    save()
    log(f"DONE stk2d FOM gates [{report['total_seconds']:.0f}s] "
        f"complete={report['complete']} -> {out}")


if __name__ == "__main__":
    main()
