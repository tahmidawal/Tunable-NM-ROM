"""Feasibility harness for a faster and more useful Poisson NM-ROM warm start.

The delivered answer is always produced by the same true-residual counting CG used by the
audited hybrid study. Only x0 changes.

Online neural path:
    known source parameters -> train-only-calibrated RBF -> K=8 latent
    -> FiLM decode on a fixed coarse grid -> linear prolongation -> hard-BC reset

An optional q-by-q sine residual correction exactly removes those error modes before CG.
The correction is classical, so learned-only, combined, and spectral-only arms are distinct.

Usage:
  PKL=... NS=256,512 TEST_SEED=0 \
    /absolute/venv/python feasibility.py out.json

Environment defaults are deliberately small only for explicit SMOKE=1. Real results must set
SMOKE=0, run on a GPU, use f64/highest precision, and retain every timing repetition.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


HERE = os.path.dirname(os.path.abspath(__file__))
EXPS = os.path.dirname(HERE)
for path in (
    os.path.join(HERE, "deps"),
    os.environ.get("GROUP_ARCH_DIR", ""),
    os.path.abspath(os.path.join(
        HERE, "..", "..", "..", "2026-08-19-nonlinear-decoder-architecture",
        "experiments", "nonlinear-decoder-architecture",
    )),
    os.path.join(EXPS, "poisson2d-rom-objective"),
    os.path.join(EXPS, "poisson2d-rom-objective", "followup"),
    os.path.join(EXPS, "rom-warmstart-fom"),
):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import pro_common as pc  # noqa: E402
from pro_common import mp, F64  # noqa: E402
from fu_eq import eq_fit as legacy_eq_fit  # noqa: E402
from wsf_poisson import make_lm_obj_jit  # noqa: E402
import wsf_util as wu  # noqa: E402


OUT = sys.argv[1]
PKL = os.environ["PKL"]
PARAM_PKL = os.environ.get("PARAM_PKL", "")
GROUP_PKL = os.environ.get("GROUP_PKL", "")
TRANSPORT_PKL = os.environ.get("TRANSPORT_PKL", "")
SMOKE = bool(int(os.environ.get("SMOKE", "0")))
NS = [int(v) for v in os.environ.get("NS", "32" if SMOKE else "256,512").split(",")]
FOM_TAUS = [float(v) for v in os.environ.get(
    "FOM_TAUS", "1e-6" if SMOKE else "1e-6,1e-8,1e-10").split(",")]
N_TEST = int(os.environ.get("N_TEST", "2" if SMOKE else "16"))
N_TIME = int(os.environ.get("N_TIME", "1" if SMOKE else "8"))
TIME_REPS = int(os.environ.get("TIME_REPS", "2" if SMOKE else "7"))
TIME_WARM = int(os.environ.get("TIME_WARM", "1" if SMOKE else "2"))
BURN_S = float(os.environ.get("BURN_S", "0" if SMOKE else "3"))
CG_MAXITER = int(os.environ.get("CG_MAXITER", "50000"))
TEST_SEED = int(os.environ.get("TEST_SEED", "0"))
M_MODES = int(os.environ.get("M", "64"))
MQ = int(os.environ.get("MQ", "256"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
LM_ROM_TAU = float(os.environ.get("LM_ROM_TAU", "0.01"))
GROUP_M = int(os.environ.get("GROUP_M", "128"))
GROUP_MQ = int(os.environ.get("GROUP_MQ", "512"))
GROUP_GN_ITERS = int(os.environ.get("GROUP_GN_ITERS", "60"))
TRANSPORT_TR_DELTA = float(os.environ.get("TRANSPORT_TR_DELTA", "0.25"))
TR_SCALE = float(os.environ.get("TR_SCALE", "1.0"))
EQ_SNAPS = int(os.environ.get("EQ_SNAPS", "64"))
EQ_PERTURB = int(os.environ.get("EQ_PERTURB", "3"))
EQ_ROWS = int(os.environ.get("EQ_ROWS", "3072"))
EQ_CAND = int(os.environ.get("EQ_CAND_OFF", "4096"))
EQ_SEED = int(os.environ.get("EQ_SEED", str(mp.SEED + 20259)))
CAL_SEED = int(os.environ.get("CAL_SEED", "73129"))
CAL_FRAC = float(os.environ.get("CAL_FRAC", "0.2"))
RBF_LENGTHS = [float(v) for v in os.environ.get(
    "RBF_LENGTHS", "0.35,0.5,0.7,1.0,1.4,2.0,3.0,5.0").split(",")]
RBF_RIDGES = [float(v) for v in os.environ.get(
    "RBF_RIDGES", "1e-8,1e-6,1e-4,1e-2,1e-1,1").split(",")]
ARM_NAMES = [v for v in os.environ.get(
    "ARMS", "rbf_c64_q0,nearest_c64_q0,rbf_c64_q8,spectral_q8,spectral_q16"
).split(",") if v]
NATIVE_ARM_NAMES = [v for v in os.environ.get(
    "NATIVE_ARMS", "spectral_q8,param1_c64_q8,lmmean_c64_q0"
).split(",") if v]
BOOTSTRAP_REPS = int(os.environ.get("BOOTSTRAP_REPS", "1000" if SMOKE else "10000"))
BOOTSTRAP_SEED = int(os.environ.get("BOOTSTRAP_SEED", "821731"))
PAIRWISE_DIAGNOSTIC = bool(int(os.environ.get("PAIRWISE_DIAGNOSTIC", "1")))


@dataclass(frozen=True)
class Arm:
    name: str
    predictor: str
    coarse_n: int
    q: int
    rom_tau: float | None = None
    weak_M: int | None = None
    weak_m: int | None = None
    gn_budget: int | None = None
    base_q: int = 0


def parse_arm(name: str) -> Arm:
    if name.startswith("rbf_c"):
        left, qpart = name.split("_q")
        return Arm(name, "rbf", int(left[len("rbf_c"):]), int(qpart))
    if name.startswith("nearest_c"):
        left, qpart = name.split("_q")
        return Arm(name, "nearest", int(left[len("nearest_c"):]), int(qpart))
    lm_prefixes = {
        "lmmean_c": "lm_mean",
        "lmnearest_c": "lm_nearest",
        "lmtrmean_c": "lm_tr_mean",
        "lmtrnearest_c": "lm_tr_nearest",
    }
    prefix = next((p for p in lm_prefixes if name.startswith(p)), None)
    if prefix is not None:
        predictor = lm_prefixes[prefix]
        left, qpart = name.split("_q")
        ctext = left[len(prefix):]
        return Arm(name, predictor, -1 if ctext == "full" else int(ctext), int(qpart))
    if name.startswith("spectral_q"):
        return Arm(name, "none", 0, int(name[len("spectral_q"):]))
    if name.startswith("groupn_rt"):
        body, qpart = name.rsplit("_q", 1)
        prefix, ctext = body.rsplit("_c", 1)
        code = prefix[len("groupn_rt"):]
        tau_by_code = {"30": 0.30, "10": 0.10, "03": 0.03,
                       "01": 0.01, "001": 0.001}
        if code not in tau_by_code:
            raise ValueError(f"unknown GroupFiLM ROM-tolerance code {code!r}")
        return Arm(name, "group_nearest", -1 if ctext == "full" else int(ctext),
                   int(qpart), tau_by_code[code])
    if name.startswith("tp"):
        # tp0_m24_c64 / tp1_m32_c64: direct initializer or one LM attempt,
        # selected high-tail weak modes, and a q16 exact base in both cases.
        budget_text, mtext, ctext = name.split("_")
        budget = int(budget_text[2:])
        weak_M = int(mtext[1:])
        weak_m_by_M = {24: 96, 32: 128}
        if budget not in (0, 1) or weak_M not in weak_m_by_M:
            raise ValueError(f"unsupported bounded transport arm {name!r}")
        return Arm(name, "transport_tail", int(ctext[1:]), 0, None,
                   weak_M, weak_m_by_M[weak_M], budget, 16)
    if name.startswith("param1_c") or name.startswith("paramall_c"):
        prefix = "param1_c" if name.startswith("param1_c") else "paramall_c"
        predictor = "param_s1" if prefix == "param1_c" else "param_all"
        left, qpart = name.split("_q")
        ctext = left[len(prefix):]
        return Arm(name, predictor, -1 if ctext == "full" else int(ctext), int(qpart))
    raise ValueError(f"unknown arm {name!r}")


ARMS = [parse_arm(v) for v in ARM_NAMES]
if N_TIME > N_TEST:
    raise SystemExit("N_TIME must be <= N_TEST")
if not (0.0 < CAL_FRAC < 0.5):
    raise SystemExit("CAL_FRAC must lie in (0, 0.5)")
if not SMOKE and jax.default_backend() != "gpu":
    raise SystemExit("real feasibility runs require jax_backend=gpu")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def squared_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * x @ y.T,
        0.0,
    )


def fit_rbf_train_only(P: np.ndarray, Z: np.ndarray) -> tuple[dict, dict]:
    """Choose hyperparameters on a fixed split of training latents only.

    The PDE held-out set is not decoded, solved, or inspected during selection. The target is
    standardized latent MSE, averaged equally over latent coordinates.
    """
    t0 = time.time()
    rng = np.random.default_rng(CAL_SEED)
    perm = rng.permutation(len(P))
    n_cal = max(1, int(round(CAL_FRAC * len(P))))
    iva, itr = perm[:n_cal], perm[n_cal:]
    zmu = Z[itr].mean(axis=0)
    zsd = Z[itr].std(axis=0) + 1e-12
    Ys = (Z - zmu) / zsd
    dtr = squared_dist(P[itr], P[itr])
    dva = squared_dist(P[iva], P[itr])
    trials = []
    best = None
    eye = np.eye(len(itr))
    for length in RBF_LENGTHS:
        K = np.exp(-0.5 * dtr / length**2)
        Kv = np.exp(-0.5 * dva / length**2)
        for ridge in RBF_RIDGES:
            alpha = np.linalg.solve(K + ridge * eye, Ys[itr])
            pred = Kv @ alpha
            mse = float(np.mean((pred - Ys[iva]) ** 2))
            row = dict(length=length, ridge=ridge, calibration_standardized_latent_mse=mse)
            trials.append(row)
            if best is None or (mse, length, ridge) < (
                best["calibration_standardized_latent_mse"], best["length"], best["ridge"]
            ):
                best = row
    assert best is not None
    # Recompute standardization and fit on every training sample after selection.
    zmu = Z.mean(axis=0)
    zsd = Z.std(axis=0) + 1e-12
    Ys = (Z - zmu) / zsd
    dall = squared_dist(P, P)
    K = np.exp(-0.5 * dall / best["length"]**2)
    alpha = np.linalg.solve(K + best["ridge"] * np.eye(len(P)), Ys)
    fit = dict(
        P=jnp.asarray(P),
        alpha=jnp.asarray(alpha),
        zmu=jnp.asarray(zmu),
        zsd=jnp.asarray(zsd),
        length=float(best["length"]),
    )
    info = dict(
        selection="fixed train-only calibration split; no PDE held-out cases",
        cal_seed=CAL_SEED,
        cal_fraction=CAL_FRAC,
        n_fit_selection=int(len(itr)),
        n_calibration=int(len(iva)),
        trials=trials,
        selected=best,
        refit_train_standardized_latent_mse=float(np.mean((K @ alpha - Ys) ** 2)),
        fit_seconds=time.time() - t0,
    )
    return fit, info


def make_rbf_predictor(fit: dict):
    P, alpha = fit["P"], fit["alpha"]
    zmu, zsd, length = fit["zmu"], fit["zsd"], fit["length"]

    def predict(p):
        d2 = jnp.sum((P - p[None, :]) ** 2, axis=1)
        k = jnp.exp(-0.5 * d2 / length**2)
        return zmu + zsd * (k @ alpha)

    return predict


def paired_time(fn_a, fn_b, grade_a, grade_b, reps: int, warm: int):
    """Back-to-back timing with alternating order and telemetry from each timed output."""
    for _ in range(warm):
        jax.block_until_ready(fn_a()); jax.block_until_ready(fn_b())
    aa, bb, ga, gb = [], [], [], []

    def one(fn, grade):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - t0
        return elapsed, grade(out)

    for r in range(reps):
        if r % 2 == 0:
            t, g = one(fn_a, grade_a); aa.append(t); ga.append(g)
            t, g = one(fn_b, grade_b); bb.append(t); gb.append(g)
        else:
            t, g = one(fn_b, grade_b); bb.append(t); gb.append(g)
            t, g = one(fn_a, grade_a); aa.append(t); ga.append(g)
    return aa, bb, ga, gb


def grade_cg_output(out, F, Uref, op):
    x, iters, returned_rel_residual, flag = out
    recomputed = jnp.linalg.norm(op(x) - F) / jnp.maximum(jnp.linalg.norm(F), 1e-300)
    rel_l2 = jnp.linalg.norm(x - Uref) / jnp.maximum(jnp.linalg.norm(Uref), 1e-300)
    return dict(
        iterations=int(iters),
        returned_true_rel_residual=float(returned_rel_residual),
        recomputed_true_rel_residual=float(recomputed),
        rel_l2_vs_exact_dst=float(rel_l2),
        flag=int(flag),
    )


def repetition_summary(all_s, seed):
    """Case-median estimate, deterministic case bootstrap, and fixed outlier rule."""
    a = np.asarray(all_s, dtype=float)
    case_medians = np.median(a, axis=1)
    estimate = float(np.mean(case_medians))
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(case_medians), size=(BOOTSTRAP_REPS, len(case_medians)))
    boot = case_medians[sample].mean(axis=1)
    # Pre-registered clock/outlier diagnostic; no observations are removed.
    threshold = 1.5 * case_medians[:, None]
    return dict(
        mean_of_case_medians_s=estimate,
        case_medians_s=case_medians.tolist(),
        bootstrap_case_resample_reps=BOOTSTRAP_REPS,
        bootstrap_ci95_s=[float(np.quantile(boot, 0.025)),
                          float(np.quantile(boot, 0.975))],
        outlier_rule="timing repetition > 1.5 * its case median; retained in raw arrays",
        outlier_count=int(np.sum(a > threshold)),
        n_repetitions=int(a.size),
    )


def paired_ratio_ci(numerator_s, denominator_s, seed):
    """Case-resampled CI for the ratio of mean case medians on paired inputs."""
    num = np.median(np.asarray(numerator_s, dtype=float), axis=1)
    den = np.median(np.asarray(denominator_s, dtype=float), axis=1)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(num), size=(BOOTSTRAP_REPS, len(num)))
    ratios = num[sample].mean(axis=1) / den[sample].mean(axis=1)
    return [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]


def endpoint_bilinear(uc, out_n: int):
    """Coordinate-consistent bilinear prolongation for endpoint-including grids.

    Unlike pixel-center image resizing, target coordinate x=j/(out_n-1) maps exactly to
    coarse array index x*(coarse_n-1). Both physical endpoints are therefore invariant.
    """
    coarse_n = uc.shape[0]
    t = jnp.linspace(0.0, float(coarse_n - 1), out_n, dtype=uc.dtype)
    i0 = jnp.floor(t).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, coarse_n - 1)
    w = t - i0
    along_x = (1.0 - w[:, None]) * uc[i0, :] + w[:, None] * uc[i1, :]
    return (1.0 - w[None, :]) * along_x[:, i0] + w[None, :] * along_x[:, i1]


def check_endpoint_bilinear():
    """Affine functions must interpolate exactly on the physical grid."""
    cin, cout = 7, 23
    x = jnp.linspace(0.0, 1.0, cin)
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    u = 1.25 + 2.0 * X - 3.0 * Y
    xo = jnp.linspace(0.0, 1.0, cout)
    Xo, Yo = jnp.meshgrid(xo, xo, indexing="ij")
    err = float(jnp.max(jnp.abs(endpoint_bilinear(u, cout) - (1.25 + 2.0 * Xo - 3.0 * Yo))))
    if err > 5e-14:
        raise SystemExit(f"endpoint-bilinear affine self-check failed: {err:.3e}")
    return err


def dst1_ortho(x, axis):
    """Orthonormal DST-I through an odd FFT extension; self-inverse."""
    y = jnp.moveaxis(x, axis, -1)
    n = y.shape[-1]
    z = jnp.zeros(y.shape[:-1] + (1,), dtype=y.dtype)
    ext = jnp.concatenate([z, y, z, -y[..., ::-1]], axis=-1)
    raw = -0.5 * jnp.imag(jnp.fft.fft(ext, axis=-1)[..., 1:n + 1])
    out = jnp.sqrt(jnp.asarray(2.0 / (n + 1), y.dtype)) * raw
    return jnp.moveaxis(out, -1, axis)


def dst2_ortho(x):
    return dst1_ortho(dst1_ortho(x, 0), 1)


def check_dst():
    """The FFT DST must match the explicit orthonormal sine matrix."""
    n = 17
    rng = np.random.default_rng(119)
    x = jnp.asarray(rng.standard_normal((n, n)))
    p = np.arange(1, n + 1)
    S = jnp.asarray(np.sqrt(2.0 / (n + 1)) * np.sin(np.pi * np.outer(p, p) / (n + 1)))
    coeff_err = float(jnp.max(jnp.abs(dst2_ortho(x) - S.T @ x @ S)))
    inverse_err = float(jnp.max(jnp.abs(dst2_ortho(dst2_ortho(x)) - x)))
    if max(coeff_err, inverse_err) > 2e-13:
        raise SystemExit(f"FFT DST self-check failed: coeff={coeff_err:.3e} inv={inverse_err:.3e}")
    return dict(coeff_maxabs=coeff_err, inverse_maxabs=inverse_err)


def make_lm_obj_tr_jit(dec, K, pts, wq, PhiT, Wl, budget, obj_rel, delta):
    """Audited objective-reduction LM plus one training-cloud trust-region gate.

    This is `wsf_poisson.make_lm_obj_jit` with only `||dz|| <= delta` added to the
    finite-step gate. A rejected long step follows the unchanged `lambda *= 10` path.
    `delta=inf` is asserted numerically equivalent to the imported reference in `main`.
    """
    pts = jnp.asarray(pts); wq = jnp.asarray(wq)
    PhiT = jnp.asarray(PhiT); Wl = jnp.asarray(Wl)

    def r_of(z, f_m):
        return Wl * (PhiT @ (wq * dec(z, pts))) - f_m

    rJ = lambda z, f_m: (r_of(z, f_m), jax.jacfwd(r_of)(z, f_m))
    rn_fn = lambda z, f_m: jnp.linalg.norm(r_of(z, f_m))

    def lm(z0, f_m):
        r0, J0 = rJ(z0, f_m)
        v0 = jnp.linalg.norm(r0)
        obj_target = obj_rel * v0
        init = (z0, J0, r0, v0, jnp.asarray(1e-6, F64), jnp.int32(0), jnp.int32(0),
                jnp.int32(1), jnp.where(jnp.isfinite(v0), jnp.int32(0), jnp.int32(5)))

        def cond(s):
            return (s[8] == 0) & (s[5] < budget)

        def body(s):
            z, J, r, val, lam, att, acc, nJ, _ = s
            H = J.T @ J; g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            within = jnp.linalg.norm(dz) <= delta
            valid = finite & within
            z_new = z + jnp.where(valid, dz, 0.0)
            v_new = jnp.where(valid, rn_fn(z_new, f_m), jnp.inf)
            accept = valid & jnp.isfinite(v_new) & (v_new < val)
            rel_dec = jnp.where(accept, (val - v_new) / (jnp.abs(val) + 1e-300), 1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new, f_m), lambda: (r, J))
            z = jnp.where(accept, z_new, z); val = jnp.where(accept, v_new, val)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32); nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(accept & (obj_rel > 0) & (val <= obj_target), jnp.int32(6),
                       jnp.where(accept & ((rel_dec < 1e-12) | (step < 1e-13)), jnp.int32(1),
                        jnp.where((~accept) & (lam >= 1e12), jnp.int32(3), jnp.int32(0))))
            return (z, J2, r2, val, lam, att + 1, acc, nJ, reason)

        z, J, r, val, lam, att, acc, nJ, reason = jax.lax.while_loop(cond, body, init)
        return z, val, v0, nJ, acc, att, reason

    return jax.jit(lm)


def continuum_modes(grid, M):
    """Continuous sine-mode indices selected exactly as the audited off-grid weak form."""
    kk = np.arange(1, grid.N - 1)
    II, JJ = np.meshgrid(kk, kk, indexing="ij")
    lam_all = (np.pi**2 * (II**2 + JJ**2)).reshape(-1)
    m_eff = min(M, lam_all.size)
    keep = lam_all <= np.sort(lam_all)[m_eff - 1]
    return II.reshape(-1)[keep], JJ.reshape(-1)[keep], lam_all[keep]


def continuum_tail_modes(grid, M, base_q):
    """Lowest complete eigenshells not already spanned by a q-by-q exact base."""
    kk = np.arange(1, grid.N - 1)
    II, JJ = np.meshgrid(kk, kk, indexing="ij")
    eligible = ~((II <= base_q) & (JJ <= base_q))
    I, J = II[eligible], JJ[eligible]
    lam = np.pi**2 * (I**2 + J**2)
    m_eff = min(M, len(lam))
    keep = lam <= np.sort(lam)[m_eff - 1]
    return I[keep], J[keep], lam[keep]


def make_separable_source_projector(grid, M, alpha=1.0, modes=None):
    """O(N^2 sqrt(M)) source projection with no captured O(M N^2) constant."""
    I, J, lam = continuum_modes(grid, M) if modes is None else modes
    maxk = int(max(np.max(I), np.max(J)))
    x = np.arange(1, grid.N - 1) / (grid.N - 1)
    S = jnp.asarray(np.sin(np.pi * np.outer(x, np.arange(1, maxk + 1))))
    Ii, Jj = jnp.asarray(I - 1), jnp.asarray(J - 1)
    weight = jnp.asarray(2.0 * grid.dx**2 * lam**(-alpha))

    def apply(F):
        C = S.T @ F @ S
        return C[Ii, Jj] * weight

    return jax.jit(apply), dict(n_modes=len(I), max_1d_mode=maxk,
                                captured_table_shape=list(S.shape))


def eq_fit_streamed(dec, grid, Ztr, K, M, m, modes=None, full_dec=None):
    """The audited off-grid NNLS-EQ fit with streamed/separable full-grid targets.

    It preserves the fixed RNG stream, snapshots, row normalization, row subset, capped NNLS,
    and final refit of `fu_eq.eq_fit`. It avoids materializing the N-grid decoder fields and
    the (N^2,M) mode table, which would retain ~2 GiB + ~510 MiB at N=1024.
    """
    t0 = time.time()
    rng = np.random.default_rng(EQ_SEED)
    idx = rng.choice(len(Ztr), size=min(EQ_SNAPS, len(Ztr)), replace=False)
    cand_np = np.random.default_rng(mp.SEED + 12345).uniform(0.0, 1.0, size=(EQ_CAND, 2))
    I, J, lam = continuum_modes(grid, M) if modes is None else modes
    Phi = (2.0 * np.sin(np.pi * cand_np[:, 0, None] * I[None, :])
           * np.sin(np.pi * cand_np[:, 1, None] * J[None, :]))
    maxk = int(max(np.max(I), np.max(J)))
    x = np.arange(1, grid.N - 1) / (grid.N - 1)
    S = jnp.asarray(np.sin(np.pi * np.outer(x, np.arange(1, maxk + 1))))
    cand = jnp.asarray(cand_np)
    snap_fn = jax.jit(lambda z: dec(z, cand))
    full_fn = (jax.jit(lambda z: dec(z, grid.coords_int).reshape(grid.n_i, grid.n_i))
               if full_dec is None else jax.jit(full_dec))
    snaps, targets = [], []
    for i in idx:
        z = jnp.asarray(Ztr[i])
        zlist = [z] + [z + 0.05 * jnp.asarray(rng.standard_normal(K))
                       for _ in range(EQ_PERTURB)]
        for zz in zlist:
            snaps.append(np.asarray(snap_fn(zz)))
            U = full_fn(zz)
            C = np.asarray(S.T @ U @ S)
            targets.append(2.0 * grid.dx**2 * C[I - 1, J - 1])
    R = np.stack(snaps)
    T = np.stack(targets)
    # Full-candidate row norms without forming (snap, mode, candidate).
    row_sc = np.sqrt(np.maximum((R * R) @ (Phi * Phi), 0.0)) + 1e-300
    n_snap, n_modes = T.shape
    rows = rng.choice(n_snap * n_modes, size=min(n_snap * n_modes, EQ_ROWS), replace=False)
    si, mi = rows // n_modes, rows % n_modes
    Gsel = (R[si, :] * Phi[:, mi].T) / row_sc[si, mi, None]
    bsel = T[si, mi] / row_sc[si, mi]
    wts, rnorm, _ = pc.nnls_capped(Gsel, bsel, max_support=m)
    supp = np.nonzero(wts > 0)[0]
    if len(supp) >= m:
        keep = supp[np.argsort(-wts[supp])[:m]]; padded = 0
    else:
        rest = np.setdiff1d(np.arange(EQ_CAND), supp)
        pad = rest[np.argsort(-np.abs(R).mean(0)[rest])[:m - len(supp)]]
        keep = np.concatenate([supp, pad]); padded = len(pad)
    Gk = np.einsum("sp,pm->smp", R[:, keep], Phi[keep, :])
    Gk = (Gk / row_sc[:, :, None]).reshape(-1, len(keep))
    b = (T / row_sc).reshape(-1)
    wq, _, _ = pc.nnls_capped(Gk, b, max_support=len(keep))
    wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
    res = Gk @ wq - b
    rel_rows = np.abs(res) / (np.abs(b) + 1e-300)
    info = dict(
        implementation="streamed_separable_exact_equivalent",
        M=M,
        n_modes=n_modes,
        m=int(len(keep)),
        pool="offgrid",
        grid_N=grid.N,
        support=int(len(supp)),
        padded=int(padded),
        rnorm_capped=float(rnorm),
        rnorm_final=float(np.linalg.norm(res)),
        b_norm=float(np.linalg.norm(b)),
        rel_fit=float(np.linalg.norm(res) / np.linalg.norm(b)),
        row_rel_median=float(np.median(rel_rows)),
        row_rel_p95=float(np.quantile(rel_rows, 0.95)),
        row_rel_max=float(np.max(rel_rows)),
        n_rows=int(len(rows)),
        n_cand=EQ_CAND,
        n_snapshots=n_snap,
        full_grid_peak_retained_fields=1,
        full_grid_mode_table_materialized=False,
        max_1d_mode=maxk,
        seconds=time.time() - t0,
    )
    print(f"  STREAMED NNLS-EQ M={M} m={m} @N={grid.N}: support {len(supp)} "
          f"(+{padded}) rel fit {info['rel_fit']:.2e} [{info['seconds']:.0f}s]", flush=True)
    return cand_np[keep], wq, info


def source_cases(n: int):
    if TEST_SEED == mp.SEED:
        cx, cy, w, a, p = mp.sample_params(seed=mp.SEED, m=mp.N_TRAIN + N_TEST)
        sl = slice(mp.N_TRAIN, mp.N_TRAIN + N_TEST)
    else:
        cx, cy, w, a, p = mp.sample_params(seed=TEST_SEED, m=N_TEST)
        sl = slice(0, N_TEST)
    Fs = np.stack([mp.source_interior(n, cx[i], cy[i], w[i], a[i])
                   for i in range(*sl.indices(len(cx)))])
    return Fs, np.asarray(p[sl])


def norm_metrics(x0, uref, op, F) -> dict:
    e = x0 - uref

    def anorm(v):
        return float(jnp.sqrt(jnp.maximum(jnp.sum(v * op(v)), 0.0)))

    return dict(
        guess_rel_l2=float(jnp.linalg.norm(e) / jnp.maximum(jnp.linalg.norm(uref), 1e-300)),
        guess_a_norm_ratio=anorm(e) / max(anorm(uref), 1e-300),
        guess_true_rel_residual=float(
            jnp.linalg.norm(op(x0) - F) / jnp.maximum(jnp.linalg.norm(F), 1e-300)
        ),
    )


def bc_factor(xy):
    """The exact hard-Dirichlet multiplier used to train both Poisson decoders."""
    return 16.0 * xy[:, 0] * (1.0 - xy[:, 0]) * xy[:, 1] * (1.0 - xy[:, 1])


def main():
    prov = wu.provenance(HERE)
    prov["source_sha256"] = wu.source_hashes(HERE, ("*.py",))
    print(
        f"jax_backend={jax.default_backend()} device={jax.devices()[0]} "
        f"NS={NS} taus={FOM_TAUS} arms={ARM_NAMES} test_seed={TEST_SEED} "
        f"smoke={int(SMOKE)} reps={TIME_REPS} warm={TIME_WARM}",
        flush=True,
    )
    interp_affine_err = check_endpoint_bilinear()
    dst_check = check_dst()
    d, cfg, stages, Ztr, hard_bc = pc.load_pkl(PKL)
    if not hard_bc:
        raise SystemExit("this harness requires the hard-BC checkpoint")
    if cfg["K_LAT"] != 8:
        raise SystemExit(f"pre-registered feasibility requires K=8, got {cfg['K_LAT']}")
    dec = pc.make_decoder(stages[:1], hard_bc=True)

    group_info = None
    group_model = None
    if any(a.predictor == "group_nearest" for a in ARMS):
        if not GROUP_PKL:
            raise SystemExit("a groupn_* arm requires GROUP_PKL")
        import nda_arch as nda

        gd = pickle.load(open(GROUP_PKL, "rb"))
        gcfg = gd["config"]
        gst = gd["stages"][0]
        gdcfg = gst["decoder_config"]
        if not bool(gcfg.get("hard_bc", 0)):
            raise SystemExit("GroupFiLM checkpoint must enforce hard boundary conditions")
        if gdcfg["name"] != "groupfilm":
            raise SystemExit(f"expected groupfilm checkpoint, got {gdcfg['name']!r}")
        if int(gcfg["n_train"]) != mp.N_TRAIN:
            raise SystemExit("GroupFiLM checkpoint training set does not match source lookup")
        gparams = jax.tree_util.tree_map(jnp.asarray, gst["params"])
        gz = jnp.asarray(gd["z_tr"])
        gk = int(gcfg["K_LAT"])
        gn_freq = int(gst["n_freq"])
        geps = float(gst["eps"])
        gz_ff = int(gst.get("z_ff", gdcfg.get("z_ff", 0)))

        def group_raw_dec(z, xy):
            return geps * bc_factor(xy) * nda.apply(
                gparams, z, xy, gn_freq, gdcfg, gz_ff
            )

        group_model = dict(
            module=nda,
            params=gparams,
            Z=gz,
            K=gk,
            n_freq=gn_freq,
            eps=geps,
            z_ff=gz_ff,
            decoder_config=gdcfg,
            raw_dec=group_raw_dec,
        )
        group_info = dict(
            pkl=os.path.basename(GROUP_PKL),
            pkl_sha256=sha256(GROUP_PKL),
            architecture_source=os.path.basename(nda.__file__),
            architecture_source_sha256=sha256(nda.__file__),
            config=gcfg,
            n_params=nda.parameter_count(gparams),
            cached_coordinate_stem=True,
            fixed_output_basis=False,
        )

    transport_info = None
    transport_model = None
    if any(a.predictor == "transport_tail" for a in ARMS):
        if not TRANSPORT_PKL:
            raise SystemExit("a tp* arm requires TRANSPORT_PKL")
        import transport_arch as ta

        td = pickle.load(open(TRANSPORT_PKL, "rb"))
        tcfg = td["config"]
        if tcfg["architecture"]["name"] != "separable_transport_tail":
            raise SystemExit("unexpected transported-tail architecture")
        if int(tcfg["q_base"]) != 16 or int(tcfg["K"]) != 6:
            raise SystemExit("transport checkpoint must be the pre-registered q16/K6 model")
        tparams = jax.tree_util.tree_map(jnp.asarray, td["params"])
        tinit = jax.tree_util.tree_map(jnp.asarray, td["initializer"])
        transport_model = dict(
            module=ta, params=tparams, eps=float(td["eps"]),
            initializer=tinit, config=tcfg, Z=jnp.asarray(td["z_train"]),
        )
        transport_info = dict(
            pkl=os.path.basename(TRANSPORT_PKL), pkl_sha256=sha256(TRANSPORT_PKL),
            architecture_source=os.path.basename(ta.__file__),
            architecture_source_sha256=sha256(ta.__file__),
            config=tcfg, selected_step=td["selected_step"],
            selection_key=td["selection_key"],
            initializer_info=td["initializer_info"],
            n_params=ta.parameter_count(tparams), fixed_output_basis=False,
            full_grid_complexity="O(R*N^2 + width*R*N)",
        )

    param_info = None
    param_stages = None
    if any(a.predictor.startswith("param_") for a in ARMS):
        if not PARAM_PKL:
            raise SystemExit("a param_* arm requires PARAM_PKL")
        pd = pickle.load(open(PARAM_PKL, "rb"))
        pcfg = pd["config"]
        for key in ("N", "n_train", "n_val", "seed", "hidden", "n_layers"):
            if pcfg[key] != mp.CONFIG[key]:
                raise SystemExit(f"PARAM_PKL config mismatch {key}: {pcfg[key]} != {mp.CONFIG[key]}")
        param_stages = mp.stages_from_np(pd["stages"])
        param_info = dict(
            pkl=os.path.basename(PARAM_PKL),
            pkl_sha256=sha256(PARAM_PKL),
            config=pcfg,
            n_stages=len(param_stages),
        )

    # Predictor fitting sees the canonical training parameters and trained latents only.
    _, _, _, _, P_all = mp.sample_params(seed=mp.SEED, m=mp.N_TRAIN)
    P_train = np.asarray(P_all[:mp.N_TRAIN])
    rbf_fit, rbf_info = fit_rbf_train_only(P_train, np.asarray(Ztr))
    rbf_predict = make_rbf_predictor(rbf_fit)
    nearest_P = jnp.asarray(P_train)
    nearest_Z = jnp.asarray(Ztr)
    z_mean = jnp.asarray(np.asarray(Ztr).mean(axis=0))

    report = dict(
        complete=False,
        config=dict(
            pde="poisson2d",
            pkl=os.path.basename(PKL),
            pkl_sha256=sha256(PKL),
            pkl_config=cfg,
            hard_bc=int(hard_bc),
            ns=NS,
            fom_taus=FOM_TAUS,
            n_test=N_TEST,
            n_time=N_TIME,
            time_reps=TIME_REPS,
            time_warm=TIME_WARM,
            burn_s=BURN_S,
            test_seed=TEST_SEED,
            canonical_training_seed=mp.SEED,
            M=M_MODES,
            m=MQ,
            gn_iters=GN_ITERS,
            lm_rom_tau=LM_ROM_TAU,
            group_M=GROUP_M,
            group_m=GROUP_MQ,
            group_gn_iters=GROUP_GN_ITERS,
            transport_trust_delta=TRANSPORT_TR_DELTA,
            trust_region_scale=TR_SCALE,
            arms=[a.__dict__ for a in ARMS],
            native_cg_sensitivity_arms=NATIVE_ARM_NAMES,
            preconditioned_baseline=(
                "native jax.scipy CG with exact diagonal/Jacobi inverse; for the constant-"
                "coefficient Poisson stencil this is a scalar and should not change Krylov work"
            ),
            bootstrap_reps=BOOTSTRAP_REPS,
            bootstrap_seed=BOOTSTRAP_SEED,
            timing_outlier_rule=(
                "repetition > 1.5 * its case median; diagnose only, never discard"
            ),
            pairwise_diagnostic=PAIRWISE_DIAGNOSTIC,
            cg_maxiter=CG_MAXITER,
            matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "unset"),
            dtype="f64",
            prolongation="endpoint-aligned separable bilinear",
            prolongation_affine_selfcheck_maxabs=interp_affine_err,
            fft_dst_selfcheck=dst_check,
        ),
        provenance=prov,
        rbf_train_only=rbf_info,
        parameter_aligned_checkpoint=param_info,
        nonlinear_groupfilm_checkpoint=group_info,
        transported_tail_checkpoint=transport_info,
        rows=[],
        mesh_checks=[],
    )

    def save():
        with open(OUT, "w") as f:
            json.dump(report, f, indent=1, default=float, allow_nan=False)

    save()
    for n in NS:
        mesh_t0 = time.time()
        grid = pc.Grid(n)
        ni = grid.n_i
        op = lambda u: mp.neg_lap_interior(u, n)
        cg = wu.make_cg(op, maxiter=CG_MAXITER)
        zero = jnp.zeros((ni, ni), dtype=F64)
        Fs_np, params_np = source_cases(n)
        Fs = [jnp.asarray(v) for v in Fs_np]
        params = [jnp.asarray(v) for v in params_np]

        # Exact O(N^2 log N) FFT-DST inverse of the same Dirichlet FD operator.
        direct = jax.jit(lambda F: dst2_ortho(dst2_ortho(F) / grid.lam))
        # Dense sine diagonalization is a distinct O(N^3) direct baseline.  At
        # moderate N its GEMMs can beat the FFT implementation on a GPU, but its
        # attained f64 residual must be reported rather than assumed exact.
        Sfull = grid.S
        dense_direct = jax.jit(
            lambda F: Sfull @ ((Sfull.T @ F @ Sfull) / grid.lam) @ Sfull.T
        )
        Uref = [direct(F) for F in Fs]
        d_res = [float(jnp.linalg.norm(op(Uref[i]) - Fs[i]) / jnp.linalg.norm(Fs[i]))
                 for i in range(N_TEST)]
        Udense = [dense_direct(F) for F in Fs]
        dense_res = [
            float(jnp.linalg.norm(op(Udense[i]) - Fs[i]) / jnp.linalg.norm(Fs[i]))
            for i in range(N_TEST)
        ]
        dense_vs_fft = [
            float(jnp.linalg.norm(Udense[i] - Uref[i]) / jnp.linalg.norm(Uref[i]))
            for i in range(N_TEST)
        ]

        if BURN_S > 0:
            burn_n = wu.gpu_burn(
                lambda: cg(Fs[0], zero, FOM_TAUS[0])[0].block_until_ready(), BURN_S
            )
        else:
            burn_n = 0

        direct_reps = []
        for i in range(N_TIME):
            _, reps = wu.time_fn(
                lambda ii=i: direct(Fs[ii]).block_until_ready(), TIME_REPS, TIME_WARM
            )
            direct_reps.append(reps)

        mesh_check = dict(
            N=n,
            n_dof=ni**2,
            test_parameters=params_np.tolist(),
            exact_direct_true_rel_residual_max=float(max(d_res)),
            dense_dst_direct_true_rel_residual_max=float(max(dense_res)),
            dense_dst_direct_rel_l2_vs_fft_max=float(max(dense_vs_fft)),
            pre_eq_diagnostic_exact_direct_all_s=direct_reps,
            pre_eq_diagnostic_exact_direct_ms=(
                float(np.mean([np.median(v) for v in direct_reps])) * 1e3
            ),
            burn_iterations=burn_n,
        )
        report["mesh_checks"].append(mesh_check)
        save()

        # Predictor-only costs are explicitly recorded; the total callable below includes them.
        rbf_jit = jax.jit(rbf_predict)

        def nearest_predict(p):
            idx = jnp.argmin(jnp.sum((nearest_P - p[None, :]) ** 2, axis=1))
            return nearest_Z[idx]

        nearest_jit = jax.jit(nearest_predict)
        lookup_fns = [("rbf", rbf_jit), ("nearest", nearest_jit)]
        if group_model is not None:
            group_nearest_jit = jax.jit(lambda p: group_model["Z"][
                jnp.argmin(jnp.sum((nearest_P - p[None, :]) ** 2, axis=1))
            ])
            lookup_fns.append(("group_nearest", group_nearest_jit))
        predictor_reps = {key: [] for key, _ in lookup_fns}
        for key, fn in lookup_fns:
            for i in range(N_TIME):
                _, reps = wu.time_fn(
                    lambda ii=i, ff=fn: ff(params[ii]).block_until_ready(),
                    TIME_REPS,
                    TIME_WARM,
                )
                predictor_reps[key].append(reps)
        mesh_check["predictor_all_s"] = predictor_reps

        # One offline EQ fit per mesh for every weak-LM arm. Its cost is recorded but never
        # charged per query. The online source projection, LM, lookup, and decode are all inside
        # the arm's timed construction callable.
        have_lm = any(a.predictor.startswith("lm_") for a in ARMS)
        if have_lm:
            spec = dict(kind="weak", alpha=1.0, M=M_MODES)
            pts, wq, eq_info = eq_fit_streamed(dec, grid, Ztr, cfg["K_LAT"], M_MODES, MQ)
            if SMOKE and bool(int(os.environ.get("COMPARE_LEGACY_EQ", "0"))):
                old_pts, old_wq, old_info = legacy_eq_fit(
                    dec, grid, Ztr, cfg["K_LAT"], M_MODES, MQ, "offgrid"
                )
                pts_diff = float(np.max(np.abs(np.asarray(pts) - np.asarray(old_pts))))
                wq_diff = float(np.max(np.abs(np.asarray(wq) - np.asarray(old_wq))))
                if pts_diff != 0.0 or wq_diff > 1e-10 * (1.0 + float(np.max(np.abs(old_wq)))):
                    raise SystemExit(f"streamed EQ mismatch: pts={pts_diff:.3e} wq={wq_diff:.3e}")
                eq_info["legacy_smoke_comparison"] = dict(
                    pts_maxabs=pts_diff,
                    weights_maxabs=wq_diff,
                    legacy_rel_fit=old_info["rel_fit"],
                )
            PhiT, Wl = pc.colloc_mode_table(grid, spec, "offgrid", pts)
            pre_apply, pre_info = make_separable_source_projector(grid, M_MODES, alpha=1.0)
            lm_base = make_lm_obj_jit(
                dec, cfg["K_LAT"], pts, wq, PhiT, Wl, GN_ITERS, LM_ROM_TAU
            )
            train_radius = float(np.max(np.linalg.norm(
                np.asarray(Ztr) - np.asarray(Ztr).mean(axis=0), axis=1
            )))
            lm_tr = make_lm_obj_tr_jit(
                dec, cfg["K_LAT"], pts, wq, PhiT, Wl, GN_ITERS, LM_ROM_TAU,
                TR_SCALE * train_radius,
            )
            lm_inf = make_lm_obj_tr_jit(
                dec, cfg["K_LAT"], pts, wq, PhiT, Wl, GN_ITERS, LM_ROM_TAU, jnp.inf
            )
            f0m = pre_apply(Fs[0])
            zref = lm_base(z_mean, f0m)[0]
            zinf = lm_inf(z_mean, f0m)[0]
            lm_equiv = float(jnp.max(jnp.abs(zref - zinf)))
            if lm_equiv > 1e-12 * (1.0 + float(jnp.linalg.norm(zref))):
                raise SystemExit(f"N={n}: trust LM at delta=inf != audited LM ({lm_equiv:.3e})")
            mesh_check["eq_info"] = eq_info
            mesh_check["source_projector"] = pre_info
            if n <= 256:
                dense_ref = pc.weak_source_term(grid, spec, "offgrid", np.asarray(Fs[0]))
                pre_diff = float(jnp.max(jnp.abs(pre_apply(Fs[0]) - dense_ref)))
                if pre_diff > 1e-11:
                    raise SystemExit(f"N={n}: separable source projector mismatch {pre_diff:.3e}")
                mesh_check["source_projector_vs_dense_maxabs"] = pre_diff
            mesh_check["training_latent_radius"] = train_radius
            mesh_check["trust_delta"] = TR_SCALE * train_radius
            mesh_check["trust_inf_vs_audited_lm_maxabs"] = lm_equiv
        else:
            lm_base = lm_tr = None

        # The compact GroupFiLM is a genuine nonlinear coordinate manifold.  Its
        # coordinate-only stem is cached separately at EQ and decode points; only the
        # small latent modulation path remains online.  Hyper-reduction is refit for
        # every N, matching the repository's method contract.
        have_group = any(a.predictor == "group_nearest" for a in ARMS)
        group_solvers = {}
        group_field_decoders = {}
        if have_group:
            assert group_model is not None
            nda = group_model["module"]
            gparams = group_model["params"]
            gz = group_model["Z"]
            gk = group_model["K"]
            gdcfg = group_model["decoder_config"]
            gn_freq = group_model["n_freq"]
            geps = group_model["eps"]
            gz_ff = group_model["z_ff"]
            graw = group_model["raw_dec"]
            gpts, gwq, geq_info = eq_fit_streamed(
                graw, grid, gz, gk, GROUP_M, GROUP_MQ
            )
            gpts_j = jnp.asarray(gpts)
            gh_eq = nda.prepare_coords(gparams, gpts_j, gn_freq, gdcfg)
            gb_eq = bc_factor(gpts_j)

            def group_eq_dec(z, ignored_xy):
                del ignored_xy
                return geps * gb_eq * nda.apply_prepared(
                    gparams, z, gh_eq, gdcfg, gz_ff
                )

            gspec = dict(kind="weak", alpha=1.0, M=GROUP_M)
            gPhiT, gWl = pc.colloc_mode_table(grid, gspec, "offgrid", gpts)
            gpre_apply, gpre_info = make_separable_source_projector(
                grid, GROUP_M, alpha=1.0
            )
            for rt in sorted({a.rom_tau for a in ARMS
                              if a.predictor == "group_nearest"}):
                group_solvers[rt] = make_lm_obj_jit(
                    group_eq_dec, gk, gpts, gwq, gPhiT, gWl,
                    GROUP_GN_ITERS, rt,
                )
            for cN in sorted({n if a.coarse_n < 0 else a.coarse_n for a in ARMS
                              if a.predictor == "group_nearest"}):
                cg_grid = grid if cN == n else pc.Grid(cN)
                cxy = cg_grid.coords
                ch = nda.prepare_coords(gparams, cxy, gn_freq, gdcfg)
                cb = bc_factor(cxy)

                def cached_group_dec(z, ignored_xy, *, ch=ch, cb=cb):
                    del ignored_xy
                    return geps * cb * nda.apply_prepared(
                        gparams, z, ch, gdcfg, gz_ff
                    )

                group_field_decoders[cN] = cached_group_dec
            mesh_check["group_eq_info"] = geq_info
            mesh_check["group_source_projector"] = gpre_info
            mesh_check["group_cached_stems"] = {
                str(cN): list(group_field_decoders[cN](gz[0], jnp.empty((0, 2))).shape)
                for cN in group_field_decoders
            }
            if n <= 256:
                gdense_ref = pc.weak_source_term(grid, gspec, "offgrid", np.asarray(Fs[0]))
                gpre_diff = float(jnp.max(jnp.abs(gpre_apply(Fs[0]) - gdense_ref)))
                if gpre_diff > 1e-11:
                    raise SystemExit(
                        f"N={n}: GroupFiLM separable source projector mismatch {gpre_diff:.3e}"
                    )
                mesh_check["group_source_projector_vs_dense_maxabs"] = gpre_diff
        else:
            gpre_apply = None

        have_transport = any(a.predictor == "transport_tail" for a in ARMS)
        transport_solvers = {}
        transport_projectors = {}
        transport_field_decoders = {}
        if have_transport:
            assert transport_model is not None
            ta = transport_model["module"]
            tparams = transport_model["params"]
            teps = transport_model["eps"]
            tZ = transport_model["Z"]
            tx_full = jnp.linspace(0.0, 1.0, n, dtype=F64)

            def transport_raw_dec(z, xy):
                return ta.apply_points(tparams, z, xy, teps)

            def transport_full_int(z):
                return ta.apply_grid(tparams, z, tx_full, teps)[1:-1, 1:-1]

            mesh_check["transport_weak_configs"] = {}
            for weak_M in sorted({a.weak_M for a in ARMS
                                  if a.predictor == "transport_tail"}):
                arm0 = next(a for a in ARMS if a.predictor == "transport_tail"
                            and a.weak_M == weak_M)
                modes = continuum_tail_modes(grid, weak_M, arm0.base_q)
                tpts, twq, teq_info = eq_fit_streamed(
                    transport_raw_dec, grid, tZ, 6, weak_M, arm0.weak_m,
                    modes=modes, full_dec=transport_full_int,
                )
                I, J, tlam = modes
                tPhiT = jnp.asarray(
                    2.0 * np.sin(np.pi * np.asarray(tpts)[:, 0, None] * I[None, :])
                    * np.sin(np.pi * np.asarray(tpts)[:, 1, None] * J[None, :])
                ).T
                tWl = jnp.asarray(tlam ** -1.0)
                tpre, tpre_info = make_separable_source_projector(
                    grid, weak_M, alpha=1.0, modes=modes
                )
                transport_projectors[weak_M] = tpre
                if any(a.gn_budget == 1 for a in ARMS
                       if a.predictor == "transport_tail" and a.weak_M == weak_M):
                    transport_solvers[weak_M] = make_lm_obj_tr_jit(
                        transport_raw_dec, 6, tpts, twq, tPhiT, tWl, 1, 0.0,
                        TRANSPORT_TR_DELTA,
                    )
                mesh_check["transport_weak_configs"][str(weak_M)] = dict(
                    base_q=arm0.base_q, selected_mode_count=len(I),
                    selected_mode_i=I.tolist(), selected_mode_j=J.tolist(),
                    eq_info=teq_info, source_projector=tpre_info,
                )
            for cN in sorted({a.coarse_n for a in ARMS
                              if a.predictor == "transport_tail"}):
                cx = jnp.linspace(0.0, 1.0, cN, dtype=F64)

                def transport_grid_dec(z, ignored_xy, *, cx=cx):
                    del ignored_xy
                    return ta.apply_grid(tparams, z, cx, teps).reshape(-1)

                transport_field_decoders[cN] = transport_grid_dec

        timing_registry = {}
        def make_baseline(tau):
            return jax.jit(lambda p, F: cg(F, zero, tau))

        baseline_registry = {tau: make_baseline(tau) for tau in FOM_TAUS}
        baseline_validation = {
            tau: [baseline_registry[tau](params[i], Fs[i]) for i in range(N_TEST)]
            for tau in FOM_TAUS
        }
        guess_registry = {}
        if have_transport:
            tinit = transport_model["initializer"]
            tSq = grid.S[:, :16]
            tlamq = grid.lam[:16, :16]

            def transport_coeff(F):
                return tSq.T @ F @ tSq

            def transport_z_init(F):
                c = (transport_coeff(F) / (n - 1)).reshape(-1)
                xs = (c - tinit["mean"]) / tinit["scale"]
                return jnp.concatenate([jnp.ones((1,), dtype=F64), xs]) @ tinit["weights"]

            def transport_base(F):
                return tSq @ (transport_coeff(F) / tlamq) @ tSq.T
        for arm in ARMS:
            q = min(arm.q, ni)
            Sq = grid.S[:, :q] if q else None
            lamq = grid.lam[:q, :q] if q else None

            if arm.predictor != "none":
                cgrid = grid if arm.coarse_n < 0 else pc.Grid(arm.coarse_n)
                cN = n if arm.coarse_n < 0 else arm.coarse_n
                field_decoder = None
                latent_decoder = dec
                additive_base = lambda F: jnp.zeros_like(F)
                if arm.predictor == "rbf":
                    predictor = lambda p, F: rbf_predict(p)
                elif arm.predictor == "nearest":
                    predictor = lambda p, F: nearest_predict(p)
                elif arm.predictor in ("lm_mean", "lm_nearest", "lm_tr_mean", "lm_tr_nearest"):
                    if lm_base is None:
                        raise AssertionError("LM arm without initialized solver")

                    def predictor(p, F, *, mode=arm.predictor):
                        z0 = z_mean
                        if mode.endswith("nearest"):
                            idx = jnp.argmin(jnp.sum((nearest_P - p[None, :]) ** 2, axis=1))
                            z0 = nearest_Z[idx]
                        solver = lm_tr if mode.startswith("lm_tr_") else lm_base
                        return solver(z0, pre_apply(F))[0]
                elif arm.predictor in ("param_s1", "param_all"):
                    if param_stages is None:
                        raise AssertionError("parameter-aligned arm without checkpoint")
                    stages_use = param_stages[:1] if arm.predictor == "param_s1" else param_stages
                    field_decoder = lambda p, xy: mp.combined_apply(stages_use, p, xy)
                    predictor = None
                elif arm.predictor == "group_nearest":
                    if group_model is None or gpre_apply is None:
                        raise AssertionError("GroupFiLM arm without initialized solver")
                    group_solver = group_solvers[arm.rom_tau]
                    latent_decoder = group_field_decoders[cN]

                    def predictor(p, F, *, solver=group_solver):
                        idx = jnp.argmin(jnp.sum((nearest_P - p[None, :]) ** 2, axis=1))
                        return solver(group_model["Z"][idx], gpre_apply(F))[0]
                elif arm.predictor == "transport_tail":
                    latent_decoder = transport_field_decoders[cN]
                    additive_base = transport_base
                    tsolver = transport_solvers.get(arm.weak_M)
                    tproject = transport_projectors[arm.weak_M]

                    def predictor(p, F, *, budget=arm.gn_budget, solver=tsolver,
                                  project=tproject):
                        del p
                        z0 = transport_z_init(F)
                        return z0 if budget == 0 else solver(z0, project(F))[0]
                else:
                    raise AssertionError(arm.predictor)

                def learned_full_f(p, F, *, cgrid=cgrid, cN=cN, predictor=predictor,
                                   field_decoder=field_decoder,
                                   latent_decoder=latent_decoder):
                    if field_decoder is None:
                        z = predictor(p, F)
                        uc = latent_decoder(z, cgrid.coords).reshape(cN, cN)
                    else:
                        uc = field_decoder(p, cgrid.coords).reshape(cN, cN)
                    un = uc if cN == n else endpoint_bilinear(uc, n)
                    # Explicit reset makes hard-BC preservation independent of roundoff.
                    un = un.at[0, :].set(0.0).at[-1, :].set(0.0)
                    un = un.at[:, 0].set(0.0).at[:, -1].set(0.0)
                    return un

                def learned_guess(p, F, *, learned_full_f=learned_full_f,
                                  additive_base=additive_base,
                                  residual_gate=(arm.predictor == "transport_tail")):
                    base0 = additive_base(F)
                    x0 = base0 + learned_full_f(p, F)[1:-1, 1:-1]
                    if residual_gate:
                        base_r2 = jnp.sum(jnp.square(F - op(base0)))
                        candidate_r2 = jnp.sum(jnp.square(F - op(x0)))
                        x0 = jnp.where(candidate_r2 <= base_r2, x0, base0)
                    if q:
                        residual = F - op(x0)
                        coeff = (Sq.T @ residual @ Sq) / lamq
                        x0 = x0 + Sq @ coeff @ Sq.T
                    return x0

                raw_guess = learned_guess
            else:
                def spectral_guess(p, F):
                    del p
                    coeff = (Sq.T @ F @ Sq) / lamq
                    return Sq @ coeff @ Sq.T

                raw_guess = spectral_guess

            guess = jax.jit(raw_guess)
            guess_registry[arm.name] = guess
            X0 = [guess(params[i], Fs[i]) for i in range(N_TEST)]
            if arm.predictor != "none":
                full0 = learned_full_f(params[0], Fs[0])
                if arm.predictor == "transport_tail":
                    full0 = full0 + jnp.pad(transport_base(Fs[0]), 1)
            else:
                full0 = jnp.pad(X0[0], 1)
            boundary_check = float(jnp.max(jnp.abs(jnp.concatenate(
                [full0[0, :], full0[-1, :], full0[:, 0], full0[:, -1]]
            ))))
            if boundary_check > 1e-14:
                raise SystemExit(f"N={n} arm={arm.name}: hard-BC failure {boundary_check:.3e}")
            diagnostics = [norm_metrics(X0[i], Uref[i], op, Fs[i]) for i in range(N_TEST)]
            transport_gate_diagnostics = None
            if arm.predictor == "transport_tail":
                transport_gate_diagnostics = []
                for i in range(N_TEST):
                    b0 = transport_base(Fs[i])
                    candidate = b0 + learned_full_f(params[i], Fs[i])[1:-1, 1:-1]
                    rb = float(jnp.linalg.norm(Fs[i] - op(b0)) / jnp.linalg.norm(Fs[i]))
                    rc = float(jnp.linalg.norm(Fs[i] - op(candidate)) / jnp.linalg.norm(Fs[i]))
                    transport_gate_diagnostics.append(dict(
                        case=i, base_true_rel_residual=rb,
                        candidate_true_rel_residual=rc, accepted=bool(rc <= rb),
                    ))
            lm_diagnostics = None
            if (arm.predictor.startswith("lm_") or arm.predictor == "group_nearest"
                    or (arm.predictor == "transport_tail" and arm.gn_budget == 1)):
                solver = (group_solvers[arm.rom_tau]
                          if arm.predictor == "group_nearest" else
                          transport_solvers[arm.weak_M]
                          if arm.predictor == "transport_tail" else
                          lm_tr if arm.predictor.startswith("lm_tr_") else lm_base)
                lm_diagnostics = []
                for i in range(N_TEST):
                    z0i = z_mean
                    nearest_index = None
                    if arm.predictor == "group_nearest":
                        nearest_index = int(jnp.argmin(jnp.sum(
                            (nearest_P - params[i][None, :]) ** 2, axis=1)))
                        z0i = group_model["Z"][nearest_index]
                        source_projection = gpre_apply
                    elif arm.predictor == "transport_tail":
                        z0i = transport_z_init(Fs[i])
                        source_projection = transport_projectors[arm.weak_M]
                    else:
                        source_projection = pre_apply
                        if arm.predictor.endswith("nearest"):
                            nearest_index = int(jnp.argmin(jnp.sum(
                                (nearest_P - params[i][None, :]) ** 2, axis=1)))
                            z0i = nearest_Z[nearest_index]
                    _, val, v0, nJ, acc, att, reason = solver(
                        z0i, source_projection(Fs[i])
                    )
                    lm_diagnostics.append(dict(
                        case=i,
                        nearest_training_index=nearest_index,
                        final_objective=float(val),
                        initial_objective=float(v0),
                        objective_reduction=float(val / jnp.maximum(v0, 1e-300)),
                        jacobian_evaluations=int(nJ),
                        accepted=int(acc),
                        attempts=int(att),
                        reason_code=int(reason),
                    ))

            construction_reps = []
            if BURN_S > 0:
                wu.gpu_burn(
                    lambda: cg(Fs[0], zero, FOM_TAUS[0])[0].block_until_ready(), BURN_S
                )
            for i in range(N_TIME):
                _, reps = wu.time_fn(
                    lambda ii=i: guess(params[ii], Fs[ii]).block_until_ready(),
                    TIME_REPS,
                    TIME_WARM,
                )
                construction_reps.append(reps)

            for tau in FOM_TAUS:
                # One CG object for both arms. The hybrid callable contains construction.
                hybrid = jax.jit(lambda p, F: cg(F, raw_guess(p, F), tau))
                baseline = baseline_registry[tau]
                timing_registry[(tau, arm.name)] = hybrid
                finals = []
                for i in range(N_TEST):
                    xh, kh, rh, fh = hybrid(params[i], Fs[i])
                    finals.append((xh, int(kh), float(rh), int(fh)))
                bases = [(x, int(k), float(r), int(flag))
                         for x, k, r, flag in baseline_validation[tau]]
                if max(v[3] for v in finals + bases) != 0:
                    raise SystemExit(f"N={n} arm={arm.name} tau={tau}: CG failure flag")
                if max(v[2] for v in finals + bases) > tau:
                    raise SystemExit(f"N={n} arm={arm.name} tau={tau}: true residual gate failed")

                hybrid_reps, baseline_reps = [], []
                hybrid_timed_telemetry, baseline_timed_telemetry = [], []
                if PAIRWISE_DIAGNOSTIC:
                    for i in range(N_TIME):
                        if BURN_S > 0:
                            wu.gpu_burn(
                                lambda ii=i: baseline(params[ii], Fs[ii])[0].block_until_ready(),
                                BURN_S,
                            )
                        ha, ba, hg, bg = paired_time(
                            lambda ii=i: hybrid(params[ii], Fs[ii]),
                            lambda ii=i: baseline(params[ii], Fs[ii]),
                            lambda out, ii=i: grade_cg_output(out, Fs[ii], Uref[ii], op),
                            lambda out, ii=i: grade_cg_output(out, Fs[ii], Uref[ii], op),
                            TIME_REPS,
                            TIME_WARM,
                        )
                        hybrid_reps.append(ha); baseline_reps.append(ba)
                        hybrid_timed_telemetry.append(hg); baseline_timed_telemetry.append(bg)
                hmed = [float(np.median(v)) for v in hybrid_reps]
                bmed = [float(np.median(v)) for v in baseline_reps]
                hf = [float(jnp.linalg.norm(finals[i][0] - Uref[i])
                            / jnp.linalg.norm(Uref[i])) for i in range(N_TEST)]
                bf = [float(jnp.linalg.norm(bases[i][0] - Uref[i])
                            / jnp.linalg.norm(Uref[i])) for i in range(N_TEST)]
                row = dict(
                    N=n,
                    n_dof=ni**2,
                    arm=arm.name,
                    component=("combined_q16_transport" if
                               arm.predictor == "transport_tail" else
                               "learned" if arm.predictor != "none" and q == 0 else
                               "combined" if arm.predictor != "none" else "classical"),
                    method_family=("nmrom" if arm.predictor.startswith("lm_") else
                                   "nmrom_nonlinear_groupfilm" if
                                   arm.predictor == "group_nearest" else
                                   "nmrom_q16_transport_tail" if
                                   arm.predictor == "transport_tail" and arm.gn_budget else
                                   "direct_q16_transport_tail" if
                                   arm.predictor == "transport_tail" else
                                   "direct_surrogate" if arm.predictor.startswith("param_") else
                                   "latent_prediction_negative_control" if arm.predictor in
                                   ("rbf", "nearest") else "classical_spectral"),
                    predictor=arm.predictor,
                    rom_tau=arm.rom_tau,
                    weak_M=arm.weak_M,
                    weak_m=arm.weak_m,
                    gn_budget=arm.gn_budget,
                    exact_base_q=arm.base_q,
                    coarse_n=arm.coarse_n,
                    spectral_q=q,
                    spectral_modes=q*q,
                    fom_tau=tau,
                    construction_all_s=construction_reps,
                    construction_ms=float(np.mean(
                        [np.median(v) for v in construction_reps])) * 1e3,
                    hybrid_all_s=hybrid_reps,
                    baseline_all_s=baseline_reps,
                    hybrid_timed_telemetry=hybrid_timed_telemetry,
                    baseline_timed_telemetry=baseline_timed_telemetry,
                    hybrid_total_ms=(float(np.mean(hmed)) * 1e3 if hmed else None),
                    baseline_total_ms=(float(np.mean(bmed)) * 1e3 if bmed else None),
                    speedup_vs_zero_cg=(float(np.mean(bmed) / np.mean(hmed))
                                        if hmed else None),
                    iters_hybrid_all=[v[1] for v in finals],
                    iters_baseline_all=[v[1] for v in bases],
                    iters_hybrid_mean=float(np.mean([v[1] for v in finals])),
                    iters_baseline_mean=float(np.mean([v[1] for v in bases])),
                    iters_hybrid_timed_mean=(float(np.mean([
                        g["iterations"] for case in hybrid_timed_telemetry for g in case]))
                        if hybrid_timed_telemetry else None),
                    iters_baseline_timed_mean=(float(np.mean([
                        g["iterations"] for case in baseline_timed_telemetry for g in case]))
                        if baseline_timed_telemetry else None),
                    iter_saving_fraction=1.0 - float(np.mean([v[1] for v in finals]))
                                             / float(np.mean([v[1] for v in bases])),
                    guess_rel_l2_mean=float(np.mean([v["guess_rel_l2"] for v in diagnostics])),
                    guess_rel_l2_median=float(np.median([v["guess_rel_l2"] for v in diagnostics])),
                    guess_a_norm_ratio_mean=float(np.mean(
                        [v["guess_a_norm_ratio"] for v in diagnostics])),
                    guess_true_rel_residual_mean=float(np.mean(
                        [v["guess_true_rel_residual"] for v in diagnostics])),
                    guess_diagnostics_per_case=diagnostics,
                    lm_diagnostics_per_case=lm_diagnostics,
                    transport_residual_gate_per_case=transport_gate_diagnostics,
                    final_true_rel_residual_max=(float(max(
                        g["recomputed_true_rel_residual"]
                        for case in hybrid_timed_telemetry for g in case))
                        if hybrid_timed_telemetry else None),
                    baseline_true_rel_residual_max=(float(max(
                        g["recomputed_true_rel_residual"]
                        for case in baseline_timed_telemetry for g in case))
                        if baseline_timed_telemetry else None),
                    final_true_rel_residual_validation_max=float(max(v[2] for v in finals)),
                    baseline_true_rel_residual_validation_max=float(max(v[2] for v in bases)),
                    final_rel_l2_mean=(float(np.mean([
                        g["rel_l2_vs_exact_dst"]
                        for case in hybrid_timed_telemetry for g in case]))
                        if hybrid_timed_telemetry else None),
                    baseline_rel_l2_mean=(float(np.mean([
                        g["rel_l2_vs_exact_dst"]
                        for case in baseline_timed_telemetry for g in case]))
                        if baseline_timed_telemetry else None),
                    final_rel_l2_validation_mean=float(np.mean(hf)),
                    baseline_rel_l2_validation_mean=float(np.mean(bf)),
                    exact_direct_ms=None,
                    boundary_contract_maxabs=boundary_check,
                )
                report["rows"].append(row)
                save()
                if PAIRWISE_DIAGNOSTIC:
                    print(
                        f"RESULT N={n} {arm.name} tau={tau:.0e}: construct "
                        f"{row['construction_ms']:.3f} ms, Aerr "
                        f"{row['guess_a_norm_ratio_mean']:.3e}, iters "
                        f"{row['iters_hybrid_mean']:.1f}/{row['iters_baseline_mean']:.1f}, "
                        f"total {row['hybrid_total_ms']:.3f}/"
                        f"{row['baseline_total_ms']:.3f} ms, speedup "
                        f"{row['speedup_vs_zero_cg']:.3f}x",
                        flush=True,
                    )
                else:
                    print(
                        f"VALIDATED N={n} {arm.name} tau={tau:.0e}: construct "
                        f"{row['construction_ms']:.3f} ms, Aerr "
                        f"{row['guess_a_norm_ratio_mean']:.3e}, iters "
                        f"{row['iters_hybrid_mean']:.1f}/{row['iters_baseline_mean']:.1f}; "
                        "authoritative joint timing pending",
                        flush=True,
                    )

        # Authoritative wall clock: all arms, zero-start CG, and exact FFT-DST are rotated
        # through one post-burn block. This supports paired combined-vs-spectral deltas and
        # removes timing-order selection. Telemetry is graded from every timed invocation.
        mesh_joint = {}
        for tau in FOM_TAUS:
            baseline = baseline_registry[tau]
            direct_runtime = jax.jit(lambda p, F: direct(F))
            dense_direct_runtime = jax.jit(lambda p, F: dense_direct(F))
            native_names = ["native_zero", "native_jacobi_zero"] + [
                f"native_{v}" for v in NATIVE_ARM_NAMES if v in guess_registry
            ]
            names = (["zero_cg", "fft_dst_direct", "dense_dst_direct"]
                     + [a.name for a in ARMS]
                     + native_names)
            funcs = {
                "zero_cg": lambda p, F, baseline=baseline: baseline(p, F),
                "fft_dst_direct": lambda p, F, direct_runtime=direct_runtime: direct_runtime(p, F),
                "dense_dst_direct": (
                    lambda p, F, dense_direct_runtime=dense_direct_runtime:
                    dense_direct_runtime(p, F)
                ),
            }
            for a in ARMS:
                hfn = timing_registry[(tau, a.name)]
                funcs[a.name] = lambda p, F, hfn=hfn: hfn(p, F)
            native_zero = jax.jit(lambda p, F: jax.scipy.sparse.linalg.cg(
                op, F, x0=zero, tol=tau, maxiter=CG_MAXITER)[0])
            funcs["native_zero"] = lambda p, F, fn=native_zero: fn(p, F)
            jacobi_scale = 1.0 / (4.0 * (n - 1) ** 2)
            native_jacobi_zero = jax.jit(lambda p, F: jax.scipy.sparse.linalg.cg(
                op, F, x0=zero, tol=tau, maxiter=CG_MAXITER,
                M=lambda x: jacobi_scale * x)[0])
            funcs["native_jacobi_zero"] = (
                lambda p, F, fn=native_jacobi_zero: fn(p, F)
            )
            for arm_name in NATIVE_ARM_NAMES:
                if arm_name not in guess_registry:
                    continue
                gfn = guess_registry[arm_name]
                nfn = jax.jit(lambda p, F, gfn=gfn: jax.scipy.sparse.linalg.cg(
                    op, F, x0=gfn(p, F), tol=tau, maxiter=CG_MAXITER)[0])
                funcs[f"native_{arm_name}"] = lambda p, F, fn=nfn: fn(p, F)
            all_times = {name: [] for name in names}
            all_telemetry = {name: [] for name in names}
            for i in range(N_TIME):
                # Compile/warm every function, then burn immediately before the timed block.
                for name in names:
                    jax.block_until_ready(funcs[name](params[i], Fs[i]))
                if BURN_S > 0:
                    wu.gpu_burn(
                        lambda ii=i: baseline(params[ii], Fs[ii])[0].block_until_ready(),
                        BURN_S,
                    )
                case_t = {name: [] for name in names}
                case_g = {name: [] for name in names}
                for rep in range(TIME_REPS):
                    shift = (i + rep) % len(names)
                    order = names[shift:] + names[:shift]
                    for name in order:
                        t0 = time.perf_counter()
                        out = funcs[name](params[i], Fs[i])
                        jax.block_until_ready(out)
                        case_t[name].append(time.perf_counter() - t0)
                        if name in ("fft_dst_direct", "dense_dst_direct") or name.startswith("native_"):
                            x = out
                            case_g[name].append(dict(
                                recomputed_true_rel_residual=float(
                                    jnp.linalg.norm(op(x) - Fs[i]) / jnp.linalg.norm(Fs[i])),
                                rel_l2_vs_exact_dst=float(
                                    jnp.linalg.norm(x - Uref[i]) / jnp.linalg.norm(Uref[i])),
                            ))
                        else:
                            case_g[name].append(grade_cg_output(out, Fs[i], Uref[i], op))
                for name in names:
                    all_times[name].append(case_t[name])
                    all_telemetry[name].append(case_g[name])
            means = {name: float(np.mean([np.median(v) for v in all_times[name]]))
                     for name in names}
            summary_seed = BOOTSTRAP_SEED + 1009 * n + int(round(-np.log10(tau)))
            timing_summaries = {
                name: repetition_summary(all_times[name], summary_seed + 37 * j)
                for j, name in enumerate(names)
            }
            paired_deltas_by_control = {}
            # Pair every candidate directly against every spectral arm present in this
            # invocation.  In particular, final confirmation must compare learned arms
            # with the validation-locked rank, which need not be q8 or q16.
            spectral_controls = [name for name in names if name.startswith("spectral_q")]
            for control in spectral_controls:
                st = all_times[control]
                control_rows = {}
                for name in names:
                    if (name in ("zero_cg", "fft_dst_direct", "dense_dst_direct", control)
                            or name.startswith("native_")):
                        continue
                    delta = [[all_times[name][i][r] - st[i][r]
                              for r in range(TIME_REPS)] for i in range(N_TIME)]
                    dsummary = repetition_summary(
                        delta, summary_seed + 10000 + 53 * names.index(name)
                    )
                    control_rows[name] = dict(
                        all_s=delta,
                        mean_ms=dsummary["mean_of_case_medians_s"] * 1e3,
                        bootstrap_ci95_ms=[v * 1e3 for v in dsummary["bootstrap_ci95_s"]],
                        speedup_spectral_over_arm=means[control] / means[name],
                        speedup_bootstrap_ci95=paired_ratio_ci(
                            st, all_times[name],
                            summary_seed + 20000 + 71 * names.index(name),
                        ),
                    )
                paired_deltas_by_control[control] = control_rows
            mesh_joint[str(tau)] = dict(
                order_base=names,
                order_rule="cyclic shift by (case + repetition)",
                all_s=all_times,
                timed_telemetry=all_telemetry,
                mean_of_case_medians_ms={k: v * 1e3 for k, v in means.items()},
                timing_summaries=timing_summaries,
                paired_delta_by_spectral_control=paired_deltas_by_control,
            )
            mesh_check.setdefault("authoritative_exact_direct_ms", {})[str(tau)] = (
                means["fft_dst_direct"] * 1e3
            )
            mesh_check.setdefault("authoritative_dense_dst_direct_ms", {})[str(tau)] = (
                means["dense_dst_direct"] * 1e3
            )
            # Replace earlier pairwise diagnostic wall clock with authoritative joint values.
            base_ms = means["zero_cg"] * 1e3
            for row in report["rows"]:
                if row["N"] != n or row["fom_tau"] != tau:
                    continue
                name = row["arm"]
                row["pairwise_diagnostic_hybrid_total_ms"] = row["hybrid_total_ms"]
                row["pairwise_diagnostic_baseline_total_ms"] = row["baseline_total_ms"]
                row["joint_timing_authoritative"] = True
                row["hybrid_total_ms"] = means[name] * 1e3
                row["baseline_total_ms"] = base_ms
                row["speedup_vs_zero_cg"] = means["zero_cg"] / means[name]
                row["hybrid_total_bootstrap_ci95_ms"] = [
                    v * 1e3 for v in timing_summaries[name]["bootstrap_ci95_s"]
                ]
                row["hybrid_timing_outlier_count"] = timing_summaries[name]["outlier_count"]
                row["baseline_total_bootstrap_ci95_ms"] = [
                    v * 1e3 for v in timing_summaries["zero_cg"]["bootstrap_ci95_s"]
                ]
                row["baseline_timing_outlier_count"] = timing_summaries["zero_cg"][
                    "outlier_count"
                ]
                row["speedup_vs_zero_cg_bootstrap_ci95"] = paired_ratio_ci(
                    all_times["zero_cg"], all_times[name],
                    summary_seed + 30000 + 97 * names.index(name),
                )
                row["exact_direct_ms"] = means["fft_dst_direct"] * 1e3
                row["dense_dst_direct_ms"] = means["dense_dst_direct"] * 1e3
                row["speedup_spectral_q8_over_arm"] = (
                    means["spectral_q8"] / means[name] if "spectral_q8" in means else None
                )
                tele = all_telemetry[name]
                btele = all_telemetry["zero_cg"]
                row["iters_hybrid_timed_mean"] = float(np.mean([
                    g["iterations"] for case in tele for g in case]))
                row["iters_baseline_timed_mean"] = float(np.mean([
                    g["iterations"] for case in btele for g in case]))
                row["final_true_rel_residual_max"] = float(max(
                    g["recomputed_true_rel_residual"] for case in tele for g in case))
                row["final_rel_l2_mean"] = float(np.mean([
                    g["rel_l2_vs_exact_dst"] for case in tele for g in case]))
                row["joint_timed_telemetry"] = tele
                row["baseline_true_rel_residual_max"] = float(max(
                    g["recomputed_true_rel_residual"] for case in btele for g in case))
                row["baseline_rel_l2_mean"] = float(np.mean([
                    g["rel_l2_vs_exact_dst"] for case in btele for g in case]))
                row["joint_baseline_timed_telemetry"] = btele
                for control, deltas in paired_deltas_by_control.items():
                    if name in deltas:
                        row[f"paired_delta_vs_{control}"] = deltas[name]
                        row[f"speedup_{control}_over_arm"] = means[control] / means[name]
            save()
        mesh_check["joint_timing"] = mesh_joint

        mesh_check["wall_seconds"] = time.time() - mesh_t0
        save()
    report["complete"] = True
    save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
