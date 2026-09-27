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
    os.path.join(EXPS, "poisson2d-rom-objective"),
    os.path.join(EXPS, "poisson2d-rom-objective", "followup"),
    os.path.join(EXPS, "rom-warmstart-fom"),
):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import pro_common as pc  # noqa: E402
from pro_common import mp, F64  # noqa: E402
from fu_eq import eq_fit, weak_source_projector  # noqa: E402
from wsf_poisson import make_lm_obj_jit  # noqa: E402
import wsf_util as wu  # noqa: E402


OUT = sys.argv[1]
PKL = os.environ["PKL"]
PARAM_PKL = os.environ.get("PARAM_PKL", "")
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
TR_SCALE = float(os.environ.get("TR_SCALE", "1.0"))
CAL_SEED = int(os.environ.get("CAL_SEED", "73129"))
CAL_FRAC = float(os.environ.get("CAL_FRAC", "0.2"))
RBF_LENGTHS = [float(v) for v in os.environ.get(
    "RBF_LENGTHS", "0.35,0.5,0.7,1.0,1.4,2.0,3.0,5.0").split(",")]
RBF_RIDGES = [float(v) for v in os.environ.get(
    "RBF_RIDGES", "1e-8,1e-6,1e-4,1e-2,1e-1,1").split(",")]
ARM_NAMES = [v for v in os.environ.get(
    "ARMS", "rbf_c64_q0,nearest_c64_q0,rbf_c64_q8,spectral_q8,spectral_q16"
).split(",") if v]


@dataclass(frozen=True)
class Arm:
    name: str
    predictor: str
    coarse_n: int
    q: int


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


def paired_time(fn_a, fn_b, reps: int, warm: int) -> tuple[list[float], list[float]]:
    """Back-to-back timing with alternating order; every call must block."""
    for _ in range(warm):
        fn_a(); fn_b()
    aa, bb = [], []

    def one(fn):
        t0 = time.perf_counter(); fn(); return time.perf_counter() - t0

    for r in range(reps):
        if r % 2 == 0:
            aa.append(one(fn_a)); bb.append(one(fn_b))
        else:
            bb.append(one(fn_b)); aa.append(one(fn_a))
    return aa, bb


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
    d, cfg, stages, Ztr, hard_bc = pc.load_pkl(PKL)
    if not hard_bc:
        raise SystemExit("this harness requires the hard-BC checkpoint")
    if cfg["K_LAT"] != 8:
        raise SystemExit(f"pre-registered feasibility requires K=8, got {cfg['K_LAT']}")
    dec = pc.make_decoder(stages[:1], hard_bc=True)

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
            trust_region_scale=TR_SCALE,
            arms=[a.__dict__ for a in ARMS],
            cg_maxiter=CG_MAXITER,
            matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "unset"),
            dtype="f64",
            prolongation="endpoint-aligned separable bilinear",
            prolongation_affine_selfcheck_maxabs=interp_affine_err,
        ),
        provenance=prov,
        rbf_train_only=rbf_info,
        parameter_aligned_checkpoint=param_info,
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

        # The exact sine inverse is both the error reference and the strongest baseline.
        direct = jax.jit(lambda F: grid.S @ ((grid.S.T @ F @ grid.S) / grid.lam) @ grid.S.T)
        Uref = [direct(F) for F in Fs]
        d_res = [float(jnp.linalg.norm(op(Uref[i]) - Fs[i]) / jnp.linalg.norm(Fs[i]))
                 for i in range(N_TEST)]

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
            exact_direct_true_rel_residual_max=float(max(d_res)),
            exact_direct_all_s=direct_reps,
            exact_direct_ms=float(np.mean([np.median(v) for v in direct_reps])) * 1e3,
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
        predictor_reps = {"rbf": [], "nearest": []}
        for key, fn in (("rbf", rbf_jit), ("nearest", nearest_jit)):
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
            pts, wq, eq_info = eq_fit(dec, grid, Ztr, cfg["K_LAT"], M_MODES, MQ, "offgrid")
            PhiT, Wl = pc.colloc_mode_table(grid, spec, "offgrid", pts)
            pre_apply, pre_build_s = weak_source_projector(grid, spec, "offgrid")
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
            mesh_check["source_projector_offline_build_s"] = pre_build_s
            mesh_check["training_latent_radius"] = train_radius
            mesh_check["trust_delta"] = TR_SCALE * train_radius
            mesh_check["trust_inf_vs_audited_lm_maxabs"] = lm_equiv
        else:
            lm_base = lm_tr = None

        for arm in ARMS:
            q = min(arm.q, ni)
            Sq = grid.S[:, :q] if q else None
            lamq = grid.lam[:q, :q] if q else None

            if arm.predictor != "none":
                cgrid = grid if arm.coarse_n < 0 else pc.Grid(arm.coarse_n)
                cN = n if arm.coarse_n < 0 else arm.coarse_n
                field_decoder = None
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
                else:
                    raise AssertionError(arm.predictor)

                def learned_full_f(p, F, *, cgrid=cgrid, cN=cN, predictor=predictor,
                                   field_decoder=field_decoder):
                    if field_decoder is None:
                        z = predictor(p, F)
                        uc = dec(z, cgrid.coords).reshape(cN, cN)
                    else:
                        uc = field_decoder(p, cgrid.coords).reshape(cN, cN)
                    un = endpoint_bilinear(uc, n)
                    # Explicit reset makes hard-BC preservation independent of roundoff.
                    un = un.at[0, :].set(0.0).at[-1, :].set(0.0)
                    un = un.at[:, 0].set(0.0).at[:, -1].set(0.0)
                    return un

                def learned_guess(p, F, *, learned_full_f=learned_full_f):
                    x0 = learned_full_f(p, F)[1:-1, 1:-1]
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
            X0 = [guess(params[i], Fs[i]) for i in range(N_TEST)]
            if arm.predictor != "none":
                full0 = learned_full_f(params[0], Fs[0])
            else:
                full0 = jnp.pad(X0[0], 1)
            boundary_check = float(jnp.max(jnp.abs(jnp.concatenate(
                [full0[0, :], full0[-1, :], full0[:, 0], full0[:, -1]]
            ))))
            if boundary_check > 1e-14:
                raise SystemExit(f"N={n} arm={arm.name}: hard-BC failure {boundary_check:.3e}")
            diagnostics = [norm_metrics(X0[i], Uref[i], op, Fs[i]) for i in range(N_TEST)]

            construction_reps = []
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
                baseline = jax.jit(lambda p, F: cg(F, zero, tau))
                finals, bases = [], []
                for i in range(N_TEST):
                    xh, kh, rh, fh = hybrid(params[i], Fs[i])
                    xb, kb, rb, fb = baseline(params[i], Fs[i])
                    finals.append((xh, int(kh), float(rh), int(fh)))
                    bases.append((xb, int(kb), float(rb), int(fb)))
                if max(v[3] for v in finals + bases) != 0:
                    raise SystemExit(f"N={n} arm={arm.name} tau={tau}: CG failure flag")
                if max(v[2] for v in finals + bases) > tau:
                    raise SystemExit(f"N={n} arm={arm.name} tau={tau}: true residual gate failed")

                hybrid_reps, baseline_reps = [], []
                for i in range(N_TIME):
                    ha, ba = paired_time(
                        lambda ii=i: hybrid(params[ii], Fs[ii])[0].block_until_ready(),
                        lambda ii=i: baseline(params[ii], Fs[ii])[0].block_until_ready(),
                        TIME_REPS,
                        TIME_WARM,
                    )
                    hybrid_reps.append(ha); baseline_reps.append(ba)
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
                    component=("learned" if arm.predictor != "none" and q == 0 else
                               "combined" if arm.predictor != "none" else "classical"),
                    method_family=("nmrom" if arm.predictor.startswith("lm_") else
                                   "direct_surrogate" if arm.predictor.startswith("param_") else
                                   "latent_prediction_negative_control" if arm.predictor in
                                   ("rbf", "nearest") else "classical_spectral"),
                    predictor=arm.predictor,
                    coarse_n=arm.coarse_n,
                    spectral_q=q,
                    spectral_modes=q*q,
                    fom_tau=tau,
                    construction_all_s=construction_reps,
                    construction_ms=float(np.mean(
                        [np.median(v) for v in construction_reps])) * 1e3,
                    hybrid_all_s=hybrid_reps,
                    baseline_all_s=baseline_reps,
                    hybrid_total_ms=float(np.mean(hmed)) * 1e3,
                    baseline_total_ms=float(np.mean(bmed)) * 1e3,
                    speedup_vs_zero_cg=float(np.mean(bmed) / np.mean(hmed)),
                    iters_hybrid_all=[v[1] for v in finals],
                    iters_baseline_all=[v[1] for v in bases],
                    iters_hybrid_mean=float(np.mean([v[1] for v in finals])),
                    iters_baseline_mean=float(np.mean([v[1] for v in bases])),
                    iter_saving_fraction=1.0 - float(np.mean([v[1] for v in finals]))
                                             / float(np.mean([v[1] for v in bases])),
                    guess_rel_l2_mean=float(np.mean([v["guess_rel_l2"] for v in diagnostics])),
                    guess_rel_l2_median=float(np.median([v["guess_rel_l2"] for v in diagnostics])),
                    guess_a_norm_ratio_mean=float(np.mean(
                        [v["guess_a_norm_ratio"] for v in diagnostics])),
                    guess_true_rel_residual_mean=float(np.mean(
                        [v["guess_true_rel_residual"] for v in diagnostics])),
                    final_true_rel_residual_max=float(max(v[2] for v in finals)),
                    baseline_true_rel_residual_max=float(max(v[2] for v in bases)),
                    final_rel_l2_mean=float(np.mean(hf)),
                    baseline_rel_l2_mean=float(np.mean(bf)),
                    exact_direct_ms=mesh_check["exact_direct_ms"],
                    boundary_contract_maxabs=boundary_check,
                )
                report["rows"].append(row)
                save()
                print(
                    f"RESULT N={n} {arm.name} tau={tau:.0e}: construct "
                    f"{row['construction_ms']:.3f} ms, Aerr {row['guess_a_norm_ratio_mean']:.3e}, "
                    f"iters {row['iters_hybrid_mean']:.1f}/{row['iters_baseline_mean']:.1f}, "
                    f"total {row['hybrid_total_ms']:.3f}/{row['baseline_total_ms']:.3f} ms, "
                    f"speedup {row['speedup_vs_zero_cg']:.3f}x",
                    flush=True,
                )

        mesh_check["wall_seconds"] = time.time() - mesh_t0
        save()
    report["complete"] = True
    save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
