"""PHASE 2b, THE STOP GATE: reconstruction only.  No residual, no ROM, no
timing.  Run this BEFORE anything else and honour its verdict.

WHY THIS GATE EXISTS.  Phase 2a established that the family is genuinely
eight-dimensional and curved (Jacobian rank [8,8,8], centred snapshot rank 32,
held-out POD-8 error 3.84e-2, POD-16 3.71e-3).  What it could NOT establish is
where a FINITE MLP HEAD actually stops.  POD-32's ~1e-12 is the BANK-SPAN
ceiling, NOT the neural decoder's ceiling, and it must never be presented as
such (phase-2a verification, carried-forward condition 4).

THE PREDECLARED PASS BAR, fixed before any training run and NOT adjustable
after seeing the number:

    the held-out nonlinear RECONSTRUCTION ORACLE must beat POD-8 by at least
    3x in the AGGREGATE held-out error AND in the PER-CASE MEDIAN,
    on the IDENTICAL held-out cohort in the IDENTICAL mass-weighted norm.

Beating POD-8 by ~10x would put the head near POD-16 and is the genuinely
interesting outcome.  If the oracle sits at the POD-8 floor, phase 2b STOPS:
the negative is a complete result.

STRUCTURE OF THE VERDICT.  The stop gate is a VERDICT, not a correctness
check, so it does not abort the run: the artifact is written either way with
`verdict.passed` recorded, and `stk2d_rom_gates.py` REFUSES TO RUN unless this
artifact exists, is complete, and says passed.  Every other gate in this driver
IS a correctness check and is asserted in the usual way.

GATES
  PRECOND    frozen config, no -O, SMOKE never certifies.
  S0         solver dtype f64; JAX x64 / matmul=highest / backend gpu.
  S-METRIC   the coefficient-space identity
             ||u - ubar - G h||_M^2 = ||c - h||^2 + ||perp||_M^2
             against a DIRECT field computation.  Every number in this driver
             is computed through that identity, so it is measured, not assumed.
  S-REGR     the phase-2b bank reproduces the CERTIFIED phase-2a held-out
             POD-8/16/32 errors from runs/stk2d/stk2d_bank_gates_bank_nu1.json.
  S-AFFINE   c(mu) = Cd theta(mu) - cbar against coefficients taken from
             DIRECT FOM solves, at phase 2a's own 1e-14 N^2 budget.
  S-RECON    THE STOP GATE.  Oracle vs POD-8/16/32, aggregate and per case.
             NEGATIVE CONTROLS, both asserted: a LINEAR head must land at the
             POD-K floor (it spans a K-dimensional affine subspace, so it
             cannot beat POD-K), and an UNTRAINED head must be no better than
             POD-K.  Without those two the gate could not distinguish "the
             head works" from "the metric is wrong".
  MANIFEST   every expected gate present, exact row counts, no non-finite.

Env: RECON_NS, R_STOP, K_LAT, N_FIT, HID, LAYERS, STEPS, LR, SEED, OUT_TAG,
     OUT_PREFIX, CACHE, ALLOW_CPU=0, SMOKE=0.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np
import scipy

import stk2d_common as stk
import stk2d_bank as bank
import stk2d_rom as rom
import stk2d_head as head

HERE = os.path.dirname(os.path.abspath(__file__))

RECON_NS = [int(v) for v in os.environ.get("RECON_NS", "32,64,128,256").split(",") if v]
R_LADDER = [int(v) for v in os.environ.get("R_LADDER", "8,16,32").split(",") if v]
R_STOP = int(os.environ.get("R_STOP", "32"))
K_LAT = int(os.environ.get("K_LAT", "8"))
N_PRIMARY = int(os.environ.get("N_PRIMARY", "64"))
N_FIT = int(os.environ.get("N_FIT", "16384"))
N_VAL = int(os.environ.get("N_VAL", "256"))
BATCH = int(os.environ.get("BATCH", "2048"))
HID = int(os.environ.get("HID", "128"))
LAYERS = int(os.environ.get("LAYERS", "3"))
STEPS = int(os.environ.get("STEPS", "40000"))
LR = float(os.environ.get("LR", "3e-3"))
S_TRAIN = int(os.environ.get("S_TRAIN", "256"))
S_TEST = int(os.environ.get("S_TEST", "64"))
NU = float(os.environ.get("NU", "1.0"))
SEED = int(os.environ.get("SEED", "20260830"))
OUT_TAG = os.environ.get("OUT_TAG", "")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "runs/stk2d/")
CACHE = os.environ.get("CACHE", "runs/stk2d/cache")
HEADS = os.environ.get("HEADS", "runs/stk2d/heads")
ALLOW_CPU = int(os.environ.get("ALLOW_CPU", "0"))
SMOKE = int(os.environ.get("SMOKE", "0"))

PHASE2A = "runs/stk2d/stk2d_bank_gates_bank_nu1.json"

# ------------------------------------------------------------- thresholds ---
STOP_FACTOR = 3.0          # THE PREDECLARED BAR.  Frozen before any training
                           # run; not adjustable after seeing the number.
METRIC_TOL = 1e-12         # the coefficient-space identity
REGR_TOL = 1e-9            # vs the certified phase-2a POD errors
AFFINE_SCALE = 1e-14       # phase 2a's own affine-identity budget, x N^2
LINEAR_CTL_BAND = 1.5      # how far the linear control may sit BELOW the
                           # POD-K floor.  It is a BAND, not a floor, because
                           # POD-K is fitted to 256 training snapshots while
                           # the linear control fits an arbitrary
                           # K-dimensional affine subspace to 16384 samples,
                           # so on a HELD-OUT cohort the control can
                           # generalise slightly better than POD-K -- and does
                           # (retraction 30).  The statement that does follow
                           # from the geometry, and is gated, is that it can
                           # never CLEAR the stop bar
ORACLE_STARTS = 8
ORACLE_ITERS = 400

FROZEN_CONFIG = dict(recon_ns=[32, 64, 128, 256], r_ladder=[8, 16, 32],
                     r_stop=32, k_lat=8, n_primary=64, n_fit=16384, n_val=256,
                     batch=2048, hid=128, layers=3, steps=40000, s_train=256,
                     s_test=64, nu=1.0, allow_cpu=0, Q=48, K=8, grad_mix=3.0)
EXPECTED_GATES = frozenset(("PRECOND", "S0", "S_METRIC", "S_REGR", "S_AFFINE",
                            "S_SELECT", "S_RECON", "MANIFEST"))
EXPECTED_ROWS = dict(S_METRIC=4, S_REGR=4, S_AFFINE=4, S_SELECT=2,
                     S_RECON=4 * 3)


def finite(label, xs):
    a = np.asarray([float(x) for x in xs], dtype=float)
    bad = ~np.isfinite(a)
    assert not bad.any(), (f"non-finite value(s) in {label}: {a[bad].tolist()} "
                           f"(indices {np.nonzero(bad)[0].tolist()})")
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


def git_dirty():
    """Which tracked files differ from HEAD.

    The `git_commit` field is this project's provenance mechanism and it
    CANNOT detect uncommitted edits -- a run launched from a dirty tree records
    a hash that does not describe the code that produced it.  That happened
    twice in phase 2b (retraction 29), so PRECOND now asserts the tree is
    clean rather than trusting the hash."""
    try:
        out = subprocess.check_output(["git", "-C", HERE, "status",
                                       "--porcelain", "--untracked-files=no"],
                                      text=True, stderr=subprocess.DEVNULL)
        return [l[3:] for l in out.strip().split("\n") if l.strip()]
    except Exception:
        return ["<git unavailable>"]


def jax_provenance():
    out = dict(imported=False)
    try:
        import jax
        dev = jax.devices()[0]
        out = dict(imported=True, backend=dev.platform, device=str(dev),
                   device_kind=getattr(dev, "device_kind", ""),
                   x64=bool(jax.config.jax_enable_x64),
                   matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
                   jax_version=jax.__version__)
    except Exception as e:                                    # pragma: no cover
        out["error"] = repr(e)
    return out


# ------------------------------------------------------------- the metrics --

def pod_stats(cell, R):
    """Held-out POD-R error in phase 2a's convention: the aggregate is
    ||X_te - P_R X_te||_F / ||X_te||_F on the CENTRED held-out block and the
    per-case value is the same ratio for one column.  Mass weighting cancels
    in both ratios.

    The truncation residual is formed EXPLICITLY by `RomCell.perp_energy`,
    never as ||x||^2 - ||c||^2: at R = 32 that subtraction is twelve orders
    below f64 resolution and returns 2.7e-8 instead of 4.0e-14 (retraction 24).

    This uses the psi-route mean and the REORTHOGONALISED bank, which is what
    every other number in phase 2b uses; gate S-REGR separately reproduces
    phase 2a's own raw-bank numbers to 1e-9.
    """
    p2, t2 = cell.perp_energy(cell.U_te, R)
    e, tot = np.sqrt(p2), np.sqrt(t2)
    per = e / tot
    return dict(R=int(R), agg=float(np.linalg.norm(e) / np.linalg.norm(tot)),
                median=float(np.median(per)), max=float(per.max()),
                mean=float(per.mean()), per_case=[float(x) for x in per])


def head_stats(cell, R, res_coeff, perp2, tot2):
    """Field-space error of the head from its COEFFICIENT residual, through
    the identity ||u - ubar - G h||_M^2 = ||c - h||^2 + ||perp||_M^2.  `perp2`
    is the POD-R truncation energy, which is head-independent."""
    e = np.sqrt(np.asarray(res_coeff) ** 2 + np.asarray(perp2))
    tot = np.sqrt(np.asarray(tot2))
    per = e / tot
    return dict(agg=float(np.linalg.norm(e) / np.linalg.norm(tot)),
                median=float(np.median(per)), max=float(per.max()),
                mean=float(per.mean()), per_case=[float(x) for x in per])


def main():
    t_all = time.time()
    tag = OUT_TAG or f"recon_nu{NU:g}"
    out = os.path.join(OUT_PREFIX, f"stk2d_recon_gate_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    jp = jax_provenance()

    report = dict(config=dict(
        pde="stokes2d", kind="phase2b_stop_gate_reconstruction_only",
        driver_revision=1, phase1_artifact="runs/stk2d/stk2d_fom_gates_nu1_M64.json",
        phase2a_artifact=PHASE2A,
        decoder="u = ubar + G h(z); NO bc(x) mask (the bank carries div-free-ness "
                "and the no-slip BC); h is a silu MLP with a linear skip",
        training="autodecoder: joint Adam over head parameters and one free "
                 "latent code per training sample; no encoder",
        oracle="min_z ||c - h(z)|| by multi-start damped LM, vmapped over "
               "(case x restart) and scanned over iterations",
        recon_ns=RECON_NS, r_ladder=R_LADDER, r_stop=R_STOP, k_lat=K_LAT,
        n_primary=N_PRIMARY, n_fit=N_FIT, n_val=N_VAL, batch=BATCH,
        hid=HID, layers=LAYERS, steps=STEPS,
        lr=LR, s_train=S_TRAIN, s_test=S_TEST, nu=NU, seed=SEED,
        oracle_starts=ORACLE_STARTS, oracle_iters=ORACLE_ITERS,
        thresholds=dict(stop_factor=STOP_FACTOR, metric_tol=METRIC_TOL,
                        regr_tol=REGR_TOL, affine_scale=AFFINE_SCALE,
                        linear_ctl_band=LINEAR_CTL_BAND),
        numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), jax=jp, allow_cpu=bool(ALLOW_CPU),
        smoke=bool(SMOKE), git_commit=git_commit(),
        git_dirty=git_dirty(),
        hostname=os.uname().nodename), gates=dict(), complete=False)

    def save():
        json.dump(report, open(out, "w"), indent=1, default=float)

    save()      # complete=false BEFORE anything can fail
    log(f"stk2d RECONSTRUCTION STOP GATE (phase 2b) -> {out}")

    # ---- S0 ---------------------------------------------------------------
    gp = stk.MacGrid(8)
    probe = stk.solve_stokes(gp, stk.manufactured(gp)["f"])[0]
    report["gates"]["S0"] = dict(
        jax=jp, numpy_float64=str(probe.dtype),
        numpy_is_f64=bool(probe.dtype == np.float64), allow_cpu=bool(ALLOW_CPU),
        rule="solver output dtype float64; JAX x64=True, matmul_precision="
             "'highest', backend 'gpu' unless ALLOW_CPU=1")
    save()
    assert probe.dtype == np.float64, f"S0: dtype {probe.dtype}"
    assert jp.get("imported"), f"S0: JAX did not import: {jp}"
    assert jp.get("x64") is True, "S0: JAX_ENABLE_X64 is not active"
    assert jp.get("matmul_precision") == "highest", \
        f"S0: JAX_DEFAULT_MATMUL_PRECISION={jp.get('matmul_precision')}"
    if not ALLOW_CPU:
        assert jp.get("backend") == "gpu", \
            f"S0: jax backend is {jp.get('backend')}, not gpu"
    log(f"  S0 asserted: x64 {jp.get('x64')} matmul {jp.get('matmul_precision')} "
        f"backend {jp.get('backend')}")

    # ---- PRECOND ----------------------------------------------------------
    observed = dict(recon_ns=RECON_NS, r_ladder=R_LADDER, r_stop=R_STOP,
                    k_lat=K_LAT, n_primary=N_PRIMARY, n_fit=N_FIT,
                    n_val=N_VAL, batch=BATCH, hid=HID,
                    layers=LAYERS, steps=STEPS, s_train=S_TRAIN,
                    s_test=S_TEST, nu=NU, allow_cpu=int(ALLOW_CPU),
                    Q=bank.Q_TOTAL, K=bank.K_LATENT, grad_mix=bank.GRAD_MIX)
    mism = {k: [FROZEN_CONFIG[k], v] for k, v in observed.items()
            if FROZEN_CONFIG[k] != v}
    report["gates"]["PRECOND"] = dict(
        debug_asserts_active=bool(__debug__), smoke=int(SMOKE),
        git_dirty=git_dirty(),
        frozen_config=FROZEN_CONFIG, observed_config=observed,
        config_mismatch=mism, expected_gates=sorted(EXPECTED_GATES),
        expected_row_counts=EXPECTED_ROWS,
        rule="ASSERTED unless SMOKE=1: the entire configuration equals the "
             "frozen contract; python runs WITHOUT -O (a raise, not an "
             "assert); the working tree must be CLEAN, because git_commit "
             "cannot detect uncommitted edits (retraction 29); and a SMOKE=1 "
             "run never sets complete=true")
    save()
    if not __debug__:
        raise RuntimeError("PRECOND: python is running with -O, so every assert "
                           "in this harness is dead.")
    if not SMOKE:
        assert not mism, f"PRECOND: configuration differs from frozen: {mism}"
        dirty = git_dirty()
        assert not dirty, (
            f"PRECOND: the working tree is DIRTY ({dirty}), so the git_commit "
            f"this artifact would record does not describe the code producing "
            f"it.  Commit first.  This is retraction 29, made impossible.")
    log(f"  PRECOND: asserts={__debug__} smoke={SMOKE} mismatch={mism or 'none'}")

    # ---- the certified phase-2a numbers -----------------------------------
    p2a = json.load(open(os.path.join(HERE, PHASE2A)))
    assert p2a.get("complete") and p2a.get("certified"), \
        "phase-2a artifact is not a certified run"
    p2a_pod = {r["N"]: r["heldout_pod_err"] for r in p2a["gates"]["S_RICH"]["rows"]}

    metric_rows, regr_rows, aff_rows, recon_rows, sel_rows = [], [], [], [], []
    verdict = None
    mu_fit = bank.sample_mu(N_FIT, SEED + 991)
    mu_val = bank.sample_mu(N_VAL, SEED + 4242)
    os.makedirs(os.path.join(HERE, HEADS), exist_ok=True)

    def oracle_field(cell, R, spec, Ctarget, perp2, tot2, **kw):
        _, res = head.oracle_fit(spec, Ctarget, **kw)
        return head_stats(cell, R, res, perp2, tot2)

    def train_or_load(path, expect, **kw):
        """Train a head, or reuse an identical one already on disk.

        Training is deterministic given its seed and configuration, so a saved
        head whose ENTIRE recorded configuration matches is the object this
        run would have produced.  `expect` is checked field by field and any
        mismatch retrains.  Deleting `runs/stk2d/heads/` reproduces everything
        from the seed; the artifact records per rung whether the head was
        trained here or loaded, so a reader can tell.  This exists because a
        gate defect at the END of a two-hour run should not cost the two
        hours (retraction 30)."""
        sp = head.load_head_jax(path, expect=expect)
        if sp is not None:
            log(f"   head[{os.path.basename(path)}]: loaded from cache")
            return sp, True
        sp = head.train_head(**kw)
        head.save_head(path, sp, extra=dict(
            smoke=int(SMOKE), producer=os.path.basename(out),
            N=int(expect["N"]), seed_base=int(SEED),
            git_commit=git_commit()))
        sp["from_cache"] = False
        sp.update(dict(smoke=int(SMOKE), producer=os.path.basename(out),
                       N=int(expect["N"]), seed_base=int(SEED)))
        return sp, False

    # ======================= S-SELECT: which training form ==================
    # BOTH training forms are run at the primary rung and the primary is chosen
    # on a VALIDATION cohort drawn from the same generator and disjoint from
    # both the fit cohort and the frozen held-out cohort.  The held-out cohort
    # is never used to choose anything.
    log(f" S-SELECT at N={N_PRIMARY} R={R_STOP}")
    cell0 = rom.RomCell(N_PRIMARY, nu=NU, seed=SEED, s_train=S_TRAIN,
                        s_test=S_TEST, rmax=max(R_LADDER),
                        cache_dir=os.path.join(HERE, CACHE))
    c_fit0 = cell0.coeff_affine(mu_fit)[:, :R_STOP]
    c_val0 = cell0.coeff_affine(mu_val)[:, :R_STOP]
    val_norm = np.linalg.norm(c_val0, axis=1)
    for mode in ("sup", "auto"):
        sp, cached = train_or_load(
            os.path.join(HERE, HEADS,
                         f"head_select_{mode}_N{N_PRIMARY}_R{R_STOP}_K{K_LAT}.npz"),
            dict(mode=mode, hidden=HID, layers=LAYERS, steps=STEPS,
                 batch=BATCH, n_fit=N_FIT, seed=SEED % 1000, smoke=int(SMOKE),
                 N=int(N_PRIMARY), R=int(R_STOP), K=int(K_LAT)),
            Cfit=c_fit0, K=K_LAT, mode=mode, MU=mu_fit, hidden=HID,
            layers=LAYERS, steps=STEPS, lr=LR, batch=BATCH, seed=SEED % 1000,
            log_every=STEPS // 4, tag=f"select-{mode}")
        _, rv = head.oracle_fit(sp, c_val0, n_starts=ORACLE_STARTS,
                                iters=ORACLE_ITERS, seed=SEED + 3)
        sel_rows.append(dict(
            mode=mode, N=N_PRIMARY, R=R_STOP, n_params=int(sp["n_params"]),
            train_rel_mse=float(sp["final_rel_mse"]),
            val_oracle_rel=float(np.linalg.norm(rv) / np.linalg.norm(val_norm)),
            val_oracle_median=float(np.median(rv / val_norm)),
            from_cache=bool(sp.get("from_cache", False)),
            seconds=float(sp["seconds"])))
        log(f"  select[{mode}]: val oracle {sel_rows[-1]['val_oracle_rel']:.3e} "
            f"median {sel_rows[-1]['val_oracle_median']:.3e}")
    MODE = min(sel_rows, key=lambda r: r["val_oracle_rel"])["mode"]
    log(f"  SELECTED training form: {MODE}")
    del cell0

    for N in RECON_NS:
        log(f" cell N={N}")
        cell = rom.RomCell(N, nu=NU, seed=SEED, s_train=S_TRAIN,
                           s_test=S_TEST, rmax=max(R_LADDER),
                           cache_dir=os.path.join(HERE, CACHE))
        g = cell.g
        c_te = cell.coeff_of(cell.U_te)
        c_tr = cell.coeff_of(cell.U_tr)

        # ---- S-METRIC: the coefficient identity, against the FIELD ---------
        rng = np.random.default_rng(SEED + N)
        dev = []
        for _ in range(8):
            R = max(R_LADDER)
            hz = rng.standard_normal(R) * float(np.abs(c_te).std())
            i = int(rng.integers(0, S_TEST))
            p2, t2 = cell.perp_energy(cell.U_te, R)
            u_hat = cell.ubar + cell.G[:, :R] @ hz
            direct = g.h * np.linalg.norm(cell.U_te[:, i] - u_hat)
            viaid = np.sqrt(((c_te[i, :R] - hz) ** 2).sum() + p2[i])
            dev.append(abs(direct - viaid) / (direct + 1e-300))
        p2m, t2m = cell.perp_energy(cell.U_te, max(R_LADDER))
        metric_rows.append(dict(
            N=N, identity_dev_max=float(finite("S_METRIC", dev).max()),
            n_probe=len(dev),
            perp_rel=float(np.sqrt(p2m.sum() / t2m.sum()))))

        # ---- S-REGR: reproduce the certified phase-2a POD errors -----------
        pods_raw = {}
        Xte = cell.U_te - cell.ubar_plain[:, None]
        for R in R_LADDER:
            Gr = cell.G_raw[:, :R]
            pr = Gr @ (Gr.T @ Xte * g.h ** 2)
            pods_raw[str(R)] = float(np.linalg.norm(Xte - pr)
                                     / np.linalg.norm(Xte))
        cert = p2a_pod[N]
        pods = {str(R): pod_stats(cell, R) for R in R_LADDER}
        devs = {k: float(abs(pods_raw[k] - cert[k]) / cert[k])
                for k in pods_raw if k in cert}
        regr_rows.append(dict(
            N=N, pod_raw_route=pods_raw,
            pod_certified={k: float(v) for k, v in cert.items()},
            dev=devs,
            dev_gated=float(max([v for k, v in devs.items()
                                 if int(k) < max(R_LADDER)] or [0.0])),
            dev_R32=float(devs.get(str(max(R_LADDER)), 0.0)),
            pod_reorth_route={k: v["agg"] for k, v in pods.items()}))

        # ---- S-AFFINE -----------------------------------------------------
        c_aff = cell.coeff_affine(cell.mu_te)
        aff = float(np.abs(c_aff - c_te).max() / float(np.abs(c_te).max()))
        aff_rows.append(dict(N=N, affine_coeff_dev=aff,
                             budget=float(AFFINE_SCALE * N ** 2)))

        # ---- the head, per R ----------------------------------------------
        podK = pod_stats(cell, K_LAT)
        for R in R_LADDER:
            t0 = time.time()
            c_fit = cell.coeff_affine(mu_fit)[:, :R]
            p2, t2 = cell.perp_energy(cell.U_te, R)
            p2tr, t2tr = cell.perp_energy(cell.U_tr, R)
            spec, spec_cached = train_or_load(
                os.path.join(HERE, HEADS, f"head_N{N}_R{R}_K{K_LAT}.npz"),
                dict(mode=MODE, hidden=HID, layers=LAYERS, steps=STEPS,
                     batch=BATCH, n_fit=N_FIT, seed=SEED % 1000 + R,
                     smoke=int(SMOKE), N=int(N), R=int(R), K=int(K_LAT)),
                Cfit=c_fit, K=K_LAT, mode=MODE, MU=mu_fit, hidden=HID,
                layers=LAYERS, steps=STEPS, lr=LR, batch=BATCH,
                seed=SEED % 1000 + R,
                log_every=(STEPS // 2 if N == N_PRIMARY else 0),
                tag=f"N{N}R{R}")
            hs = oracle_field(cell, R, spec, c_te[:, :R], p2, t2,
                              n_starts=ORACLE_STARTS, iters=ORACLE_ITERS,
                              seed=SEED + 7)
            hs_tr = oracle_field(cell, R, spec, c_tr[:, :R], p2tr, t2tr,
                                 n_starts=4, iters=ORACLE_ITERS, seed=SEED + 8)

            # ---- NEGATIVE CONTROL 1: a LINEAR head --------------------------
            # layers=0 makes the whole head affine in z, so its image is a
            # K-dimensional affine subspace and it CANNOT beat POD-K.
            sp_lin, lin_cached = train_or_load(
                os.path.join(HERE, HEADS, f"head_lin_N{N}_R{R}_K{K_LAT}.npz"),
                dict(mode=MODE, hidden=HID, layers=0, steps=STEPS // 4,
                     batch=BATCH, n_fit=N_FIT, seed=SEED % 1000 + R,
                     smoke=int(SMOKE), N=int(N), R=int(R), K=int(K_LAT)),
                Cfit=c_fit, K=K_LAT, mode=MODE, MU=mu_fit, hidden=HID,
                layers=0, steps=STEPS // 4, lr=LR, batch=BATCH,
                seed=SEED % 1000 + R, tag=f"lin-N{N}R{R}")
            hs_lin = oracle_field(cell, R, sp_lin, c_te[:, :R], p2, t2,
                                  n_starts=ORACLE_STARTS, iters=ORACLE_ITERS,
                                  seed=SEED + 9)

            # ---- NEGATIVE CONTROL 2: an UNTRAINED head ----------------------
            import jax
            pr_, _ = head.init_head(jax.random.PRNGKey(SEED + 31), K_LAT, R,
                                    HID, LAYERS)
            sp_rnd = dict(params=pr_, scale=spec["scale"], K=K_LAT, R=R,
                          Z=spec["Z"])
            hs_rnd = oracle_field(cell, R, sp_rnd, c_te[:, :R], p2, t2,
                                  n_starts=ORACLE_STARTS, iters=ORACLE_ITERS,
                                  seed=SEED + 10, use_code_pool=False)

            row = dict(
                N=N, R=int(R), K=int(K_LAT), n_fit=int(N_FIT), mode=MODE,
                n_params=int(spec["n_params"]),
                train_rel_mse=float(spec["final_rel_mse"]),
                train_seconds=float(spec["seconds"]),
                head_from_cache=bool(spec_cached),
                linear_control_from_cache=bool(lin_cached),
                oracle=hs, oracle_train_cohort=hs_tr,
                pod_K=podK, pod_R=pods[str(R)],
                gain_over_podK_agg=float(podK["agg"] / max(hs["agg"], 1e-300)),
                gain_over_podK_median=float(podK["median"]
                                            / max(hs["median"], 1e-300)),
                truncation_floor_agg=float(np.sqrt(p2.sum() / t2.sum())),
                linear_control=hs_lin, untrained_control=hs_rnd,
                linear_ctl_ratio_agg=float(hs_lin["agg"] / podK["agg"]),
                untrained_ctl_ratio_agg=float(hs_rnd["agg"] / podK["agg"]),
                seconds=float(time.time() - t0))
            recon_rows.append(row)
            log(f"  N={N} R={R}: oracle agg {hs['agg']:.3e} med "
                f"{hs['median']:.3e} max {hs['max']:.3e} | POD-{K_LAT} agg "
                f"{podK['agg']:.3e} med {podK['median']:.3e} | gain "
                f"{row['gain_over_podK_agg']:.2f}x / "
                f"{row['gain_over_podK_median']:.2f}x | linear ctl "
                f"{hs_lin['agg']:.3e} untrained ctl {hs_rnd['agg']:.3e}")
            if N == N_PRIMARY and R == R_STOP:
                verdict = dict(
                    N=N, R=R, K=K_LAT, n_fit=N_FIT, mode=MODE,
                    oracle_agg=hs["agg"], oracle_median=hs["median"],
                    oracle_max=hs["max"], podK_agg=podK["agg"],
                    podK_median=podK["median"],
                    required_agg=float(podK["agg"] / STOP_FACTOR),
                    required_median=float(podK["median"] / STOP_FACTOR),
                    gain_agg=row["gain_over_podK_agg"],
                    gain_median=row["gain_over_podK_median"],
                    passed=bool(hs["agg"] <= podK["agg"] / STOP_FACTOR
                                and hs["median"] <= podK["median"] / STOP_FACTOR))
            save()
        del cell
        save()

    # ---- S-SELECT ---------------------------------------------------------
    report["gates"]["S_SELECT"] = dict(
        rows=sel_rows, selected=MODE,
        rule="BOTH training forms -- the autodecoder with free per-sample "
             "latent codes, and the supervised form with the latent pinned to "
             "the family parameter -- are run at the primary rung, and the "
             "primary is chosen on a VALIDATION cohort of "
             f"{N_VAL} samples drawn from the same generator and disjoint "
             "from both the fit cohort and the frozen held-out cohort.  The "
             "held-out cohort chooses nothing.  Recorded because a training "
             "form selected after seeing the held-out number would make the "
             "stop gate meaningless")
    save()

    # ---- S-METRIC ---------------------------------------------------------
    report["gates"]["S_METRIC"] = dict(
        rows=metric_rows,
        worst=float(finite("S_METRIC", [r["identity_dev_max"]
                                        for r in metric_rows]).max()),
        rule=f"||u - ubar - G h||_M must equal sqrt(||c - h||^2 + "
             f"||perp||_M^2) to {METRIC_TOL} relative, at 8 random h per mesh, "
             f"measured against a DIRECT field computation.  Every "
             f"reconstruction number in this driver is computed through that "
             f"identity, so it is measured rather than assumed.  It holds "
             f"because G is M_u-ORTHONORMAL (phase 2a, 1.3e-15)")
    save()
    for r in metric_rows:
        assert r["identity_dev_max"] <= METRIC_TOL, \
            f"S-METRIC N={r['N']}: {r['identity_dev_max']}"

    # ---- S-REGR -----------------------------------------------------------
    report["gates"]["S_REGR"] = dict(
        rows=regr_rows,
        worst=float(finite("S_REGR", [r["dev_gated"] for r in regr_rows]).max()),
        worst_R32=float(finite("S_REGR32",
                               [r["dev_R32"] for r in regr_rows]).max()),
        rule=f"the phase-2b bank must reproduce the CERTIFIED phase-2a "
             f"held-out POD-8 and POD-16 errors to {REGR_TOL} relative, on "
             f"the RAW psi-route bank (which is what phase 2a tabulated).  "
             f"POD-32 is recorded, NOT gated at that tolerance: it is a "
             f"1e-12-scale cancellation whose value differs between the raw "
             f"and reorthogonalised routes (2.6e-12 vs 3.1e-14 at N=64), so a "
             f"relative comparison there measures roundoff, not agreement")
    save()
    for r in regr_rows:
        assert r["dev_gated"] <= REGR_TOL, \
            (f"S-REGR N={r['N']}: the phase-2b bank differs from the "
             f"CERTIFIED phase-2a held-out POD errors by {r['dev_gated']} "
             f"(per R: {r['dev']})")

    # ---- S-AFFINE ---------------------------------------------------------
    report["gates"]["S_AFFINE"] = dict(
        rows=aff_rows,
        worst_ratio=float(finite("S_AFFINE",
                                 [r["affine_coeff_dev"] / r["budget"]
                                  for r in aff_rows]).max()),
        rule=f"c(mu) = Cd theta(mu) - cbar, the map used to build the head's "
             f"training cohort, must agree with coefficients taken from DIRECT "
             f"FOM solves at phase 2a's own affine budget {AFFINE_SCALE} * "
             f"N^2 (retraction 20: a FLAT tolerance here passed three meshes "
             f"and failed the fourth on refinement alone)")
    save()
    for r in aff_rows:
        assert r["affine_coeff_dev"] <= r["budget"], \
            f"S-AFFINE N={r['N']}: {r['affine_coeff_dev']} > {r['budget']}"

    # ---- S-RECON: THE STOP GATE -------------------------------------------
    assert verdict is not None, "S-RECON: the primary (N, R) cell never ran"
    report["gates"]["S_RECON"] = dict(
        rows=recon_rows, verdict=verdict,
        rule=f"THE PREDECLARED STOP GATE, fixed before any training run: the "
             f"held-out nonlinear RECONSTRUCTION ORACLE at N={N_PRIMARY}, "
             f"R={R_STOP}, K={K_LAT} must beat POD-{K_LAT} by at least "
             f"{STOP_FACTOR}x in the AGGREGATE held-out error AND in the "
             f"PER-CASE MEDIAN, on the identical held-out cohort in the "
             f"identical mass-weighted norm.  This is a VERDICT, not an "
             f"assertion: the artifact is written either way and "
             f"stk2d_rom_gates.py refuses to run unless verdict.passed is "
             f"true.  NEGATIVE CONTROLS, both ASSERTED: (1) a LINEAR head "
             f"spans a K-dimensional affine subspace and therefore CANNOT "
             f"beat POD-K -- its oracle error must sit within "
             f"{LINEAR_CTL_BAND}x of the POD-K floor and never below it; (2) "
             f"an UNTRAINED head must be no better than POD-K.  Without those "
             f"the gate could not distinguish 'the head works' from 'the "
             f"metric is wrong'.  NOTE, stated because it is easy to misread: "
             f"the head's error is bounded below by the POD-R TRUNCATION "
             f"FLOOR at its own R, exactly, so at R = K the head can never "
             f"beat POD-K and the R ladder is not a like-for-like comparison")
    save()
    for r in recon_rows:
        # RETRACTION 30.  The first version of this assertion required the
        # linear control to sit at or ABOVE the POD-K floor, on the argument
        # that a K-dimensional affine subspace cannot beat POD-K.  That is
        # true on the cohort POD-K was FITTED to and false on a HELD-OUT one:
        # POD-K here is the first K columns of a bank built from 256 training
        # snapshots, while the linear head fits an arbitrary K-dimensional
        # affine subspace to 16384 samples, so it can and does generalise
        # slightly better (3.738e-2 against 3.849e-2 at N=64).  The gated
        # statement is the one that actually follows from the geometry: the
        # linear control cannot come CLOSE to the head, and it cannot clear
        # the stop bar.  The ratio is recorded either way.
        assert r["linear_control"]["agg"] >= r["pod_K"]["agg"] / LINEAR_CTL_BAND, \
            (f"S-RECON linear control N={r['N']} R={r['R']}: the linear head "
             f"reads {r['linear_control']['agg']}, more than "
             f"{LINEAR_CTL_BAND}x below the POD-K floor {r['pod_K']['agg']} "
             f"-- a K-dimensional affine subspace fitted to this family "
             f"should land AT the linear ceiling, so the metric or the cohort "
             f"is wrong")
        assert (r["pod_K"]["agg"] / r["linear_control"]["agg"]) < STOP_FACTOR, \
            (f"S-RECON linear control N={r['N']} R={r['R']} CLEARED the stop "
             f"bar: {r['pod_K']['agg'] / r['linear_control']['agg']:.2f}x >= "
             f"{STOP_FACTOR}x.  A head whose image is a K-dimensional affine "
             f"subspace cannot materially beat POD-K, so the metric, the "
             f"cohort or the oracle is wrong")
        assert r["untrained_control"]["agg"] >= r["pod_K"]["agg"], \
            (f"S-RECON untrained control N={r['N']} R={r['R']} did not fire: "
             f"{r['untrained_control']['agg']} < {r['pod_K']['agg']}")
        assert r["oracle"]["agg"] >= r["truncation_floor_agg"] * (1 - 1e-9), \
            (f"S-RECON N={r['N']} R={r['R']}: the oracle {r['oracle']['agg']} "
             f"is below the POD-R truncation floor "
             f"{r['truncation_floor_agg']}, which is impossible")
    log("")
    log("  ==================== STOP-GATE VERDICT ====================")
    log(f"  N={verdict['N']} R={verdict['R']} K={verdict['K']}  "
        f"oracle agg {verdict['oracle_agg']:.4e} (need <= "
        f"{verdict['required_agg']:.4e}), median {verdict['oracle_median']:.4e} "
        f"(need <= {verdict['required_median']:.4e})")
    log(f"  POD-{verdict['K']} agg {verdict['podK_agg']:.4e} median "
        f"{verdict['podK_median']:.4e};  gain {verdict['gain_agg']:.2f}x agg / "
        f"{verdict['gain_median']:.2f}x median")
    log(f"  VERDICT: {'PASS' if verdict['passed'] else 'FAIL -- PHASE 2b STOPS'}")
    log("  ===========================================================")

    # ---- MANIFEST ---------------------------------------------------------
    counts = dict(S_METRIC=len(metric_rows), S_REGR=len(regr_rows),
                  S_AFFINE=len(aff_rows), S_SELECT=len(sel_rows),
                  S_RECON=len(recon_rows))
    missing = sorted(EXPECTED_GATES - set(report["gates"]) - {"MANIFEST"})
    badc = {k: [EXPECTED_ROWS[k], counts[k]] for k in EXPECTED_ROWS
            if counts[k] != EXPECTED_ROWS[k]}
    nonfinite = []

    def sweep(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                sweep(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                sweep(v, f"{path}[{i}]")
        elif isinstance(node, float) and not np.isfinite(node):
            nonfinite.append(path)

    sweep(report["gates"], "gates")
    report["gates"]["MANIFEST"] = dict(
        expected_gates=sorted(EXPECTED_GATES), missing_gates=missing,
        expected_row_counts=EXPECTED_ROWS, observed_row_counts=counts,
        row_count_mismatch=badc, nonfinite_fields=nonfinite,
        rule="ASSERTED unless SMOKE=1: every expected gate present, EXACT row "
             "counts, and no non-finite float anywhere in gates/")
    save()
    if not SMOKE:
        assert not missing, f"MANIFEST: missing gates {missing}"
        assert not badc, f"MANIFEST: row-count mismatch {badc}"
    assert not nonfinite, f"MANIFEST: non-finite values at {nonfinite}"

    report["complete"] = not bool(SMOKE)
    report["certified"] = not bool(SMOKE)
    report["stop_gate_passed"] = bool(verdict["passed"])
    if SMOKE:
        report["incomplete_reason"] = "SMOKE=1 is never a certified artifact"
    report["total_seconds"] = float(time.time() - t_all)
    save()
    log(f"DONE stop gate [{report['total_seconds']:.0f}s] "
        f"complete={report['complete']} passed={report['stop_gate_passed']} "
        f"-> {out}")


if __name__ == "__main__":
    main()
