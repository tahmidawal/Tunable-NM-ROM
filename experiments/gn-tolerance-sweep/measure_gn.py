"""Measure GN iterations-to-tolerance of the real NM-ROM solvers.

Design (v3, after adversarial review + smoke findings):

Replica-based history recording is UNRELIABLE: the GN body's backtracking
line search takes a discrete argmin, so scan-vs-while floating-point
differences flip step choices and the chains diverge (observed: probe 4%
above threshold at the iteration where the real loop crossed). Histories are
therefore extracted from the REAL solver itself, same program semantics by
construction:

    the while-loop carry's gnorm at exit equals ||g|| at the pre-update
    iterate of the last executed body, so running the real solver EAGERLY
    with (gn_rel_tol=0, gn_max_iters=j) returns exactly gnorms[j-1] of the
    true chain. Sweeping j = 1..PROBE_ITERS reconstructs the exact history.

POISSON — independent cold starts. Per val solve: exact history via the
j-sweep; real eager solves at each tolerance in TOLERANCES (max=PROBE_ITERS)
for actual counts and the model rel-L2 at each stopping iterate. VALIDATION
(hard): history-derived counts must exactly match the real counts — same
program, so any mismatch is a logic bug.

HEAT — warm-started rollout: chains depend on the stopping tolerance, so the
real JITTED rollout runs separately per tolerance (per-step counts + final
rel-L2). Histories + hard validation are done at STEP 0 (whose start state is
tolerance-independent) via a verbatim gnorm-returning copy of _step run
eagerly with the j-sweep, validated against eager per-tolerance _step counts.
Jitted-rollout step-0 counts are cross-checked and any jit-vs-eager
discrepancies are reported as xprog warnings (fp, not logic), not failures.

Iteration-count semantics (both packages; gnorm0 = max(gnorms[0], 1e-30)):
    iters(tol, max) = min over j>=0 with gnorms[j] <= tol*gnorm0 of (j+1),
                      else max.       (j=0 reachable: heat warm starts can
                                       have exactly-zero start residual.)

Usage:
  python measure_gn.py --package {poisson,heat} --config C --data D --ckpt K \
      --out out.json [--num-solves N] [--note "..."]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

import jax
import jax.numpy as jnp

PROBE_ITERS = int(os.environ.get("MAX_ITERS", "30"))
TOLERANCES = [1e-2, 1e-3, 1e-4]


def iters_to_tol(gnorms: np.ndarray, tol: float, max_iters: int) -> int:
    """Replicates the packages' while-loop count for a given tolerance."""
    gnorm0 = max(float(gnorms[0]), 1e-30)
    hit = np.where(np.asarray(gnorms) <= tol * gnorm0)[0]
    if len(hit) == 0:
        return int(max_iters)
    return int(min(hit[0] + 1, max_iters))


# --------------------------------------------------------------------------
def run_poisson(args):
    sys.path.insert(0, os.path.join(WT_ROOT, "poisson", "src"))
    from tunable_rom_poisson.models.autoencoder import ViTLinearCPAutoencoder
    from tunable_rom_poisson.fom.poisson import PoissonFOM, source_field
    from tunable_rom_poisson.eq.nnls import compute_eq_weights, build_v_eq
    from tunable_rom_poisson.solver.nm_rom import NMROMSolver
    from tunable_rom_poisson.utils.config import load_config
    from tunable_rom_poisson.utils.training import load_checkpoint

    cfg = load_config(args.config)
    data = np.load(args.data)
    U_train, U_val = data["U_train"], data["U_val"]
    freqs_val = data["freqs_val"]
    params = load_checkpoint(args.ckpt)["params"]

    model = ViTLinearCPAutoencoder(
        N=cfg.N, spatial_dim=cfg.spatial_dim, patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
        num_enc_layers=cfg.num_enc_layers, latent_dim=cfg.latent_dim,
        rank=cfg.rank, hidden_dim=cfg.hidden_dim)

    fom = PoissonFOM(N=cfg.N, spatial_dim=cfg.spatial_dim)
    K_op_jit = jax.jit(lambda u: fom.K_op(u))
    eq_flat, eq_w = compute_eq_weights(
        snapshots=U_train, K_op_numpy=lambda u: np.asarray(K_op_jit(jnp.asarray(u))),
        N=cfg.N, spatial_dim=cfg.spatial_dim,
        n_eq_samples=cfg.n_eq_samples, min_eq_points=cfg.min_eq_points)
    v_eq_st, stencil_idx = build_v_eq(params["decoder"], eq_flat, cfg.N, cfg.spatial_dim)

    def make_solver(tol, max_iters):
        return NMROMSolver(
            autoencoder=model, params=params, N=cfg.N, spatial_dim=cfg.spatial_dim,
            dx=fom.dx, eq_flat_indices=eq_flat, eq_weights=eq_w,
            v_eq_stencil=v_eq_st, stencil_indices=stencil_idx,
            gn_max_iters=max_iters, gn_rel_tol=tol)

    latent_dim = params["decoder"]["W_direct"]["kernel"].shape[0]
    solvers_tol = {t: make_solver(t, PROBE_ITERS) for t in TOLERANCES}
    # gnorms[j-1] of the true chain = real solver run with (tol=0, max=j).
    solvers_hist = [make_solver(0.0, j) for j in range(1, PROBE_ITERS + 1)]

    n = min(args.num_solves or U_val.shape[0], U_val.shape[0])
    histories = []
    iters_per_tol = {t: [] for t in TOLERANCES}
    rel_per_tol = {t: [] for t in TOLERANCES}
    ok = True
    for i in range(n):
        F_full = source_field(fom, list(freqs_val[i]))
        F_eq = F_full[eq_flat]
        gnorms = np.array([float(sv.solve(F_eq)[1]) for sv in solvers_hist])
        histories.append([float(g) for g in gnorms])
        u_true = jnp.asarray(U_val[i])
        for t in TOLERANCES:
            z, _, iters = solvers_tol[t].solve(F_eq)
            iters = int(iters)
            derived = iters_to_tol(gnorms, t, PROBE_ITERS)
            if derived != iters:
                ok = False
                print(f"  sample {i} tol={t:g}: MISMATCH derived={derived} real={iters}")
            u_rom = solvers_tol[t].decode(z)
            rel_per_tol[t].append(float(jnp.linalg.norm(u_rom - u_true)
                                        / jnp.linalg.norm(u_true)))
            iters_per_tol[t].append(iters)
        if i == 0:
            print(f"  sample 0 iters per tol: "
                  f"{[iters_per_tol[t][0] for t in TOLERANCES]}", flush=True)
    print(f"VALIDATION {'OK' if ok else 'FAIL'} (poisson, {n} solves x "
          f"{len(TOLERANCES)} tolerances, probe={PROBE_ITERS})")
    return {
        "package": "poisson", "N": int(cfg.N), "latent_dim": int(latent_dim),
        "backend": jax.default_backend(), "probe_iters": PROBE_ITERS,
        "tolerances": TOLERANCES,
        "layout": "histories[solve][iter] (exact, real-solver j-sweep); "
                  "iters_per_tol[tol][solve]",
        "histories": histories,
        "iters_per_tol": {f"{t:g}": v for t, v in iters_per_tol.items()},
        "rel_l2_per_tol": {f"{t:g}": float(np.mean(v))
                           for t, v in rel_per_tol.items()},
        "model_rel_l2": float(np.mean(rel_per_tol[TOLERANCES[-1]])),
        "n_eq": int(eq_flat.shape[0]), "validation_ok": ok,
        "note": args.note,
    }


# --------------------------------------------------------------------------
def run_heat(args):
    sys.path.insert(0, os.path.join(WT_ROOT, "heat", "src"))
    from tunable_rom_heat.models.autoencoder import ViTCPAutoencoder
    from tunable_rom_heat.fom.heat import HeatFOM, NUM_STEPS
    from tunable_rom_heat.eq.nnls import compute_eq_weights, build_v_eq
    from tunable_rom_heat.solver.nm_rom import NMROMSolver
    from tunable_rom_heat.utils.config import load_config
    from tunable_rom_heat.utils.training import load_checkpoint

    cfg = load_config(args.config)
    data = np.load(args.data)
    U_train, U_val = data["U_train"], data["U_val"]
    val_kappa = data["val_kappa"]
    params = load_checkpoint(args.ckpt)["params"]

    model = ViTCPAutoencoder(
        N=cfg.N, spatial_dim=cfg.spatial_dim, patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
        num_enc_layers=cfg.num_enc_layers, latent_dim=cfg.latent_dim,
        rank=cfg.rank, hidden_dim=cfg.hidden_dim)

    eq_flat, eq_w = compute_eq_weights(
        model=model, params=params, snapshots=U_train, N=cfg.N,
        spatial_dim=cfg.spatial_dim, n_eq_samples=cfg.n_eq_samples,
        min_eq_points=cfg.min_eq_points)
    v_eq_st, stencil_idx = build_v_eq(params["decoder"], eq_flat, cfg.N, cfg.spatial_dim)
    fom = HeatFOM(N=cfg.N, spatial_dim=cfg.spatial_dim)

    def make_solver(tol, max_iters):
        return NMROMSolver(
            autoencoder=model, params=params, N=cfg.N, spatial_dim=cfg.spatial_dim,
            dx=fom.dx, eq_flat_indices=eq_flat, eq_weights=eq_w,
            v_eq_stencil=v_eq_st, stencil_indices=stencil_idx,
            gn_max_iters=max_iters, gn_rel_tol=tol)

    num_steps = int(getattr(cfg, "num_rom_steps", NUM_STEPS))
    k = int(cfg.latent_dim)
    solvers_tol = {t: make_solver(t, PROBE_ITERS) for t in TOLERANCES}
    rollouts_tol = {t: jax.jit(lambda u0, kp, s=solvers_tol[t]:
                               s.rollout(u0, kp, num_steps)) for t in TOLERANCES}
    solvers_hist = [make_solver(0.0, j) for j in range(1, PROBE_ITERS + 1)]

    def step_with_gnorm(sv, z, scale, u_prev_eq, kappa):
        """Verbatim copy of NMROMSolver._step that also returns the carry's
        exit gnorm (= ||g|| at the pre-update iterate of the last body)."""
        def loss_only(z_s):  # noqa: F841  (kept to mirror _step verbatim)
            z_local, s_local = z_s[:-1], z_s[-1]
            R = sv.residual(z_local, s_local, u_prev_eq, kappa)
            return 0.5 * jnp.sum(sv.w_eq * R**2)

        def step_body(carry):
            z_s, gnorm0, gnorm, itr = carry
            zc, sc = z_s[:-1], z_s[-1]
            R = sv.residual(zc, sc, u_prev_eq, kappa)
            J_z = jax.jacfwd(lambda zz: sv.residual(zz, sc, u_prev_eq, kappa))(zc)
            f_norm_vec = sv.f_norm_eq(zc, kappa)
            J = jnp.concatenate([J_z, f_norm_vec[:, None]], axis=1)
            JtW = J.T * sv.w_eq[None, :]
            H = JtW @ J
            g = JtW @ R
            damp = jnp.maximum(sv.lm_damping * jnp.trace(H) / (z_s.size), 1e-8)
            dz_s = jnp.linalg.solve(H + damp * jnp.eye(z_s.size), -g)
            steps = jnp.asarray([1.0, 0.5, 0.25, 0.125])
            def try_step(a):
                cand = z_s + a * dz_s
                Rc = sv.residual(cand[:-1], cand[-1], u_prev_eq, kappa)
                return 0.5 * jnp.sum(sv.w_eq * Rc**2)
            losses = jax.vmap(try_step)(steps)
            best = jnp.argmin(losses)
            z_s_new = z_s + steps[best] * dz_s
            gnew = jnp.linalg.norm(g)
            return (z_s_new, gnorm0, gnew, itr + 1)

        def cond(carry):
            z_s, gnorm0, gnorm, itr = carry
            return jnp.logical_and(gnorm > sv.gn_rel_tol * gnorm0,
                                   itr < sv.gn_max_iters)

        zR0 = sv.residual(z, scale, u_prev_eq, kappa)
        zJ_z0 = jax.jacfwd(lambda zz: sv.residual(zz, scale, u_prev_eq, kappa))(z)
        f0 = sv.f_norm_eq(z, kappa)
        J0 = jnp.concatenate([zJ_z0, f0[:, None]], axis=1)
        g0 = (J0.T * sv.w_eq[None, :]) @ zR0
        gnorm0 = jnp.maximum(jnp.linalg.norm(g0), 1e-30)
        z_s0 = jnp.concatenate([z, jnp.array([scale])])
        z_s_f, _, gnorm_f, iters = jax.lax.while_loop(
            cond, step_body, (z_s0, gnorm0, gnorm0, 0))
        return z_s_f[:-1], z_s_f[-1], iters, gnorm_f

    val_starts = np.arange(0, U_val.shape[0], NUM_STEPS + 1)
    n = min(args.num_solves or len(val_starts), len(val_starts))
    step0_histories = []
    iters_per_tol = {t: [] for t in TOLERANCES}
    rel_per_tol = {t: [] for t in TOLERANCES}
    ok = True
    xprog_mismatches = 0
    for i in range(n):
        u0 = jnp.asarray(U_val[val_starts[i]])
        kappa = jnp.float32(val_kappa[i])
        u_true = jnp.asarray(U_val[val_starts[i] + NUM_STEPS])
        z0, scale0 = model.apply({"params": params}, u0, method=model.encode)
        u0_eq = u0[jnp.asarray(eq_flat)]

        # Exact step-0 history: eager real while-loop, (tol=0, max=j) sweep.
        gnorms0 = np.array([
            float(step_with_gnorm(sv, z0, scale0, u0_eq, kappa)[3])
            for sv in solvers_hist])
        step0_histories.append([float(g) for g in gnorms0])

        for t in TOLERANCES:
            # Real jitted rollout at this tolerance (the measurement).
            u_rom, iters_buf = rollouts_tol[t](u0, kappa)
            iters_buf = np.asarray(iters_buf)
            iters_per_tol[t].append([int(x) for x in iters_buf])
            rel_per_tol[t].append(float(jnp.linalg.norm(u_rom - u_true)
                                        / jnp.linalg.norm(u_true)))
            # Hard validation: derived vs eager real step-0 count (same
            # program family as the history sweep).
            _, _, it_eager, _ = step_with_gnorm(
                solvers_tol[t], z0, scale0, u0_eq, kappa)
            derived0 = iters_to_tol(gnorms0, t, PROBE_ITERS)
            if derived0 != int(it_eager):
                ok = False
                print(f"  traj {i} tol={t:g}: step-0 MISMATCH "
                      f"derived={derived0} eager-real={int(it_eager)}")
            # Cross-program check (jit rollout vs eager): warn only.
            if int(iters_buf[0]) != int(it_eager):
                xprog_mismatches += 1
    if xprog_mismatches:
        print(f"  note: {xprog_mismatches} jit-vs-eager step-0 count "
              f"discrepancies (fp near threshold; recorded, not a failure)")
    print(f"VALIDATION {'OK' if ok else 'FAIL'} (heat, {n} trajectories x "
          f"{len(TOLERANCES)} tolerances, {num_steps} steps, probe={PROBE_ITERS})")
    return {
        "package": "heat", "N": int(cfg.N), "latent_dim": k,
        "backend": jax.default_backend(), "probe_iters": PROBE_ITERS,
        "tolerances": TOLERANCES,
        "layout": ("step0_histories[traj][iter] (exact, eager j-sweep); "
                   "iters_per_tol[tol][traj][step] (real jitted rollouts, "
                   "real stopping rule at that tolerance)"),
        "step0_histories": step0_histories,
        "iters_per_tol": {f"{t:g}": v for t, v in iters_per_tol.items()},
        "rel_l2_per_tol": {f"{t:g}": float(np.mean(v))
                           for t, v in rel_per_tol.items()},
        "model_rel_l2": float(np.mean(rel_per_tol[TOLERANCES[-1]])),
        "n_eq": int(eq_flat.shape[0]), "validation_ok": ok,
        "xprog_step0_mismatches": xprog_mismatches,
        "note": args.note,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", required=True, choices=["poisson", "heat"])
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-solves", type=int, default=None)
    p.add_argument("--note", default="")
    args = p.parse_args()

    print(f"jax_backend={jax.default_backend()}", flush=True)
    result = run_poisson(args) if args.package == "poisson" else run_heat(args)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.out}  (rel_l2_per_tol={result['rel_l2_per_tol']}, "
          f"n_eq={result['n_eq']}, validation_ok={result['validation_ok']})")
    sys.exit(0 if result["validation_ok"] else 1)


if __name__ == "__main__":
    main()
