"""Genuine weak FiLM NM-ROM construction for the final negative control.

The decoder and trajectory-wise weak LSPG solve are the audited K=8 Burgers
NM-ROM.  EQ weights are refit from decoder-output snapshots at every mesh with
M=64 and m=256, retaining the exact FOM upwind stencil.  The old online cold
start searched a 512-by-N^2 training-field bank; this optimized control instead
maps moments derived from the supplied u0 (plus known nu) to a latent start and
charges those reductions.  It still performs the weak reduced PDE solve at all
50 steps and is therefore a genuine NM-ROM, not a direct surrogate.  The 50
latent states are decoded only on a fixed 64x64 grid and bilinearly prolonged
inside the charged end-to-end JAX path.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_FILM_CANDIDATES = (
    os.path.join(HERE, "deps", "burgers2d-rom-latent-stepping"),
    os.path.join(_WT_ROOT, "2026-08-16-burgers2d-rom-latent-stepping",
                 "experiments", "burgers2d-rom-latent-stepping"),
    os.path.join(os.path.dirname(HERE), "burgers2d-rom-latent-stepping"),
)
for _film_dir in _FILM_CANDIDATES:
    if os.path.isfile(os.path.join(_film_dir, "blat_common.py")):
        sys.path.insert(0, os.path.join(_film_dir, "followup"))
        sys.path.insert(0, _film_dir)
        break
else:
    raise ImportError("cannot locate burgers2d-rom-latent-stepping dependencies")

import blat_common as rc  # noqa: E402
import fu_common as fu  # noqa: E402

F64 = jnp.float64
EQ_RNG_SEED = 4321
VARIANT = "lspg:eq256:weak64"
EQ_GRID_POOL = int(os.environ.get("EQ_GRID_POOL", "4096"))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_bounded_grid_eq(decoder, n, mode_count, point_count, z_snapshots):
    """Fit EQ on a deterministic tensor subset of target-grid candidates.

    The projection targets still use every interior node and exact discrete
    upwind advection.  Only the candidate columns passed to NNLS are bounded,
    avoiding an impossible 8192-by-one-million host matrix at N=1024.  Every
    retained quadrature node is a target-grid node with its full FOM stencil.
    """
    started = time.time()
    _, _, phi, _, _ = rc.test_modes(n, mode_count)
    coords = jnp.asarray(rc.grid_coords(n))
    interior = rc.interior_indices(n)
    interior_side = n - 2
    pool_side = min(interior_side, int(np.floor(np.sqrt(EQ_GRID_POOL))))
    axis = np.unique(np.rint(np.linspace(1, n - 2, pool_side)).astype(np.int64))
    ii, jj = np.meshgrid(axis, axis, indexing="ij")
    candidate_indices = (ii * n + jj).reshape(-1)
    positions = np.searchsorted(interior, candidate_indices)
    if not np.all(interior[positions] == candidate_indices):
        raise ValueError("bounded EQ candidate is not on the interior grid")
    phi_candidate = np.asarray(phi[positions])

    full_coords = coords[jnp.asarray(interior)]
    stencils = rc.stencil_indices(candidate_indices, n)
    stencil_coords = coords[jnp.asarray(stencils.reshape(-1))]
    dx = 1.0 / (n - 1)

    @jax.jit
    def snapshot_values(z, full_points, stencil_points):
        full = decoder(z, full_points)
        stencil = decoder(z, stencil_points).reshape(-1, 5)
        center, xp, xm, yp, ym = [stencil[:, column] for column in range(5)]
        ux = jnp.where(center > 0.0, (center - xm) / dx, (xp - center) / dx)
        uy = jnp.where(center > 0.0, (center - ym) / dx, (yp - center) / dx)
        return full, center, center * (ux + uy)

    blocks, targets = [], []
    phi_t = np.asarray(phi).T
    phi_candidate_t = phi_candidate.T
    for z in z_snapshots:
        full, candidate_u, candidate_advection = [
            np.asarray(value) for value in snapshot_values(
                jnp.asarray(z, F64), full_coords, stencil_coords
            )
        ]
        full_advection = np.asarray(rc.upwind_adv_field(jnp.asarray(full), n))
        for full_value, candidate_value in (
            (full, candidate_u),
            (full_advection, candidate_advection),
        ):
            targets.append(phi_t @ full_value)
            blocks.append(phi_candidate_t * candidate_value[None, :])

    matrix = np.concatenate(blocks, axis=0)
    target = np.concatenate(targets)
    row_scale = np.linalg.norm(matrix, axis=1) + 1e-300
    matrix = matrix / row_scale[:, None]
    target = target / row_scale
    weights, capped_norm, outer = rc.nnls_capped(
        matrix, target, max_support=point_count
    )
    support = np.nonzero(weights > 0)[0]
    padded = 0
    if support.size >= point_count:
        keep = support[np.argsort(-weights[support])[:point_count]]
    else:
        rest = np.setdiff1d(np.arange(candidate_indices.size), support)
        score = np.abs(matrix).mean(axis=0)
        pad = rest[np.argsort(-score[rest])[:point_count - support.size]]
        keep = np.concatenate((support, pad))
        padded = int(pad.size)
    final_weights, final_norm, _ = rc.nnls_capped(
        matrix[:, keep], target, max_support=keep.size
    )
    final_weights = np.where(
        final_weights > 0,
        final_weights,
        1e-8 * max(final_weights.max(), 1e-300),
    )
    residual_rows = matrix[:, keep] @ final_weights - target
    relative_rows = np.abs(residual_rows) / (np.abs(target) + 1e-300)
    info = {
        "support": int(support.size),
        "padded": padded,
        "rnorm_capped": float(capped_norm),
        "rnorm_final": float(np.linalg.norm(residual_rows)),
        "b_norm": float(np.linalg.norm(target)),
        "rel_fit": float(np.linalg.norm(residual_rows) / np.linalg.norm(target)),
        "n_rows": int(matrix.shape[0]),
        "row_rel_median": float(np.median(relative_rows)),
        "row_rel_p95": float(np.quantile(relative_rows, 0.95)),
        "row_rel_max": float(np.max(relative_rows)),
        "n_cand": int(candidate_indices.size),
        "n_interior": int(interior.size),
        "candidate_strategy": "deterministic_uniform_tensor_target_grid",
        "candidate_cap": EQ_GRID_POOL,
        "exact_full_grid_projection_targets": True,
        "exact_upwind_candidate_stencils": True,
        "secs": float(time.time() - started),
        "M": int(mode_count),
        "m": int(keep.size),
        "kind": "weak",
        "pool": "bounded_grid",
        "nnls_outer": int(outer),
    }
    rc.log(
        f"  bounded-grid NNLS-EQ M={mode_count} m={keep.size}: "
        f"candidates {candidate_indices.size}/{interior.size}, "
        f"rel fit {info['rel_fit']:.2e} "
        f"(p95 {info['row_rel_p95']:.1e}) [{info['secs']:.0f}s]"
    )
    return {
        "kind": "grid",
        "idx": candidate_indices[keep],
        "w": final_weights,
        "info": info,
    }


def build_ops(decoder, n, z_snapshots):
    """Audited weak64/EQ256 path, with scalable per-N NNLS refit."""
    solver, collocation_name, objective = VARIANT.split(":")
    kind = "weak"
    mode_count = int(objective[len(kind):])
    point_count = int(collocation_name[2:])
    collocation = fit_bounded_grid_eq(
        decoder, n, mode_count, point_count, z_snapshots
    )
    return rc.make_weak_ops(
        decoder, n, collocation, kind=kind, M=mode_count, solver=solver
    )


class FilmControl:
    """Checkpoint plus per-resolution weak operators and charged constructor."""

    def __init__(self, checkpoint_path, decode_chunk=2, decode_resolution=64):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = file_sha256(checkpoint_path)
        self.decode_chunk = int(decode_chunk)
        self.decode_resolution = int(decode_resolution)
        with open(checkpoint_path, "rb") as handle:
            checkpoint = pickle.load(handle)
        for key, value in (
            ("bc_mode", rc.BC_MODE),
            ("N", rc.N),
            ("ad_hidden", rc.AD_HIDDEN),
            ("ad_layers", rc.AD_LAYERS),
            ("n_train", rc.N_TRAIN),
            ("seed", rc.SEED),
        ):
            if checkpoint["config"][key] != value:
                raise ValueError(
                    f"checkpoint config mismatch {key}: "
                    f"{checkpoint['config'][key]} vs {value}"
                )
        self.checkpoint_config = checkpoint["config"]
        self.decoder = rc.CoordDecoder(
            jax.tree_util.tree_map(jnp.asarray, checkpoint["params"]),
            checkpoint["n_freq"],
            checkpoint["eps"],
            checkpoint["k_lat"],
        )
        self.z_train = np.asarray(checkpoint["Z_train"], np.float64)
        total = self.z_train.shape[0] * self.z_train.shape[1]
        indices = np.random.default_rng(EQ_RNG_SEED).choice(
            total, rc.EQ_SNAPS, replace=False
        )
        self.z_snapshots = self.z_train.reshape(-1, self.decoder.k)[indices]

        # Offline latent initializer: generator parameters are available only
        # while training.  Online inputs are the same five summaries recovered
        # from u0 plus nu; no hidden test metadata enters the deployed path.
        *_, z_parameters = rc.bf.sample_params()
        design = np.concatenate(
            [np.ones((rc.N_TRAIN, 1)), np.asarray(z_parameters[:rc.N_TRAIN], np.float64)],
            axis=1,
        )
        ridge = 1e-8 * np.trace(design.T @ design) / design.shape[1]
        self.initializer_weights = np.linalg.solve(
            design.T @ design + ridge * np.eye(design.shape[1]),
            design.T @ self.z_train[:, 0],
        )
        self.mean_initial_latent = self.z_train[:, 0].mean(axis=0)

    def build(
        self,
        n,
        latent_extrapolation_scale=0.0,
        max_step_jacobians=None,
        shared_ops=None,
    ):
        """Build one deployable weak-rollout variant.

        ``latent_extrapolation_scale=0`` is the audited previous-latent LM
        start.  ``1`` uses ``2 z_n - z_{n-1}`` after the first step.  In both
        cases the LM solver evaluates the weak objective at that start and
        exits immediately when it is already below the absolute tolerance.
        """
        decoder = self.decoder
        max_step_jacobians = (
            rc.GN_BUDGET + 1
            if max_step_jacobians is None
            else int(max_step_jacobians)
        )
        if max_step_jacobians < 1:
            raise ValueError("max_step_jacobians must be positive")
        # lm_step_jit always evaluates the objective/Jacobian once at the
        # supplied start. Its budget counts subsequent trial attempts.
        rollout_attempt_budget = max_step_jacobians - 1
        coords = jnp.asarray(rc.grid_coords(n))
        decode_coords = jnp.asarray(rc.grid_coords(self.decode_resolution))
        # A same-mesh history gate shares the exact fitted EQ rule.  This is
        # offline work either way, but avoiding a second deterministic NNLS
        # fit keeps the gate bounded and makes equality structural.
        ops = shared_ops if shared_ops is not None else build_ops(
            decoder, n, self.z_snapshots
        )
        collocation = ops.get("colloc_used")
        if collocation is None or collocation.get("kind") != "grid":
            raise ValueError("FiLM control requires grid EQ for exact FOM upwind")
        fit_initial = fu.make_fit_ic_jit(
            decoder,
            n,
            rc.IC_BUDGET,
            coords=coords,
            idx=collocation["idx"],
            w=collocation.get("w"),
        )
        chunk = self.decode_chunk

        @jax.jit
        def decode_all(latents):
            padding = (-latents.shape[0]) % chunk
            padded = (
                jnp.concatenate((latents, jnp.zeros((padding, latents.shape[1]), F64)))
                if padding
                else latents
            )
            decoded = jax.lax.map(
                lambda latent_chunk: jax.vmap(
                    lambda latent: decoder(latent, decode_coords)
                )(latent_chunk),
                padded.reshape(-1, chunk, latents.shape[1]),
            )
            coarse = decoded.reshape(
                -1, self.decode_resolution, self.decode_resolution
            )[:latents.shape[0]]
            if n == self.decode_resolution:
                return coarse.reshape(latents.shape[0], n * n)
            fine = jax.image.resize(
                coarse,
                (latents.shape[0], n, n),
                method="linear",
                antialias=False,
            )
            return fine.reshape(latents.shape[0], n * n)

        initializer_weights = jnp.asarray(self.initializer_weights, F64)
        mean_initial = jnp.asarray(self.mean_initial_latent, F64)
        history_scale = jnp.asarray(float(latent_extrapolation_scale), F64)
        tolerance_scale = float(ops.get("tol_scale", np.sqrt((n - 2) ** 2)))
        grid_axis = jnp.linspace(0.0, 1.0, n)

        def rollout_history(z0, nu, tolerances):
            """Weak LSPG scan with a charged, zero-allocation history start."""
            def body(carry, tolerance):
                z_older, z_previous, previous_centers, step = carry
                extrapolated = z_previous + history_scale * (z_previous - z_older)
                z_start = jnp.where(step == 0, z_previous, extrapolated)
                z_new, rn, n_jac, accepted, reason, attempts = ops["step_jit"](
                    z_start, previous_centers, nu, tolerance, rollout_attempt_budget
                )
                next_carry = (
                    z_previous,
                    z_new,
                    ops["prev_of"](z_new),
                    step + jnp.int32(1),
                )
                return next_carry, (z_new, rn, n_jac, reason, attempts)

            initial = (z0, z0, ops["prev_of"](z0), jnp.int32(0))
            _, outputs = jax.lax.scan(body, initial, tolerances)
            return outputs

        def field_features(u0, nu):
            field = jnp.maximum(u0.reshape(n, n), 0.0)
            mass = jnp.sum(field) + 1e-300
            row_mass = jnp.sum(field, axis=1)
            col_mass = jnp.sum(field, axis=0)
            cx = jnp.sum(row_mass * grid_axis) / mass
            cy = jnp.sum(col_mass * grid_axis) / mass
            x_variance = jnp.sum(row_mass * (grid_axis - cx) ** 2) / mass
            y_variance = jnp.sum(col_mass * (grid_axis - cy) ** 2) / mass
            width = jnp.sqrt(jnp.maximum((x_variance + y_variance) / 2.0, 1e-12))
            amplitude = jnp.max(field)
            lognu = jnp.log(nu)
            return jnp.asarray([
                (cx - 0.5) / 0.35,
                (cy - 0.5) / 0.35,
                (width - 0.125) / 0.075,
                (amplitude - 1.25) / 0.75,
                (lognu - jnp.log(jnp.sqrt(0.001))) / (0.5 * jnp.log(10.0)),
            ], F64)

        @jax.jit
        def construct(u0, nu):
            features = field_features(u0, nu)
            predicted_initial = jnp.concatenate((jnp.ones((1,), F64), features)) @ initializer_weights
            starts = jnp.stack((mean_initial, predicted_initial))
            u0_rms = jnp.sqrt(jnp.mean(u0.reshape(n, n)[1:-1, 1:-1] ** 2))
            tolerance = jnp.full(
                (rc.NUM_STEPS,), rc.GN_TOL * u0_rms * tolerance_scale
            )
            z_initial, ic_relative, ic_jacobians, best_start, ic_attempts = fit_initial(
                u0, starts
            )
            (
                latents,
                reduced_norm,
                step_jacobians,
                reason,
                step_attempts,
            ) = rollout_history(z_initial, nu, tolerance)
            # Fixed-coarse neural decode plus charged, fused bilinear
            # prolongation.  The FOM receives a full target-grid guess.
            guesses = decode_all(latents)
            return (
                guesses,
                ic_relative,
                ic_jacobians,
                best_start,
                ic_attempts,
                reduced_norm,
                step_jacobians,
                reason,
                step_attempts,
                features,
            )

        return {
            "construct": construct,
            "ops": ops,
            "variant": VARIANT,
            "N": n,
            "decode_chunk": chunk,
            "decode_resolution": self.decode_resolution,
            "latent_extrapolation_scale": float(latent_extrapolation_scale),
            "max_step_jacobians": max_step_jacobians,
            "rollout_attempt_budget": rollout_attempt_budget,
            "eq_info": collocation.get("info"),
            "eq_indices": np.asarray(collocation["idx"]),
            "eq_weights": np.asarray(collocation["w"]),
        }
