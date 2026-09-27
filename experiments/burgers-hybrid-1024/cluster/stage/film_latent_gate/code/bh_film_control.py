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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_ops(decoder, n, z_snapshots):
    """Audited weak64/EQ256 path, with per-N decoder-output NNLS refit."""
    solver, collocation_name, objective = VARIANT.split(":")
    kind = "weak"
    mode_count = int(objective[len(kind):])
    point_count = int(collocation_name[2:])
    collocation = rc.fit_eq_weights(
        decoder,
        n,
        mode_count,
        point_count,
        z_snapshots,
        kind=kind,
        pool="grid",
        rng=np.random.default_rng(EQ_RNG_SEED),
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

    def build(self, n, latent_extrapolation_scale=0.0):
        """Build one deployable weak-rollout variant.

        ``latent_extrapolation_scale=0`` is the audited previous-latent LM
        start.  ``1`` uses ``2 z_n - z_{n-1}`` after the first step.  In both
        cases the LM solver evaluates the weak objective at that start and
        exits immediately when it is already below the absolute tolerance.
        """
        decoder = self.decoder
        coords = jnp.asarray(rc.grid_coords(n))
        decode_coords = jnp.asarray(rc.grid_coords(self.decode_resolution))
        ops = build_ops(decoder, n, self.z_snapshots)
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
                    z_start, previous_centers, nu, tolerance, rc.GN_BUDGET
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
            "eq_info": collocation.get("info"),
            "eq_indices": np.asarray(collocation["idx"]),
            "eq_weights": np.asarray(collocation["w"]),
        }
