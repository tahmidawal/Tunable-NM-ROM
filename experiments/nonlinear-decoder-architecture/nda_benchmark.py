"""Same-GPU decoder/Jacobian benchmark with persisted repetition arrays.

Usage:
  PDE=poisson python nda_benchmark.py control.pkl variant.pkl out.json
  PDE=burgers python nda_benchmark.py control.pkl variant.pkl out.json

For the old FiLM control, HIDDEN/N_LAYERS must match its checkpoint before
``ms_parametric`` is imported.  New residual decoders carry all architecture
dimensions in their checkpoint manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "deps", "multistage-precision"),
             os.path.abspath(os.path.join(HERE, "..", "..", "worktrees",
                                         "2026-08-14-multistage-precision",
                                         "experiments", "multistage-precision")),
             os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                         "2026-08-14-multistage-precision",
                                         "experiments", "multistage-precision"))):
    if os.path.isfile(os.path.join(cand, "ms_parametric.py")):
        sys.path.insert(0, cand)
        break
else:
    raise ImportError("ms_parametric.py not found")

import ms_parametric as mp  # noqa: E402
import nda_arch as nda       # noqa: E402

PDE = os.environ["PDE"]
CONTROL, VARIANT, OUT = sys.argv[1:4]
POINTS = [int(x) for x in os.environ.get(
    "POINTS", "256,1280,2560,4096,65536,262144").split(",")]
JAC_POINTS = [int(x) for x in os.environ.get(
    "JAC_POINTS", "256,1280,2560,4096").split(",")]
REPS = int(os.environ.get("TIME_REPS", "9"))
WARM = int(os.environ.get("TIME_WARM", "3"))
BURN_SECONDS = float(os.environ.get("BURN_SECONDS", "1.0"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bc_factor(xy):
    return 16.0 * xy[:, 0] * (1.0 - xy[:, 0]) * xy[:, 1] * (1.0 - xy[:, 1])


def load_model(path):
    raw = pickle.load(open(path, "rb"))
    if PDE == "poisson":
        stage = raw["stages"][0]
        params = jax.tree_util.tree_map(jnp.asarray, stage["params"])
        cfg = stage.get("decoder_config", dict(
            name="film", hidden=raw["config"]["hidden"],
            n_layers=raw["config"]["n_layers"], z_ff=stage.get("z_ff", 0)))
        n_freq, eps = int(stage["n_freq"]), float(stage["eps"])
        Z = np.asarray(raw["z_tr"])
        k = int(raw["config"]["K_LAT"])
        hard_bc = bool(raw["config"].get("hard_bc", 0))
    elif PDE == "burgers":
        params = jax.tree_util.tree_map(jnp.asarray, raw["params"])
        cfg = raw["config"].get("decoder_config", dict(
            name="film", hidden=raw["config"]["ad_hidden"],
            n_layers=raw["config"]["ad_layers"], z_ff=0))
        n_freq, eps = int(raw["n_freq"]), float(raw["eps"])
        Z = np.asarray(raw["Z_train"]).reshape(-1, int(raw["k_lat"]))
        k = int(raw["k_lat"])
        hard_bc = True
    else:
        raise ValueError(PDE)
    if cfg["name"] == "film":
        if int(cfg["hidden"]) != mp.HIDDEN or int(cfg["n_layers"]) != mp.N_LAYERS:
            raise ValueError(f"control env mismatch: checkpoint {cfg}, module "
                             f"hidden={mp.HIDDEN} layers={mp.N_LAYERS}")

    def apply(params_, z, xy):
        if cfg["name"] == "film":
            y = mp.film_apply(params_, z, xy, n_freq, cfg.get("z_ff", 0))
        else:
            y = nda.apply(params_, z, xy, n_freq, cfg, cfg.get("z_ff", 0))
        return eps * (bc_factor(xy) * y if hard_bc else y)

    prepared = None
    if cfg["name"] in ("resfilm", "groupfilm"):
        def prepare(params_, xy):
            return (nda.prepare_coords(params_, xy, n_freq, cfg),
                    bc_factor(xy) if hard_bc else jnp.ones((xy.shape[0],), jnp.float64))

        def apply_prepared(params_, z, packed):
            h, b = packed
            return eps * b * nda.apply_prepared(
                params_, z, h, cfg, cfg.get("z_ff", 0))
        prepared = (prepare, apply_prepared)

    meta = dict(path=os.path.basename(path), sha256=sha256(path),
                architecture=cfg, n_freq=n_freq, k=k,
                n_params=nda.parameter_count(params), hard_bc=hard_bc)
    return dict(params=params, z=jnp.asarray(Z.mean(0)), apply=apply,
                prepared=prepared, meta=meta)


def burn_gpu():
    x = jnp.ones((2048, 2048), dtype=jnp.float64)
    mm = jax.jit(lambda a: a @ a)
    mm(x).block_until_ready()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < BURN_SECONDS:
        mm(x).block_until_ready()


def timed(fn, args):
    # Burn first, then re-warm the exact kernel being measured.  Reversing this
    # order lets the large burn matmul evict/idle the decoder executable; the
    # first nominal repetition then becomes a systematic outlier (and CUDA's
    # timer reports sub-optimal accuracy) even though the median is stable.
    burn_gpu()
    for _ in range(WARM):
        fn(*args).block_until_ready()
    vals = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        vals.append(time.perf_counter() - t0)
    a = np.asarray(vals)
    med = float(np.median(a))
    return dict(all_s=[float(x) for x in a], median_s=med,
                mean_s=float(a.mean()), outliers_gt_1p5_median=int(np.sum(a > 1.5 * med)))


def points(n):
    rng = np.random.default_rng(20260819 + n)
    return jnp.asarray(rng.uniform(0.0, 1.0, size=(n, 2)), dtype=jnp.float64)


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit(f"GPU required, got {jax.default_backend()}")
    control, variant = load_model(CONTROL), load_model(VARIANT)
    report = dict(pde=PDE, backend=jax.default_backend(), device=str(jax.devices()[0]),
                  precision="f64/highest", reps=REPS, warm=WARM,
                  burn_seconds=BURN_SECONDS, points=POINTS,
                  jac_points=JAC_POINTS, models={"control": control["meta"],
                                                 "variant": variant["meta"]},
                  timings={"control": {}, "variant": {}}, checks={})

    for label, model in (("control", control), ("variant", variant)):
        fwd = jax.jit(model["apply"])
        jac = jax.jit(jax.jacfwd(model["apply"], argnums=1))
        for n in POINTS:
            xy = points(n)
            report["timings"][label][f"forward_p{n}"] = timed(
                fwd, (model["params"], model["z"], xy))
        for n in JAC_POINTS:
            xy = points(n)
            report["timings"][label][f"jacobian_p{n}"] = timed(
                jac, (model["params"], model["z"], xy))

    if variant["prepared"] is not None:
        prepare, apply_prepared = variant["prepared"]
        prep_fn = jax.jit(prepare)
        fwd_p = jax.jit(apply_prepared)
        jac_p = jax.jit(jax.jacfwd(apply_prepared, argnums=1))
        report["timings"]["variant_cached"] = {}
        eq_rel = []
        for n in sorted(set(POINTS + JAC_POINTS)):
            xy = points(n)
            packed = prep_fn(variant["params"], xy)
            jax.tree_util.tree_leaves(packed)[0].block_until_ready()
            raw = variant["apply"](variant["params"], variant["z"], xy)
            cached = apply_prepared(variant["params"], variant["z"], packed)
            eq_rel.append(float(jnp.max(jnp.abs(raw - cached))
                                / jnp.maximum(jnp.max(jnp.abs(raw)), 1e-300)))
            if n in POINTS:
                report["timings"]["variant_cached"][f"forward_p{n}"] = timed(
                    fwd_p, (variant["params"], variant["z"], packed))
            if n in JAC_POINTS:
                report["timings"]["variant_cached"][f"jacobian_p{n}"] = timed(
                    jac_p, (variant["params"], variant["z"], packed))
        report["checks"]["raw_vs_cached_max_relative"] = max(eq_rel)
        if report["checks"]["raw_vs_cached_max_relative"] > 1e-13:
            raise SystemExit("cached decoder is not numerically equivalent")

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
