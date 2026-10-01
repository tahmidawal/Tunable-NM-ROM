"""Local feasibility probe (GB10, tiny, pre-design): can a curl-coordnet bank of rank R
approach the centred POD floor at all? Not a result; it only informs DESIGN.md.

Small data (n=32, few cases), short training. Prints floors on a handful of
development-seed cases (oracle centring) for: weighted POD of the same data at R,
and the coordnet bank at R after the mesh's Leray/2-3 projection.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in ("experiments/ns3d", "experiments/ns2d", "experiments/separable-decoder",
              "experiments/ns3d-grok", "experiments/ns3d-shift"):
    sys.path.insert(0, str(ROOT / extra))
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import ns3d_fom as F  # noqa: E402
import diag_floor as D  # noqa: E402
import shift_rom as SR  # noqa: E402
import coordnet as CN  # noqa: E402


def centred_frames(par, n):
    frames, _, _ = D.generate(par, n, 0.001, 0.2)
    cj = jax.jit(lambda f: SR.shift_field(f, -SR.grid_centroid(f) * n))
    for c in range(len(frames)):
        for t in range(frames.shape[1]):
            frames[c, t] = np.asarray(cj(jnp.asarray(frames[c, t])))
    return frames


def floor(Q, dev):
    """Relative to ||u0||, worst over cases of worst over t>0 (states already centred)."""
    flat = dev.reshape(dev.shape[0], dev.shape[1], -1)
    coef = flat @ Q
    miss = np.sum(flat ** 2, -1) - np.sum(coef ** 2, -1)
    rel = np.sqrt(np.maximum(miss, 0) / np.sum(flat[:, 0] ** 2, -1)[:, None])
    return float(rel[:, 1:].max()), float(np.median(rel[:, 1:].max(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cases", type=int, default=48)
    ap.add_argument("--dev", type=int, default=8)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--f32", action="store_true")
    a = ap.parse_args()
    n, R = a.n, a.rank
    t0 = time.time()
    tr = centred_frames(F.parameters(202609201, a.cases), n)
    dv = centred_frames(F.parameters(202609202, a.dev), n)
    print(f"data {tr.shape} {dv.shape} {time.time()-t0:.0f}s", flush=True)
    X = tr.reshape(-1, 3 * n ** 3)
    w = np.repeat(1.0 / np.sum(tr[:, 0].reshape(len(tr), -1) ** 2, 1), tr.shape[1])
    Xw = X * np.sqrt(w / w.mean())[:, None]
    gram = Xw @ Xw.T
    ev, V = np.linalg.eigh(gram)
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    Ufull = Xw.T @ V[:, :min(max(a.K, R), len(ev))] / np.sqrt(ev[:min(max(a.K, R), len(ev))])
    print("spectrum rel:", [f"{x:.1e}" for x in (ev / ev[0])[[i for i in (0, 15, 31, 63, 127, 199) if i < len(ev)]]])
    print("tail beyond K:", float(ev[a.K:].sum() / ev.sum()))
    a.K = min(a.K, Ufull.shape[1]); Y = (Ufull[:, :a.K] * np.sqrt(ev[:a.K])) * n ** -1.5
    Y = Y / np.linalg.norm(Y)
    for r in (16, 32, 64, 96, 128):
        if r <= Ufull.shape[1]:
            print(f"POD(same data, weighted) r={r} dev floor worst/median", floor(Ufull[:, :r], dv))
    dt = jnp.float32 if a.f32 else jnp.float64
    p = CN.init_params(jax.random.PRNGKey(0), R, a.width, a.depth, a.kmax, dtype=dt)
    sample = CN.make_sampler(n, chunk=n ** 3 // 4)
    # output scale: match ||G||_F^2 ~ R at init
    G0 = sample(p)
    s = float(jnp.sqrt(R / jnp.sum(G0 ** 2)))
    w_, b_ = p["layers"][-1]
    p["layers"][-1] = (w_ * s, b_)
    Yj = jnp.asarray(Y, dtype=dt)
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, 200, a.steps, a.lr * 1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    st = opt.init(p)

    def loss(p, Y):
        return CN.projection_loss(sample(p), Y, ridge=1e-8 if a.f32 else 1e-12)

    @jax.jit
    def upd(p, st, Y):
        v, g = jax.value_and_grad(loss)(p, Y)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, v
    t1 = time.time()
    for i in range(a.steps):
        p, st, v = upd(p, st, Yj)
        if i % 250 == 0 or i == a.steps - 1:
            print(f"step {i} loss {float(v):.3e} rms_rel {np.sqrt(max(float(v),0)):.3e} "
                  f"{time.time()-t1:.0f}s", flush=True)
    p64 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), p)
    G = np.asarray(CN.make_sampler(n, chunk=n ** 3 // 4)(p64))
    Gp, removed = CN.leray_mask(G, n)
    print("leray/mask removed fraction: max", float(removed.max()), "median", float(np.median(removed)))
    Q, _ = np.linalg.qr(Gp)
    print(f"coordnet R={R} dev floor worst/median", floor(Q, dv))
    # autodiff divergence check at a few points
    pts = jnp.asarray(np.random.default_rng(0).uniform(size=(64, 3)))
    div = CN.divergence(p64, pts)
    vel = CN.velocity(p64, pts)
    print("autodiff div / |g| :", float(jnp.max(jnp.abs(div)) / jnp.max(jnp.abs(vel))))


if __name__ == "__main__":
    main()
