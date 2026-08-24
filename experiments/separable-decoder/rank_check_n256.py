import os, sys, numpy as np
HERE = "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-23-n256-push/experiments/separable-decoder"
sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
import sep_common as sc

def grid_interior_coords(N):
    x = np.linspace(0.0, 1.0, N); X, Y = np.meshgrid(x, x, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    return xy[np.arange(N*N).reshape(N, N)[1:-1, 1:-1].ravel()]

def spec(G, tag):
    Gj = jnp.asarray(G, dtype=jnp.float64)
    ev = np.linalg.eigvalsh(np.asarray(Gj.T @ Gj, dtype=np.float64))[::-1]
    sv = np.sqrt(np.maximum(ev, 0.0)); rel = sv / sv[0]
    ix = [i for i in (128, 255, 256, 257, 300, 383, 511, len(rel)-1) if i < len(rel)]
    print(f"  {tag}  R={G.shape[1]}")
    print("    " + "  ".join(f"sv[{i}]/sv0={rel[i]:.2e}" for i in ix), flush=True)

cj = jnp.asarray(grid_interior_coords(256), dtype=jnp.float64)
for p in ["runs/push_r2_burgers/out/sep_burgers_r2_N256_K16_R512.pkl",
          "runs/push_r2b_poisson/out/sep_poisson_r2b_N256_K16_R512.pkl"]:
    params, Z, cfg = sc.load_pkl(os.path.join(HERE, p))
    spec(sc.features(params, cj), f"TRAINED {os.path.basename(p)}")
for gh, R in [(256, 512), (512, 512), (1024, 512)]:
    p = sc.init_separable(jax.random.PRNGKey(0), 16, R, n_ff=128, ff_scale=4.0,
                          g_hidden=gh, h_hidden=256, out_scale=1.0)
    spec(sc.features(p, cj), f"INIT g_hidden={gh}")
