"""Solver check independent of any trained network: with an exact-at-nodes linear decoder (POD modes of one
trajectory, looked up at the nodes) the collocation Levenberg-Marquardt rollout must reproduce that trajectory to
its projection error. Run on 2026-11-11 (CPU): k=20 modes, M=512, 3 iterations -> rollout error 0.000 %;
k=8 -> 0.027-0.044 % against a projection error of 0.026 % (nu=0.011) and 0.003 % against 0.002 % (nu=0.1).
Also checked: stencil residual at the collocation nodes equals burgers_fom.residual there (max difference 0.0)."""
import numpy as np, jax, jax.numpy as jnp
import rom, burgers_fom as e, nf, gen_data
L = 256
P = e.params_draw(5, 2)
fom = gen_data.make_fom_all(L, .005)
for case in range(2):
    p = P[case]; u0 = e.initial(L, p)
    F = np.asarray(fom(jnp.asarray(u0), float(p[4]), 1e-6, 1e-8)[0])
    X = F.reshape(51, -1)
    Vt = np.linalg.svd(X, full_matrices=False)[2]
    for k in (8, 20):
        def dec(pp, xy, z):
            ij = jnp.rint(xy * L).astype(jnp.int32)
            return pp['Phi'][:, ij[:, 0], ij[:, 1]].T @ z
        nf.decoder = lambda cfg: dec
        nf.encode_c = lambda pp, coarse: pp['z0'][None]
        pp = dict(Phi=jnp.asarray(Vt[:k].reshape(k, L + 1, L + 1)), z0=jnp.asarray(Vt[:k] @ X[0]))
        proj = np.sqrt((((X @ Vt[:k].T) @ Vt[:k] - X)[10::10] ** 2).sum(1)).max() / np.linalg.norm(u0)
        for M, it in ((512, 1), (512, 3), (4096, 3)):
            lat, full = rom.build(dict(arm='C', k=k, width=1), M, jnp.float64, 'residual')
            f = np.asarray(jax.jit(full)(pp, jnp.asarray(u0), jnp.asarray(p[4]), jnp.asarray(p), jnp.int32(it), jnp.int32(0)))
            err = rom.metric(f[None], F[10::10][None], np.array([np.linalg.norm(u0)]))[0]
            print(f'case {case} nu {p[4]:.3f} k {k} M {M} iters {it}: rollout {100 * err:.3f} %  projection {100 * proj:.3f} %', flush=True)
