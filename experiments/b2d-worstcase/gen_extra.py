"""Extra training trajectories for the span fine-tune (DISCLOSED as beyond the baselines' 4,608-trajectory budget).

Parameters are drawn inside the training parameter box, concentrated where the validation diagnosis found the
worst projection errors (small viscosity, narrow bump, large amplitude):
  cx, cy ~ U(.15, .85)   w ~ U(w_lo, w_hi)   a ~ U(a_lo, a_hi)   nu ~ logU(nu_lo, nu_hi)
Solver, mesh, step, tolerances and stored levels are those of the baseline pool generator
(rebuttal-nfrom gen26/code/gen_data.py: fft_tight, 256 intervals, dt 0.005, levels t = 0, 0.01, ..., 0.25, float32);
the step kernel below is copied from it.

  gen_extra.py --seed S --count N --out DIR [--name extra] [--w 0.05 0.11] [--a 1 2] [--nu 0.01 0.03]
writes DIR/<name>.fields.npy (N, 26, 257*257) float32, DIR/<name>.meta.npz (params, n0), DIR/<name>.json.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import engines as e

NTOL, LTOL, DT, L = 1e-6, 1e-8, .005, 256
BOX = dict(w=(.05, .20), a=(.5, 2.), nu=(.01, .1))


def make_fom_all(L, dt, max_newton=20):
    def query(u0, nu, ntol, ltol):
        def pre(v):
            return e.helmholtz(v, nu, dt, L)

        def step(prev, _):
            scale = jnp.maximum(jnp.linalg.norm(prev), 1e-300)

            def body(state):
                u, it, rn = state
                r = e.residual(u, prev, nu, dt, L)

                def jv(v):
                    return jax.jvp(lambda q: e.residual(q, prev, nu, dt, L), (u,), (v,))[1]
                delta, _ = jax.scipy.sparse.linalg.bicgstab(jv, -r, tol=ltol, maxiter=200, M=pre)
                cand = u + delta
                return cand, it + 1, jnp.linalg.norm(e.residual(cand, prev, nu, dt, L))
            state = (prev, jnp.int32(0), jnp.linalg.norm(e.residual(prev, prev, nu, dt, L)))
            u, it, rn = jax.lax.while_loop(lambda s: (s[2] > ntol * scale) & (s[1] < max_newton) & jnp.isfinite(s[2]), body, state)
            return u, (u, it, rn / scale)
        _, (us, it, rn) = jax.lax.scan(step, u0[1:-1, 1:-1].reshape(-1), None, length=50)
        fields = jnp.pad(us.reshape(50, L - 1, L - 1), ((0, 0), (1, 1), (1, 1)))
        return jnp.concatenate((u0[None], fields)), it, rn
    return jax.jit(query)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True); p.add_argument('--count', type=int, required=True)
    p.add_argument('--out', required=True); p.add_argument('--name', default='extra')
    p.add_argument('--w', type=float, nargs=2, default=(.05, .11)); p.add_argument('--a', type=float, nargs=2, default=(1., 2.))
    p.add_argument('--nu', type=float, nargs=2, default=(.01, .03)); p.add_argument('--allow-cpu', action='store_true')
    a = p.parse_args()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    assert a.allow_cpu or jax.default_backend() == 'gpu'
    for k in ('w', 'a', 'nu'):
        lo, hi = getattr(a, k)
        assert BOX[k][0] <= lo < hi <= BOX[k][1], f'{k} range leaves the training box'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert not (out / f'{a.name}.fields.npy').exists(), 'already generated'
    r = np.random.default_rng(a.seed)
    phys = np.stack([r.uniform(.15, .85, a.count), r.uniform(.15, .85, a.count), r.uniform(*a.w, a.count),
                     r.uniform(*a.a, a.count), np.exp(r.uniform(np.log(a.nu[0]), np.log(a.nu[1]), a.count))], axis=1)
    seen = np.concatenate((e.params_draw(0, 576), e.params_draw(1000, 4032), e.params_draw(20260916, 64),
                           e.params_draw(20261114, 64)))
    assert not np.isclose(phys[:, None, :], seen[None, :, :], rtol=0, atol=1e-12).all(-1).any(), 'overlap with pool / test / fresh'
    fn = make_fom_all(L, DT)
    store = np.lib.format.open_memmap(out / f'{a.name}.fields.npy', mode='w+', dtype=np.float32,
                                      shape=(len(phys), 26, (L + 1) ** 2))
    n0, resid, its, secs = [], [], [], []
    h = hashlib.sha256()
    for i, q in enumerate(phys):
        t0 = time.perf_counter()
        f, it, rn = fn(jnp.asarray(e.initial(L, q)), float(q[4]), NTOL, LTOL)
        f = np.asarray(f, dtype=np.float64); rn = np.asarray(rn); it = np.asarray(it)
        secs.append(time.perf_counter() - t0)
        assert np.isfinite(f).all() and rn.max() <= NTOL * (1 + 1e-9), ('not converged', i, float(rn.max()))
        row = f[::2].reshape(26, -1).astype(np.float32)
        store[i] = row
        h.update(row.tobytes())
        n0.append(float(np.linalg.norm(f[0]))); resid.append(float(rn.max())); its.append(int(it.max()))
        if i % 200 == 0:
            print('CASE', i, round(secs[-1], 3), flush=True)
    store.flush()
    info = dict(cases=len(phys), seed=a.seed, ranges=dict(w=list(a.w), a=list(a.a), nu=list(a.nu)), levels=26,
                params_sha256=hashlib.sha256(np.ascontiguousarray(phys).tobytes()).hexdigest(), fields_sha256=h.hexdigest(),
                max_relative_residual=max(resid), max_newton_iterations=max(its),
                median_seconds=float(np.median(secs[1:] or secs)), total_seconds=float(sum(secs)),
                solver='fft_tight: Newton-BiCGStab, FFT preconditioner, dt 0.005, ntol 1e-6, ltol 1e-8',
                numpy=np.__version__, jax=jax.__version__, backend=jax.default_backend(),
                job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'))
    np.savez(out / f'{a.name}.meta.npz', params=phys, n0=np.array(n0))
    (out / f'{a.name}.json').write_text(json.dumps(info, indent=1) + '\n')
    print('DONE', json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
