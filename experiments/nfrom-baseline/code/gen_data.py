"""Snapshot pool for the neural-field ROM baselines: the ROM's own 4,608 Burgers 2D training trajectories
(P = params_draw(0,576) ++ params_draw(1000,4032)), 256 intervals, backward Euler dt = 0.005, first-order upwind,
solved with the test reference solver (fft_tight: Newton-BiCGStab, FFT Helmholtz preconditioner, ntol 1e-6, ltol 1e-8).

Unlike the operator pool (six output times), every second time level is kept: t = 0, 0.01, ..., 0.25 (26 levels),
because a projection ROM passes through all intermediate states.

  gen_data.py --out DIR [--stop N] [--check POOLCHUNK.fields.npy]

writes DIR/pool26.fields.npy float32 (N, 26, 257*257), DIR/pool26.meta.npz (params, n0), DIR/pool26.json.
The step kernel is burgers_fom.make_fom's, with every step returned instead of every tenth.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import burgers_fom as e

NTOL, LTOL, DT, L = 1e-6, 1e-8, .005, 256


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def pool_params():
    return np.concatenate((e.params_draw(0, 576), e.params_draw(1000, 4032)))


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
    a = argparse.ArgumentParser()
    a.add_argument('--out', required=True)
    a.add_argument('--stop', type=int, default=4608)
    a.add_argument('--check', default=None)
    a.add_argument('--allow-cpu', action='store_true')
    a = a.parse_args()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    assert a.allow_cpu or jax.default_backend() == 'gpu'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P = pool_params()
    phys = P[:a.stop]
    fn = make_fom_all(L, DT)
    store = np.lib.format.open_memmap(out / 'pool26.fields.npy', mode='w+', dtype=np.float32, shape=(len(phys), 26, (L + 1) ** 2))
    n0, resid, its, secs = [], [], [], []
    h = hashlib.sha256()
    for i, p in enumerate(phys):
        t0 = time.perf_counter()
        f, it, rn = fn(jnp.asarray(e.initial(L, p)), float(p[4]), NTOL, LTOL)
        f = np.asarray(f, dtype=np.float64); rn = np.asarray(rn); it = np.asarray(it)
        secs.append(time.perf_counter() - t0)
        assert np.isfinite(f).all() and rn.max() <= NTOL * (1 + 1e-9), ('not converged', i, float(rn.max()))
        for edge in (f[:, 0, :], f[:, -1, :], f[:, :, 0], f[:, :, -1]):
            assert not np.any(edge)
        row = f[::2].reshape(26, -1).astype(np.float32)
        store[i] = row
        h.update(row.tobytes())
        n0.append(float(np.linalg.norm(f[0]))); resid.append(float(rn.max())); its.append(int(it.max()))
        if i % 200 == 0:
            print('CASE', i, round(secs[-1], 3), flush=True)
    store.flush()
    info = dict(cases=len(phys), levels=26, times='0,0.01,...,0.25', mesh=L, dtype='float32',
                pool_sha256=sha_array(P), params_sha256=sha_array(phys), fields_sha256=h.hexdigest(),
                max_relative_residual=max(resid), max_newton_iterations=max(its),
                median_seconds=float(np.median(secs[1:] or secs)), total_seconds=float(sum(secs)),
                solver='fft_tight: Newton-BiCGStab, FFT preconditioner, dt 0.005, ntol 1e-6, ltol 1e-8',
                numpy=np.__version__, jax=jax.__version__, backend=jax.default_backend(),
                job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'))
    if a.check:   # cross-check against the operator lane's pool chunk (same parameters, same solver, six output times)
        ref = np.load(a.check, mmap_mode='r')
        n = min(64, len(phys), len(ref))
        mine = np.asarray(store[:n, ::5]).reshape(n, 6, L + 1, L + 1)
        info['check_max_abs_difference_vs_operator_pool'] = float(np.abs(mine - np.asarray(ref[:n])).max())
        info['check_cases'] = n
        assert info['check_max_abs_difference_vs_operator_pool'] < 1e-5, info
    np.savez(out / 'pool26.meta.npz', params=phys, n0=np.array(n0))
    (out / 'pool26.json').write_text(json.dumps(info, indent=1) + '\n')
    print('DONE', json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
