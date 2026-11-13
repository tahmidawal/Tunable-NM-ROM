"""Generate operator data with the test reference solver (fft_tight) on the ROM's own parameter vectors.

  gen_data.py pool  --start A --stop B --out DIR      # cases P[A:B], P = params_draw(0,576) ++ params_draw(1000,4032)
  gen_data.py test  --mesh L --out DIR                # test64 = params_draw(20260916, 64) at L intervals
  gen_data.py extra --start A --stop B --out DIR      # cases X[A:B], X = params_draw(EXTRA_SEED, EXTRA_COUNT) (operator-scaling)

pool chunks: fields float32 (n,6,257,257) [t = 0, .05, ..., .25], params float64 (n,5), u0 norms float64.
test: fields float64 (64,6,L+1,L+1). Convergence (relative Newton residual <= 1e-6) and exact-zero boundary asserted.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import burgers_fom as e

TEST_SHA = 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'
NTOL, LTOL = 1e-6, 1e-8
EXTRA_SEED, EXTRA_COUNT = 20261112, 32256   # disjoint from 0, 1000 (pool), 20260916 (test), 7090702 (dev)


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def pool_params():
    return np.concatenate((e.params_draw(0, 576), e.params_draw(1000, 4032)))


def solve(L, phys, dtype):
    fn = e.make_fom(L, .005)
    out = np.empty((len(phys), 6, L + 1, L + 1), dtype=dtype)
    n0, resid, secs, its = [], [], [], []
    for i, p in enumerate(phys):
        t0 = time.perf_counter()
        f, it, rn = fn(jnp.asarray(e.initial(L, p)), float(p[4]), NTOL, LTOL)
        f = np.asarray(f, dtype=np.float64); rn = np.asarray(rn); it = np.asarray(it)
        secs.append(time.perf_counter() - t0)
        assert np.isfinite(f).all() and rn.max() <= NTOL * (1 + 1e-9), ('not converged', i, float(rn.max()))
        for edge in (f[:, 0, :], f[:, -1, :], f[:, :, 0], f[:, :, -1]):
            assert not np.any(edge)
        out[i] = f
        n0.append(float(np.linalg.norm(f[0]))); resid.append(float(rn.max())); its.append(int(it.max()))
        if i % 100 == 0:
            print('CASE', i, round(secs[-1], 3), flush=True)
    return out, np.array(n0), dict(max_relative_residual=max(resid), max_newton_iterations=max(its),
                                   median_seconds=float(np.median(secs[1:] or secs)), total_seconds=float(sum(secs)))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('what', choices=('pool', 'test', 'dev', 'extra'))
    a.add_argument('--start', type=int); a.add_argument('--stop', type=int)
    a.add_argument('--mesh', type=int, default=256)
    a.add_argument('--out', required=True)
    a = a.parse_args()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest' and jax.default_backend() == 'gpu'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.what == 'pool':
        P = pool_params()
        phys = P[a.start:a.stop]
        fields, n0, info = solve(256, phys, np.float32)
        name = f'pool-{a.start:05d}-{a.stop:05d}'
        info.update(pool_sha256=sha_array(P), chunk_params_sha256=sha_array(phys), start=a.start, stop=a.stop)
    elif a.what == 'extra':
        X = e.params_draw(EXTRA_SEED, EXTRA_COUNT)
        others = np.concatenate((pool_params(), e.params_draw(20260916, 64), e.params_draw(7090702, 4)))
        dmin = min(float(np.abs(others - x).sum(axis=1).min()) for x in X)
        assert dmin > 0, 'extra draw repeats a pool/test/dev parameter vector'
        phys = X[a.start:a.stop]
        fields, n0, info = solve(256, phys, np.float32)
        name = f'extra-{a.start:05d}-{a.stop:05d}'
        info.update(extra_seed=EXTRA_SEED, extra_sha256=sha_array(X), chunk_params_sha256=sha_array(phys), start=a.start,
                    stop=a.stop, min_l1_distance_to_pool_test_dev=dmin)
    elif a.what == 'dev':          # 4 non-test cases, only to smoke-test evaluate.py
        phys = e.params_draw(7090702, 4)
        fields, n0, info = solve(a.mesh, phys, np.float64)
        name = f'dev4-L{a.mesh}'
    else:
        phys = e.params_draw(20260916, 64)
        got = sha_array(phys)
        assert got == TEST_SHA, got
        fields, n0, info = solve(a.mesh, phys, np.float64)
        name = f'test64-L{a.mesh}'
        info.update(physical_sha256=got)
    np.save(out / f'{name}.fields.npy', fields)
    np.savez(out / f'{name}.meta.npz', params=phys, n0=n0)
    info.update(name=name, mesh=a.mesh if a.what not in ('pool', 'extra') else 256, cases=len(phys), dtype=str(fields.dtype),
                fields_sha256=sha_array(fields), solver='fft_tight: Newton-BiCGStab, FFT preconditioner, dt 0.005, ntol 1e-6, ltol 1e-8',
                numpy=np.__version__, jax=jax.__version__, job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'))
    (out / f'{name}.json').write_text(json.dumps(info, indent=1) + '\n')
    print('DONE', json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
