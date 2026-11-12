"""Network-free floor of CROM's released (explicit) time integration on our problem, validation only.

The reference is backward Euler, dt = 0.005 (burgers_fom). CROM's released online loop advances with a forward-Euler
target q + dt_sub * f(q) (online/online.py, proj_type 'nonlinear'). Here the decoder is replaced by the identity
(the exact full-order field), so the only error left is the integrator mismatch: forward Euler with the substep rule
n_sub = ceil(0.005 * (4 nu L^2 + 2 max|u0| L) / 0.9) (positivity/stability bound of sign-upwind + 5-point Laplacian),
same spatial operator burgers_fom.spatial. Metric as PROTOCOL.md.
  check_floor.py --data DIR --val-start 4352 --cases 32 --out FILE
"""
import argparse, json, os
import numpy as np
import jax, jax.numpy as jnp
import burgers_fom as e

L, DT = 256, .005


def nsub(nu, umax):
    return int(np.ceil(DT * (4 * nu * L * L + 2 * umax * L) / 0.9))


def fe_run(u0, nu, n):
    dt = DT / n

    def sub(u, _):
        adv, lap = e.spatial(u, L)
        return u - dt * (adv - nu * lap), None

    def big(u, _):
        u, _ = jax.lax.scan(sub, u, None, length=n * 10)
        return u, jnp.pad(u.reshape(L - 1, L - 1), 1)
    _, f = jax.lax.scan(big, u0[1:-1, 1:-1].reshape(-1), None, length=5)
    return f


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--data'); a.add_argument('--val-start', type=int, default=4352); a.add_argument('--cases', type=int, default=32)
    a.add_argument('--out')
    a = a.parse_args()
    meta = np.load(os.path.join(a.data, 'pool26.meta.npz'))
    mm = np.load(os.path.join(a.data, 'pool26.fields.npy'), mmap_mode='r')
    lo, n = a.val_start, a.cases
    sl = np.asarray(mm[lo:lo + n, ::5], np.float64).reshape(n, 6, L + 1, L + 1)
    params, n0 = meta['params'][lo:lo + n], meta['n0'][lo:lo + n]
    fn = jax.jit(fe_run, static_argnums=2)
    err, subs = [], []
    for i in range(n):
        k = nsub(params[i, 4], np.abs(sl[i, 0]).max())
        f = np.asarray(fn(jnp.asarray(sl[i, 0]), params[i, 4], k))
        err.append(float(np.sqrt(((f - sl[i, 1:]) ** 2).sum((-1, -2))).max() / n0[i])); subs.append(k)
        print('CASE', i, k, err[-1], flush=True)
    err = np.array(err)
    r = dict(what='forward-Euler (CROM released integrator) on the exact field vs backward-Euler reference, no network',
             cases=n, val_start=lo, worst_percent=100 * err.max(), median_percent=100 * float(np.median(err)),
             substeps_per_0p005=dict(min=min(subs), median=float(np.median(subs)), max=max(subs)), errors=err.tolist(),
             backend=jax.default_backend(), job_id=os.environ.get('SLURM_JOB_ID'))
    open(a.out, 'w').write(json.dumps(r, indent=1) + '\n')
    print('FLOOR', json.dumps({k: v for k, v in r.items() if k != 'errors'}))


if __name__ == '__main__':
    main()
