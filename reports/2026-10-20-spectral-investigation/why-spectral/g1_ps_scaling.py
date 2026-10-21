"""G1. Shape of the spectral FOM's per-step cost beyond the lanes' largest N (local GB10 GPU, one short run).

The H200 lanes timed the 2D spectral FOM up to N = 256 and the 3D one up to N = 64; the crossover estimate needs the
per-step cost at larger N. Absolute GB10 times are NOT comparable with the H200, so only RATIOS tau(N)/tau(N_ref) are
used (N_ref = 256 in 2D, 64 in 3D, both also timed on the H200). Same solver step as the lanes (E0 pscore, ETDRK4,
3/2 padding), one jitted block of `nsteps` steps, warm, median of 5.

Integrity (the GB10 has corrupted results before): for each (d, N) the final coefficients of a short deterministic run
are compared with the same run on CPU (separate process, JAX_PLATFORMS=cpu) to <= 1e-10 relative; the run is
discarded otherwise. Must be launched as:
  source /etc/profile.d/jax-mem.sh; JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun <venv python> g1_ps_scaling.py gpu
  JAX_PLATFORMS=cpu <venv python> g1_ps_scaling.py cpu
Writes analysis/g1_<mode>.json.
"""
import json
import sys
import time

import numpy as np

MODE = sys.argv[1]
import os
os.environ['JAX_PLATFORMS'] = 'cpu' if MODE == 'cpu' else 'cuda'   # before common.py's CPU default
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import common as C
sys.path.insert(0, str(C.REF))
import pscore as PS  # noqa: E402

CASES = {2: (48, 128, 256, 384, 512, 768, 1024), 3: (16, 32, 48, 64, 96, 128)}


def make(N, d, dt, nu, nsteps):
    Lk = jnp.asarray(PS.lin_eig(N, d, nu))
    co = PS.etdrk4_coefs(Lk, dt, 32)

    def step(i, v):
        Nv, _ = PS.nonlinear(v, N, d)
        a = co['E2'] * v + co['Q'] * Nv
        Na, _ = PS.nonlinear(a, N, d)
        b = co['E2'] * v + co['Q'] * Na
        Nb, _ = PS.nonlinear(b, N, d)
        c = co['E2'] * a + co['Q'] * (2 * Nb - Nv)
        Nc, _ = PS.nonlinear(c, N, d)
        return co['E'] * v + Nv * co['f1'] + 2 * (Na + Nb) * co['f2'] + Nc * co['f3']
    return jax.jit(lambda v: jax.lax.fori_loop(0, nsteps, step, v))


def ic(N, d):
    k = np.arange(1, N)
    g = 1.0 / k ** 3 * np.exp(-k / 8.0)              # smooth deterministic coefficients
    if d == 2:
        return np.outer(g, g)
    return np.einsum('i,j,k->ijk', g, g, g)


def main():
    res = dict(mode=MODE, backend=jax.default_backend(), devices=str(jax.devices()), rows={})
    print('jax_backend=' + jax.default_backend(), flush=True)
    if MODE == 'gpu' and jax.default_backend() != 'gpu':
        sys.exit(42)
    for d, Ns in CASES.items():
        for N in Ns:
            b0 = jnp.asarray(ic(N, d))
            chk = np.asarray(make(N, d, 0.002, 0.02, 3)(b0))            # integrity fingerprint
            row = dict(check=chk.ravel()[:: max(1, chk.size // 4096)].tolist(), check_norm=float(np.linalg.norm(chk)))
            if MODE == 'gpu':
                nst = 20
                f = make(N, d, 0.002, 0.02, nst)
                f(b0).block_until_ready()
                ts = []
                for _ in range(5):
                    t = time.perf_counter(); f(b0).block_until_ready(); ts.append(time.perf_counter() - t)
                row['tau'] = float(np.median(ts)) / nst
            res['rows'][f'{d}|{N}'] = row
            print(d, N, row.get('tau'), flush=True)
    (C.OUT / f'g1_{MODE}.json').write_text(json.dumps(res))


if __name__ == '__main__':
    main()
