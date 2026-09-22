# Speed log — ns3d-shift

Every row is a same-job, same-allocation, interleaved-repetition median of a
**complete query** (initial centering and projection, the solve, and every output
reconstruction). Nothing here is compared across jobs.

## The driver fix (job 4178112, `fast04`, A100, N=32, rank 64, dt=0.01)

`ns2d_rom.make_lm` -> `shift_rom.make_frozen_run`: fixed damped Gauss-Newton sweeps
with an **analytic** Jacobian in a statically unrolled scan, extrapolated warm start,
constant Jacobian terms and the Crank-Nicolson preconditioner hoisted out of the
sweep, `jnp.linalg.solve` replaced by a Cholesky factor/solve, and the two advection
contractions replaced by one against `T + T^T`.

| arm | median ms | saving | field parity vs the LM arm |
|---|---:|---:|---:|
| reference LM (`make_lm`, budget 60, gtol 1e-7) | 23.954 | 1.00x | — |
| frozen GN, 4 sweeps | 12.891 | 1.86x | 7.87e-09 |
| **frozen GN, 3 sweeps** | **9.494** | **2.52x** | **1.46e-08** |
| frozen GN, 2 sweeps | 6.857 | 3.49x | 5.03e-06 |

The pre-registered parity gate is 1e-8, so **3 sweeps is the headline setting**.
Two sweeps is accuracy-equivalent on the cohort (0.603 % against 0.602 % evolved
worst) and 1.4x cheaper again, but its parity sits outside the gate, so it is
reported as an operating point and never used for a headline claim.

Two bugs the parity gate caught before any GPU time, both of which produced a
plausible-looking 2e-5 instead of 7e-10: a scalar Levenberg damping over-damped the
coefficient directions, because the three shift columns are more than ten times the
norm of the coefficient columns; and the `T + T^T` rewrite of the Jacobian's
advection term was off by a factor of two.

## Irreducible grid-sized work (same job)

| piece | rank 64 | rank 128 |
|---|---:|---:|
| initial centering + projection of `u0` | 0.381 ms | 0.288 ms |
| one laboratory-frame output reconstruction | 0.197 ms | 0.196 ms |

At N=32 the six-output contract plus the initial projection is about 1.37 ms. The
comparator (CNAB2 dt=0.005) is 5.34 ms, so **even a free solve caps the speedup at
N=32 near 3.9x**: the 5x bar is out of reach at this mesh for arithmetic reasons,
not for want of a better solver. Both the FOM cost and this contract scale with the
mesh, so the ceiling is set by how many steps the FOM needs, which is what the
ladder measures.
