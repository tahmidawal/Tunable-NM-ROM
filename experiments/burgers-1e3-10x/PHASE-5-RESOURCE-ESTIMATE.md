# Phase-5 P5-D resource and execution checkpoint

Status: prospective implementation estimate only; no P5-D scientific job has
been submitted.  The exact scientific cell remains capped by the Phase-5
preregistration and requires a separate root audit.

P5-D fits 35,904 locked S2 H1 coefficient targets in 16 case chunks: eight
N64 chunks of 64 cases, four N128 chunks of 32, and four N256 chunks of 16.
Each snapshot stores 3,328 f64 coefficients plus compact diagnostics.  Raw
coefficient payload is about 0.956 GB; all target chunks, metadata, main NPZ,
JSON, logs, and checksum files are budgeted below 1.2 GB.  The largest generated
truth chunk is N256 x 16 cases x 51 times, about 0.43 GB, and the eight independent
SciPy target fits are expected to remain comfortably within 96 GB host memory.

The measured Phase-4 P4-D job 2668794 completed 11,424 S2 fits (5,712 snapshots
for each of H1/H2) plus its paired cost panel in 26:33.  A linear fit-count
extrapolation is about 84 minutes for 35,904 fits; P5-D avoids the H2 spatial arm
but adds an independent sparse-design reconstruction, larger train chunks, and two
generator cost arms.  A conservative factor for the higher-N mix and I/O gives a
planning range of 3--6 hours.  The request is therefore one H200, 8 CPUs, 96 GB,
8 hours.  `OMP_NUM_THREADS`, OpenBLAS, and
MKL remain pinned to one; the eight CPUs serve the explicitly bounded eight-fit
Python thread pool, not hidden BLAS parallelism.  If the exact job reaches the
wall limit, that is an infrastructure-only incomplete attempt and may not produce
scientific promotion evidence.

The excluded local execution smoke used the mandated GB10 command, completed in
54.741 seconds, fit 2/2 synthetic targets with worst independent normal residual
4.58e-16, and executed both G1/G2 generator, non-collapse, basis/Pallas, K3/Cox,
and exact-boundary paths.  Its shared independent audit passed.  It intentionally
does not compile or time the full 51-state N1024 mandatory/max-one kernels; those
are cluster-only because compiling both locally would violate the one-minute
smoke rule.  Smoke artifacts are excluded and carry no scientific license.

Exact smoke invocation (the temporary output directory for the recorded run was
`/tmp/b10-p5d-smoke-r2-s8vRbE`):

```bash
source /etc/profile.d/jax-mem.sh
cd experiments/burgers-1e3-10x
B10_ORACLE_WORKERS=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
  timeout 59s jaxrun /home/tahmid/Dev/.venv/bin/python b10_phase5_d.py \
  --output-json /tmp/b10-p5d-smoke-r2-s8vRbE/phase5_d.json \
  --output-npz /tmp/b10-p5d-smoke-r2-s8vRbE/phase5_d.npz \
  --target-dir /tmp/b10-p5d-smoke-r2-s8vRbE/targets --smoke
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest timeout 59s \
  jaxrun /home/tahmid/Dev/.venv/bin/python b10_audit_phase5_d.py \
  /tmp/b10-p5d-smoke-r2-s8vRbE/phase5_d.json \
  /tmp/b10-p5d-smoke-r2-s8vRbE/phase5_d.npz \
  /tmp/b10-p5d-smoke-r2-s8vRbE/AUDIT.json --smoke
```

The JSON/NPZ/AUDIT SHA-256 values are respectively
`57d339dc5b5c62b017f7eeba5c508e949939f8e36148276342e1177fb619b425`,
`81fb4eb96b0dd92035ccdf867d0a3e19e73a9cbc7d82221529dad5643ede5f68`, and
`40d0dfbdd02f8a2166dc97b0dcedde68bda621138de08632eed4168233541a91`.

The synthetic scientific-shape audit fixture runs the exact 20-repetition
cyclic/reversed five-method schedule, 80 unique case/repetition rows per method,
native timing summaries/outliers, and all 51/401 coefficient-work records.  It
passes the positive fixture and rejects corrupted summary/order records while
classifying corrupted work as a negative gate:

```bash
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest \
  jaxrun /home/tahmid/Dev/.venv/bin/python \
  experiments/burgers-1e3-10x/b10_audit_phase5_d.py x x x \
  --science-shape-self-test
# phase5_d_science_shape_fixture=PASS
```
