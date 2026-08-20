# Phase-6 P6-D resource and smoke checkpoint

Status: implementation/pre-submit evidence only. No Phase-6 scientific job has
been submitted. P6-D remains capped at one H200 diagnostic cell and requires a
separate root authorization.

P6-D regenerates four seed-20260822 N=1024 tight/tighter reference trajectories
and then compiles three G1 kernels: the descriptive R0 polynomial-weak route,
the actual R1 Cox-weak/K3-full route, and a paired identity kernel. It performs
24 balanced repetitions for FOM/R0/R1, or 288 total timed trajectory
invocations. It does not regenerate, load, or refit the 35,904 Phase-5 target
chunks; only the 3,328-value coefficient mean and two scales are read from the
39 KiB P5-D main NPZ.

P5-D's H200 G1 mandatory executable used 0.479 GB of eligibility device memory.
Its Cox identity executable used 0.812 GB. The paired Phase-6 identity returns
both 51-by-N^2 field arrays, so a conservative 4 GB device estimate remains far
below the fixed 20 GB gate. Four retained reference trajectories occupy about
1.71 GB f64 host memory; a 64 GB request leaves ample room for FOM work arrays,
compiled executables, and serialization. P3-D's solver/kernel/live-FOM panel
completed in 3:20 and P5-D's target-free online portion was a small fraction of
its 31:45 run. The conservative request is one H200, four CPUs, 64 GB host
memory, and two hours. OMP/OpenBLAS/MKL remain pinned to one; no host parallel
target fitting occurs.

The final excluded local GB10 smoke used GPU backend, f64, and highest matmul
precision. It completed the driver in 36.234 seconds and the shared independent
audit in 4.0 seconds. The charged route included the locked cold recovery and
raw feature construction on each invocation. At N=32 with one weak step, R1's
K3/Cox full-field identity was `4.266269662156725e-16`; current stencils,
previous centers, weak residual, and rho were bitwise identical. An independent
comparison of the actual timed R1 executable against the identity executable
was bitwise identical for full fields, residual, and rho. R1 canonical work and
memory gates passed, with 33,943,640 eligibility device bytes. The smoke
deliberately had no FOM or scientific promotion and is excluded from all
numerical/timing claims.

Recorded commands:

```bash
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest timeout 59s \
  jaxrun /home/tahmid/Dev/.venv/bin/python \
  experiments/burgers-1e3-10x/b10_phase6_d.py \
  --output-json /tmp/b10-p6d-smoke-6fIwwZ/phase6_d.json \
  --output-npz /tmp/b10-p6d-smoke-6fIwwZ/phase6_d.npz --smoke

source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest timeout 59s \
  jaxrun /home/tahmid/Dev/.venv/bin/python \
  experiments/burgers-1e3-10x/b10_audit_phase6_d.py \
  /tmp/b10-p6d-smoke-6fIwwZ/phase6_d.json \
  /tmp/b10-p6d-smoke-6fIwwZ/phase6_d.npz \
  /tmp/b10-p6d-smoke-6fIwwZ/AUDIT.json --smoke
```

The JSON/NPZ/AUDIT SHA-256 values are
`f837d8f9a567d99d1adc3f171c3679ef507162ac9e7c51021c063021540a9d0b`,
`1c31f85e8e9d173f479dd9c00262414354d2b2b36b8d1ecd66096b8f05eceac1`,
and `3303c17ebcbea87cfcdecbf58f0e65aea803abcd495374cfb2662c3fad491b79`.
