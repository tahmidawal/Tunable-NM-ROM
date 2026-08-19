# Burgers-2D FOM-exact hybrid optimisation through N=1024

Status: active experiment. Numbers in this cell are provisional until the final
single-GPU consolidation and independent verification pass.

This cell redesigns the Burgers warm-start path after the audited FiLM NM-ROM
rollout lost every wall-clock comparison through `N=256`. The delivered field is
always finished by the same full-order backward-Euler/Newton/BiCGStab solver to a
named tolerance. The fixed problem is the original testbed: `dt=0.005`, 50 steps,
f64, the exact upwind/centred finite-difference operator.

## Pre-registered gates

The audited `N=256`, `tau_FOM=1e-6` row spent 479.569 ms constructing the ROM
guess and saved only 15.149 ms in the FOM stage. Therefore:

1. A replacement must cost less than 15 ms per 50-step rollout at `N=256` if it
   preserves only the old FOM-stage saving.
2. It must beat **linear extrapolation**, not merely the previous-state start, in
   total time and BiCGStab work.
3. Cost and accuracy come from the same invocation; timing repetitions are saved.
4. Cross-resolution wall clock comes from one sequential job on one GPU only.
5. Every published arm must finish every step to the named Newton tolerance with
   finite arithmetic. A failed cheap solve is not a speedup.

## Sequence

1. `bh_oracle.py`: controlled warm-start quality curves relative to both free
   baselines. It interpolates between linear extrapolation and the exact next
   FOM state to measure how much guess error must be removed, and how much FOM
   time that removal can buy. Oracle rows are explicitly non-deployable.
2. Cheap deployable predictors: a charged history-only **classical control**,
   followed by a learned cached low-rank correction of linear extrapolation.
   The history-line arm is not called an NM-ROM. Candidate selection is based
   on total time and BiCGStab work, not field L2 alone.
3. Audited consolidation through `N={32,64,128,256,512,1024}` at
   `tau_FOM={1e-6,1e-8,1e-10}`.

## Provenance and cluster layout

All jobs live under `/cluster/tufts/paralab/tawal01/hybb1024/`, one directory per
job, on the `gpu` partition. Batch scripts assert `jax_backend=gpu`, set
`JAX_DEFAULT_MATMUL_PRECISION=highest`, and regenerate trajectories from the
recorded seed. Pulled runs retain logs and remote/local checksums.
