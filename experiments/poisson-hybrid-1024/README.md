# Poisson NM-ROM → FOM optimization through N=1024

This experiment optimizes the complete warm-start path while keeping the delivered field
FOM-exact to a named true-residual tolerance. Results are provisional until the final
single-GPU N-ladder and fresh-seed confirmation runs are complete.

## Pre-registered first feasibility round

The audited K=8 FiLM checkpoint spends 2.73 ms in the latent solve and 4.56 ms decoding at
N=512, while its best warm start saves only 3.01 ms of CG at `fom_tau=1e-6`. The first round
therefore changes the online path without retraining the decoder:

1. Predict the trained latent directly from the four known source parameters with a Gaussian
   RBF map. Its length scale and ridge are selected using only a fixed split of the 512
   training samples; held-out PDE cases never select them.
2. Evaluate the mesh-free decoder on a fixed 64x64 grid and linearly prolongate to the FOM
   mesh. This keeps the hard boundary exactly zero and deliberately filters grid-scale error,
   which is disproportionately expensive in CG's A-norm.
3. Test an 8x8 exact sine-mode residual correction after the learned guess. This is a classical
   correction and is reported separately, alongside 8x8 and 16x16 spectral-only controls.

Initial arms are fixed before held-out timing:

- `rbf_c64_q0`: learned latent + coarse decode/prolongation;
- `nearest_c64_q0`: charged train-nearest latent lookup + the same decode, so any RBF benefit
  is attributable to prediction rather than merely bypassing LM;
- `rbf_c64_q8`: the same neural guess plus an 8x8 spectral residual correction;
- `spectral_q8` and `spectral_q16`: classical controls with no learned component.

The direct-latent screen was falsified by the local smoke: the auto-decoder chart is not a
smooth function of the four source parameters. RBF and nearest-latent predictions are therefore
retained as negative controls, not tuned on PDE validation cases. The next gate keeps the
audited weak-form LM solve and changes only its initialization/globalization and output path:
mean versus charged nearest initialization, base versus training-cloud trust region, full-grid
versus fixed-64-grid decode/prolongation, and the separately labeled 8x8 spectral correction.

Every arm is handed to the same counting CG kernel. The reported total is measured as one
callable containing guess construction and CG finishing, paired back-to-back with zero-start
CG on the same right-hand side after GPU burn-in. All timing repetitions are persisted. The
exact sine-diagonalized direct solver remains visible as the strongest baseline.

The final N-ladder will also retain the original audited weak-form LM NM-ROM as a labeled
baseline. Direct latent prediction is a learned-manifold path, but it is not the same online
algorithm and will not silently replace it in comparisons.

Before retraining, the harness also evaluates the tracked parameter-aligned FiLM control from
the multistage experiment. Its stage-1 and combined-stage maps take the known physical source
parameters directly to a field, then use the identical coarse decode/prolongation and FOM
finisher. These are labeled `direct_surrogate`, not NM-ROM; they test whether aligning the chart
is sufficient to make a useful warm start.

## Pre-registered cached nonlinear-decoder gate

The architecture gate reuses the frozen K=16 `groupfilm` checkpoint selected by the independent
nonlinear-decoder study. It has no POD/output basis: latent FiLM modulation is followed by
nonlinear activations. The coordinate-only affine stem is cached offline at EQ and coarse-grid
decode points, so this optimization does not change the learned function. Its weak objective uses
the independently selected M=128/m=512 configuration and a charged nearest-training-latent lookup.

On the model's original 64-case validation stream (`TEST_SEED=0`, cases after the 512 training
cases), the fixed calibration candidates are objective-reduction stops 0.30, 0.10, 0.03, and
0.01. Learned-only is reported for each; combined q8 is reported for 0.30, 0.10, and 0.03, next to
spectral-only q8/q16. The gate passes only if an arm reaches at most three median Jacobian
evaluations and its combined q8 total beats spectral q8 within the same rotated timing block.
Among passing candidates, validation total cost selects one stop before any fresh-seed ladder.
If none passes, GroupFiLM remains an architecture control and is not expanded into a costly
confirmation fleet.

The final fresh-seed ladder will use `TEST_SEED=20260819` and retain the selected GroupFiLM arm,
the original audited K=8 LM baseline, the parameter-aligned surrogate controls, spectral q8/q16,
zero-start counting/native CG, and the FFT-DST exact direct solver. Cost, work, residual, and error
are all taken from each timed invocation in the same rotated block.

The spectral rank search is also fixed before fresh-seed inspection. The first validation ladder
tests q=8/16/24/32/48/64 at every N. If q64 is the total-time endpoint winner at N=512 and N=1024,
a same-job extension repeats q32/q64 and adds q96/q128. Only if q128 is again the endpoint winner
does a second same-job extension repeat q64/q128 and add q192/q256. The minimum from these
validation-only ladders (or an explicit N-dependent rule if the minima differ by N) is then locked
for the independent seed. Every extension retains the FFT-DST direct solve; no timing is compared
across jobs without a repeated within-job control.

The q192/q256 extension left q256 as the total-time endpoint winner at N=1024, so it did not yet
bracket the spectral minimum. Before looking at any fresh-seed case, one final N=1024 job repeats
q192/q256 and adds q384/q512/q768 and full rank (the requested q1024 arm is clamped to the 1022
interior modes). The lowest authoritative mean-of-case-medians in that single rotated job fixes
the N=1024 rank. Because the bracket includes the dense exact endpoint and the FFT-DST direct
baseline, there is no further rank extension or held-out timing selection after this job.

## Files

- `feasibility.py`: train-only RBF calibration, coarse decoding, spectral corrections,
  same-kernel CG finishing, diagnostics, and timing.
- `cluster/`: isolated job construction, launch, pull, and explicit-ID cancellation helpers
  (added after the local smoke gate).
- `runs/`: checksummed pulled artifacts only.

## Status

Local jaxrun smoke passes interpolation, hard-boundary, GroupFiLM cached decode, solver, residual,
and persistence gates.
It falsified source-parameter-to-latent prediction for the frozen checkpoint; no local wall-clock
number is a cluster result. Full feasibility results remain pending.
