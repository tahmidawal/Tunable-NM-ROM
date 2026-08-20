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

The full-rank bracket located the N=1024 `1e-6` turnover at q512. A final validation sensitivity,
still on `TEST_SEED=0`, repeats q256/q384/q512/full rank at N=512 and N=1024 for all three FOM
tolerances. It fixes the `(N, tolerance)` rank rule before fresh-seed confirmation; lower meshes
use the selected rank clamped to their interior dimension. This sensitivity also times the full
dense sine diagonalization as a separately labeled direct baseline, including its measured f64
true residual, rather than conflating it with a partial-q-plus-CG hybrid.

## Locked fresh-seed confirmation

Validation fixes the spectral rule to `q=min(512,N-2)` at FOM tolerance `1e-6` and full rank at
`1e-8` and `1e-10`. Both `spectral_q512` and `spectral_q1024` remain in the confirmation JSON so
the matched same-job sensitivity is auditable, but `TEST_SEED=20260819` does not re-select the
rule. The dense full-rank direct solve and FFT-DST direct solve are separately labeled; a direct
field is only eligible at a tolerance when its timed invocation's measured true residual meets it.

The final job uses N=32,64,128,256,512,1024; all three tolerances; 16 accuracy cases and six
timed cases with seven repetitions each. Its fixed genuine NM-ROM controls are the original
audited K=8 full-decode weak LM (`lmmean_cfull_q0`), its optimized trust-region/coarse-decode
two-stage variant (`lmtrmean_c64_q0`), and cached nonlinear K=16 GroupFiLM both learned-only
(`groupn_rt30_c64_q0`) and with a separately labeled q8 classical third stage
(`groupn_rt30_c64_q8`). The lower-priority parameter-aligned surrogate is excluded because it is
not an NM-ROM online solve. These controls are timed alongside the two spectral arms, zero-start
counting/native CG, native Jacobi sensitivity, dense sine direct, and FFT-DST direct in one
rotated post-burn block. No fresh-seed timing selects an architecture, rank, tolerance, or
reported comparison.

## Pre-registered balanced timing audit

The 15-arm rotation in the first fresh-seed job is not authoritative for the small claimed K8
crossover: six cases times seven repetitions visit only 12 of the 15 cyclic offsets, and the raw
N=1024 timings show a material first/second-position effect. The learned crossover is therefore
withdrawn pending a narrow balanced confirmation. Architecture, weak objective, trust radius,
coarse grid, FOM kernel, and tolerances stay locked; there is no further model or hyperparameter
selection.

The audit compares only `lmtrmean_c64_q0` with the same unpreconditioned true-residual counting
CG from zero at N=512 and N=1024 and tolerances 1e-6, 1e-8, and 1e-10. It uses fresh
`TEST_SEED=20260820`, 16 accuracy cases, eight timed cases, and 12 timed repetitions per case.
Every adjacent repetition pair is exactly: burn, learned then zero; reburn, zero then learned.
Thus each method occupies first and second position six times per case. The timed return is graded
for field error, true residual, status, and iterations from that same invocation. Headline times
are medians across per-case medians; uncertainty is a paired case-clustered bootstrap over source
cases. The JSON also retains raw samples, order-specific medians, paired case/repetition signs,
per-case outliers under the fixed 1.5x rule (none removed), and per-case source parameters.

The earlier exact-direct and native-CG measurements remain separately labeled controls rather
than entering the AB/BA pair. Counting CG is authoritative because the native implementation was
slower at every tolerance and missed true residual gates at the tight high-resolution rows.

## 20 August speed push: one-update parameter-aligned nonlinear chart

The audited K8 path leaves a real counting-CG construction budget at N=1024, but not enough to
justify another multi-Jacobian FiLM sweep.  The next validation-only gate therefore reuses the
already tracked stage-1 parameter-aligned coordinate decoder as a nonlinear manifold `u(x;z)`.
The known normalized source parameters provide a deployable initial latent `z0`; exactly one
weak Gauss--Newton update is then taken before a fixed-N=64 decode and the unchanged counting-CG
finish.  This is reported as a parameter-aligned NM-ROM, while `param1_c64_q0` with no latent
update remains separately labelled a direct surrogate.

The online objective is the discrete weak form on the decoder's trained N=64 chart.  Discrete
summation by parts expresses each residual mode using smooth decoder outputs rather than
pointwise Laplacians.  NNLS weights are fitted on decoder-output snapshots, `M` is comfortably
above `k=4`, and `m=4M`.  The Gaussian source projection is separable and uses only the known
parameters, so the cold start never scans the target FOM grid.  The one-update kernel forms one
decoder Jacobian, clips the step to radius 0.35, evaluates four fixed line-search lengths in one
fused batch, and rejects the update unless its weak objective improves.

Validation seed 0 screens `M/m = 16/64, 24/96, 32/128` at N=64 and N=256.  This means the
canonical held-out slice 512:528: the harness first draws the checkpoint's 512 training cases
and then takes the next 16, so no training case enters selection.  The panel retains the
no-update parameter surrogate, zero-start counting CG, and the eligible dense/FFT/spectral
controls.  A learned candidate advances only if its same-job
construction is below 0.6 ms and it strictly improves mean held-out A-error, counting-CG work,
and end-to-end total time relative to the no-update parameter field.  Update acceptance is
reported descriptively and does not govern this already-running gate; any future acceptance gate
must fix its numerical threshold before launch.  Wall clock chooses only among
candidates that pass those mechanism gates.  Any survivor is then frozen and measured on an untouched seed with
balanced AB/BA timing through N=1024; dense DST is the production comparator at N=1024 tolerances
1e-6/1e-8, and full spectral-plus-FOM or FFT-DST is the comparator at 1e-10.  If no candidate
passes, the nonlinear latent-update route stops without a fresh-seed timing claim.

The alpha=1 field-error gate is now stopped: all three updates lowered their own weak objective
but worsened A-error, counting-CG work, and total time relative to the direct parameter field at
both meshes.  This is an objective-alignment failure, not a failed line search, so no alpha=1
candidate advances and those cases are not reused for another architecture or hyperparameter
selection.

One final, distinct objective-design gate is frozen before launch.  It changes only the weak
weight to alpha=0.5, the energy/A-norm weighting already defined in `pro_common.py`, and tests
exactly `paramritz1_m24_c64_q0`: K=4, M=24, m=96, fixed N=64 decode, trust radius 0.35, and one
GN Jacobian/update.  Alpha and M are not swept.  Development uses the wholly new seed 20260821,
its first 16 cases, N=64 and N=256, and tolerance 1e-6; neither the seed-0 cohort nor the locked
confirmation seed is inspected.  The same-job panel is fixed to the no-update
`param1_c64_q0`, `spectral_q256`, zero-start counting CG, dense DST, and FFT-DST controls.

This last candidate advances only if, separately at both N=64 and N=256, construction is below
0.6 ms and all three means are strictly below `param1_c64_q0`: initial-guess A-error,
counting-CG iterations, and same-invocation end-to-end total time.  Update acceptance is only a
diagnostic.  Failure of any gate at either mesh exhausts this Poisson nonlinear-warm-start search
without an N=1024 run.  Only a full pass permits one selection-independent confirmation on the
already frozen seed 20260822 through N=1024, retaining the production dense-DST control at
tolerances 1e-6/1e-8 and spectral/full-rank or FFT-DST at 1e-10.

## Files

- `feasibility.py`: train-only RBF calibration, coarse decoding, spectral corrections,
  same-kernel CG finishing, diagnostics, and timing.
- `cluster/`: isolated job construction, launch, pull, and explicit-ID cancellation helpers
  (added after the local smoke gate).
- `runs/`: checksummed pulled artifacts only.
- `SUMMARY.generated.md`: exhaustive tables generated from the audited run JSONs.
- `FINAL.generated.md`: the same generated tables restricted to the fresh-seed confirmation.
- `summarize_speed_push.py` and `SPEED-PUSH.generated.md`: generated closing audit for the final
  one-update mechanism gates, balanced K8 result, and production direct controls.

## Status

The fresh-seed six-resolution, three-tolerance mechanism ladder is complete and all primary
counting-CG arms meet their true-residual tolerance from their timed invocation. Its small K8
crossover claim from the multi-arm rotation is retracted because the timing order was not fully
balanced. The dedicated AB/BA audit (`pairfinal1`, cluster job 2664551) is authoritative: N=512
is tied at the two looser tolerances and loses at the tightest; N=1024 has a smaller supported win
only at 1e-6, mixed/inconclusive evidence at 1e-8, and no win at 1e-10. Exact values, paired
case/repetition signs, order-specific medians, outliers, and case-clustered intervals are generated
from the JSON in `FINAL.generated.md`.

The balanced job used the untouched selection-independent seed 20260820 and completed on an
A100 80GB PCIe with GPU preflight, f64/x64, and highest matmul precision. Every one of its six
rows passed the same-invocation true-residual and status gates; every case had six first-position
and six second-position samples per method, and no sample triggered the fixed within-case outlier
rule. Local and remote pull checksums matched, the logs contain no forbidden warning signature,
and the explicit remote job directory was deleted after verification.

Native JAX-CG sensitivity in `final1` loses for the learned warm start at both audited high
resolutions and all tolerances, and its N=512/N=1024 tight-tolerance fields fail the recomputed
true-residual gate. The true-residual counting CG is therefore the authoritative iterative
contract; native results remain visible as non-eligible sensitivity controls. Pure GroupFiLM
does not improve work. GroupFiLM plus q8 beats zero-start CG but loses decisively to the matched
spectral controls, so its apparent gain is classical rather than learned. Dense sine direct is
the fastest eligible method on most rows; at the tight N=1024 row its measured residual is not
eligible, while the full-rank warm start plus FOM refinement and FFT-DST direct remain eligible.

The source-parameter-to-latent map and transported-tail training round were stopped by their
pre-registered mechanism gates rather than tuned on held-out timings. Local wall clock remains
non-result smoke evidence only; all numeric conclusions are generated in `SUMMARY.generated.md`.
The alpha=1 one-update parameter-aligned gate also failed all solver-relevant mechanism gates.
The pre-registered alpha=0.5 energy-objective gate then failed the same A-error, CG-work, and
same-job-total tests at both development meshes despite accepting every update and lowering its
own truncated objective.  It therefore did not advance to the frozen N=1024 confirmation, and
no alpha or M tuning followed.  `SPEED-PUSH.generated.md` is the closing, JSON-derived audit.
The Poisson nonlinear-warm-start speed search is exhausted under the construction budgets and
strongest eligible direct/spectral controls recorded here.

## 20 August N=2048 extension: locked confirmation design

This extension is a resolution confirmation, not a renewed architecture or hyperparameter
search.  It branches exactly from audited Poisson closure commit `57329c0` and keeps the frozen
genuine K=8 NM-ROM `lmtrmean_c64_q0`: M=64, m=256 decoder-output EQ, alpha=1 smooth weak form,
training-cloud trust region, mean training latent initialization, fixed N=64 decode/prolongation,
and the same unpreconditioned true-residual counting-CG finish.  There is no architecture,
objective, latent-dimension, coarse-grid, quadrature, or stopping-policy selection at N=2048.

The sole scientific cohort is fixed before output to `TEST_SEED=20260826`, first 16 generated
cases, with the first eight timed.  The mesh is only N=2048 and FOM tolerances are exactly
1e-6, 1e-8, and 1e-10.  Both smoke and final are constrained to the same NVIDIA A100-80GB GPU
class; all work is f64/highest.  Every timed invocation returns its
own field, true residual, error, status, boundary contract, and iteration work; no cost or
accuracy is joined across calls or jobs.  The hard zero boundary is checked on the learned guess
and on every timed final field.  All timing samples are retained, and the fixed outlier diagnostic
is a repetition above 1.5 times its own case median; no observation is removed.

The final uses three untimed warmups.  The learned comparison then uses 12 repetitions per timed
case and tolerance.  Every adjacent pair is
fresh burn, NM-ROM then zero CG; fresh burn, zero CG then NM-ROM.  Thus each method is first and
second six times per case.  The headline is the median across eight per-case medians, with 10,000
whole-case bootstrap resamples for time, paired delta, and speed ratio.  A learned crossover is
supported only when the clustered speedup interval is above one, the learned-minus-zero delta
interval is below zero, all statuses are zero, and every recomputed true residual and exact-zero
boundary meets its named gate.

Production controls are measured in a second balanced block in the same job: zero counting CG,
dense DST, FFT-DST, partial `spectral_q1024` plus counting CG, and full
`spectral_q2048` plus counting CG (the full arm clamps to all 2046 interior modes).  Ten
repetitions use all five cyclic rotations followed by their reverses, with a fresh burn before
every order.  Each method occupies every clock position exactly twice per case and every method
pair has 5/5 precedence.  The same median-across-cases and whole-case bootstrap contract applies.
Direct methods are eligible only where their timed maximum recomputed residual meets the named
tolerance; an ineligible direct row remains visible rather than being silently promoted.  The
partial/full ranks are fixed scaling controls, not a rank sweep or selection bracket.

Before the scientific cohort is opened, an excluded N=2048 execution/memory smoke uses unrelated
seed 13579, one case, no clock claim, and the same learned/full-control compiled routes.  The
smoke uses two learned repetitions, one untimed warmup, ten balanced control repetitions, and no
burn because its only purposes are execution and peak-memory eligibility.  The final
job is licensed only if that smoke completes without OOM, captured-large-constant, nonfinite,
boundary, solver-status, or required-residual failure and reports peak device allocation at most
80% of the device limit.  Smoke timings and field errors are non-results.  The scientific run is
single-shot regardless of outcome; no seed, rank, or architecture follow-up is licensed here.

The excluded smoke completed on the required A100-80GB class and passed the frozen execution,
memory, residual, boundary, provenance, and checksum audit.  Its pulled raw artifact and generated
`AUDIT.json` are retained under `runs/n2048smoke1/`; no smoke timing or error is used as a
scientific result.

## 20 August N=2048 extension: final outcome

The single locked scientific job `2670159` completed on the required A100-80GB class from clean
commit `a98b1e8`.  Its stderr is empty; the log records GPU backend and `ALL-DONE`; f64/highest,
memory, residual, boundary, timing-balance, raw-record, source-provenance, and pull-checksum gates
all pass.  The staged `feasibility.py` hash is bound byte-for-byte to the scientific commit.

The genuine K=8 NM-ROM warm start has one supported N=2048 crossover, at FOM tolerance 1e-6.
The 1e-8 row is inconclusive and the 1e-10 row is a tie/slight loss.  This sharpens rather than
broadens the earlier conclusion: the learned method has a repeatable but small loose-tolerance
crossover against unpreconditioned counting CG, not a tolerance-independent win.

The fixed structured controls remain the operational result.  Partial `spectral_q1024` plus
counting CG is the fastest eligible method at every tested tolerance.  Dense and FFT-DST direct
methods remain visible but fail the measured 1e-10 true-residual gate; the partial and full
spectral-plus-CG rows pass it.  These controls are classical and are not attributed to learning.

All exact values and intervals in `N2048.generated.md` are generated from the immutable final
JSON by `summarize_2048.py`.  That script independently recomputes every learned and structured
timing estimator, whole-case bootstrap interval, case sign, outlier count, balance assertion,
solver gate, and source/commit binding without importing the scientific driver.  The standard
audit is `runs/n2048final1/AUDIT.json`; the separate raw-array audit is
`runs/n2048final1/INDEPENDENT-AUDIT.json`.  This branch is closed with no N=2048 tuning or rerun.

An earlier local under-one-minute wiring attempt is retained as an explicitly incomplete
non-result: its streamed EQ fit consumed almost the entire local budget and the timeout stopped it
before any row completed.  A separate synthetic unit smoke verifies the ten-order position and
precedence assertions.  Neither replaces the required A100-80GB N=2048 memory smoke.
