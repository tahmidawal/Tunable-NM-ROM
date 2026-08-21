# Burgers 1e-3 / 10x Phase-12 preregistration

Status: prospective only. Phase 12 permits exactly one seed-11 G2/q32 cell
whose first operation is a corrected structural preflight. It permits no G1
retry, third architecture, seed 29/47, weak/EQ fit, scaling, model validation,
confirmation, or downstream rollout. No implementation or submission is
authorized by this document alone.

## Independent license and immutable evidence

Phase 12 is a new prospectively justified implementation repair, not a
retrospective choice based on a favorable timing. Phase-11 r2 stopped before
runtime projection and update 1 with bitwise-unchanged weights. Its post-run
source reconciliation found that the timed executable performed and returned
an unintended 51-state full-grid Cox control in addition to the declared 50
Cox weak and 51 K3 full work. The report and auditor inferred only the latter
two counts from output shapes. Its `0.03205502685159445 s`, `3.047233624534141x`,
and `[2.396706675073385,3.7521239409805687]` interval are retracted as an
overcharged, canonical-work-invalid timing negative.

The exact retraction is committed at
`b7aa407ec06b2f816ea8daa21368731924327444`. The immutable Phase-11 r2
JSON/NPZ/checkpoint/AUDIT SHA-256 values are
`c1a3d24a0883fa5046248f294fb40b55a584de87d42e58c2d20aedd8d63e39bb`,
`f0260ef4f2127d258480b17dbedc0a8720f1e454d9298ecfb93d697822a31aec`,
`cf73892c3f0e065951154beb77fcb4113375a9957fafe1de850f681946fca851`,
and `2146c63f2e01cf0e55e5dd6212ab7d91041db7df86359069a7c4d88c92b393d8`.
Its manifest and LOCAL SHA-256 values are
`36969c5db0b1caa7671b9620ba63338d8f8384994b082eb1c11d460450c1022e`
and `0bc8ad9467db57cfaef1c34c2a3cd2d61c7a1cf1127f0628740ee2e0350380eb`.
These bind the zero-update fact and defect; they provide no speed promotion.

The accepted P4--P10 dependency chain and information boundary are exactly the
ones bound by the Phase-11 preregistration whose post-retraction SHA-256 is
`87b835bc2727ed251c8efaf412fc17c3508299f96c3cd9994c95005f6791514b`.
Phase-9 capacity remains retracted, nonportable, unaccepted, and forbidden as
an input. Phase-5 G2 `15.057138x` and Phase-6 G1 `20.582209x` are descriptive
call-graph evidence only and are not Phase-12 gates or cross-job timing claims.

Work remains on branch `exp/2026-08-19-burgers-1e3-10x` in
`worktrees/2026-08-19-burgers-1e3-10x`.

## Fixed method and population

The only arm is the existing pure nonlinear H1 R48+P32/k24, M96/m384,
DualResUpConv32 G2 with q32/state37, exact boundary, and no POD or linear
corrector. Parameter trees remain exactly 165,954 generator, 164,384 encoder,
and 2,533 predictor parameters. The generator retains six 32-channel residual
blocks, two convolutions per block. Structural cost weights and seeds are the
same deterministic Phase-11 preflight values; scientific training, if opened,
uses the same seed-11 parameter and optimizer initialization as Phase 11.

Training may use only seed-0 cases N64 `0:512`, N128 `0:128`, and N256
`0:64`, all 51 times. Exposed selection remains N64 `512:576`, N128
`512:544`, and N256 `512:528`, and stays unread until every train gate passes.
Indices beginning at 576 and confirmation seed 20261031 remain sealed.

## Corrected structural preflight

The preflight is cluster-only on the same H200 allocation used by the possible
training continuation. It uses N1024, 50 time steps, the same four fixed live
cost trajectories, a fresh eligible cubic/exact-Helmholtz FOM, f64, and
highest matmul precision.

The compiled **timed actual route**, including charged cold recovery and
feature construction, performs only:

1. the 7--32--32--37 predictor and G2 coefficient generation;
2. 50 Cox weak evaluations with the P6 arithmetic; and
3. one K3/Pallas full-field decode of all 51 states.

Its ordered f64 output leaves are exactly:

| leaf | shape | logical bytes |
|---|---:|---:|
| K3 fields | `[51,1048576]` | 427,819,008 |
| states | `[51,37]` | 15,096 |
| coefficients | `[51,3328]` | 1,357,824 |
| weak residual | `[50,96]` | 38,400 |
| weak rho | `[50]` | 400 |

The total logical output payload is exactly `429230728` bytes. On the locked
JAX 0.10.2 compiler interface, `memory_analysis().output_size_in_bytes` must be
exactly `429230768` bytes. The report persists the complete compiler memory
analysis: device argument/output/temp/alias bytes and every host counterpart.
The auditor independently recomputes leaf count, order, dtype, shape, logical
bytes, compiler output bytes, and eligibility device bytes. The timed route
must contain zero Cox full-grid control leaves/evaluations, duplicate K3 full
decodes, weak Jacobians, trials, or failures, and remain below 20 GiB.

A separately compiled, **untimed identity route** runs on the identical fixed
inputs and weights. It computes the full 51-state Cox control, the K3 fields,
the P6 Cox weak residual/rho, states, and coefficients. The actual timed output
must agree with the corresponding identity-route K3/residual/rho/state/
coefficient leaves to relative L2 `<=2e-14`; identity K3 fields must agree with
the Cox full control to `<=2e-14`. Shapes and dtypes must match, all values must
be finite, and K3 and Cox fields must both have exact binary boundary values.
The Cox control is never called or returned by the timed executable.

After one untimed execution of each method and an immediate three-second GPU
burn, the timed actual route and live FOM run 24 repetitions on each of four
trajectories in strict alternating AB/BA order. Both methods therefore have
96 retained records and position counts `[12,12]`. Every record stores elapsed
time, position, case, finite/health/work, charged cold sample count, and the
same returned output used for grading. Persist raw repetitions, per-case
medians, overall medians, burns, exact order, and within-trajectory `>1.5x`
outlier counts. There is no cross-job timing comparison.

The live FOM must be finite with zero flags/breakdowns, returned residual at
most its locked tolerance, mean trajectory relative L2 `<=1e-3`, and worst
`<=3e-3`. The corrected route must pass both identity comparisons, exact
boundary, exact work/output/memory contracts, paired median speedup `>=10`,
and trajectory-clustered 95% lower bound `>=8`. Any failure stops before
runtime projection and update 1 with bitwise-unchanged weights. It establishes
no learned-model, predictor, selection, or downstream claim.

## Conditional unchanged Phase-11 training

Only a complete corrected structural pass opens the unchanged Phase-11
training path. The exact primary loss remains per-snapshot full-grid Cox
`||u_theta-u_FOM||_2^2/max(||u_FOM||_2^2,1e-300)`, with train-only
normalization and the fixed `0.01` head-balanced coefficient auxiliary barred
from checkpoint selection. Resolution-homogeneous batches are 8/2/1 in the
exact `N64,N128,N256` cycle with deterministic without-replacement complete
epochs and terminal-only selection.

The fixed schedule is 18 encoder+G2 epochs, bitwise encoder-to-q handoff, and
54 joint G2+q epochs. Full-cohort epoch-end checks persist. Terminal full-train
control is followed by final-q-only bounded matrix-free Cox trust recovery,
at most 40 attempts per train snapshot, with the exact Phase-11 radius,
damping, CG, rho-defined, transition, work, exhaustion, and health rules.
Globalized train mean/worst must be `<=2e-4/7e-4` at every N and pooled.

Only that pass permits the unchanged 18 predictor epochs. Train direct must
pass `<=3e-4/1e-3` before selection is read. Only then may the fixed exposed
selection direct and two-start nondeployable oracle run; their unchanged gates
are oracle `<=2e-4/7e-4`, direct `<=3e-4/1e-3`, and direct/oracle mean ratio
`<=1.5` at every N and pooled, with finite, boundary, identity, provenance,
and trust health. No favorable early checkpoint, schedule change, seed, G1,
third architecture, weak/EQ, scaling, model validation, or confirmation opens.

After the structural pass but before update 1, the unchanged conservative
no-update throughput projection measures every material training/evaluation/
trust/conditional unit and requires `1.15*projected_remaining <= actual Slurm
seconds remaining`. The cell requests one H200, 8 CPUs, 96 GiB, and 16 hours.
The projection includes the corrected actual route, separate untimed identity
preflight already elapsed, full terminal work, output/checkpoint compression,
and independent audit. A projection failure stops before update 1.

## Independent audit, corruption contracts, and cap

The independent negative-aware auditor binds the exact P4--P11 artifacts,
Phase-11 retraction commit, Phase-12 preregistration, source hashes, staged
manifest, Slurm completion, H200, GPU/f64/highest, and isolated namespace. It
rebuilds both structural callables and independently verifies their outputs,
identity, exact work, complete compiler memory analysis, live FOM eligibility,
raw timing medians/order/balance/outliers, gate, and zero-update hard-stop
semantics. If training opens, every unchanged Phase-11 data, normalization,
schedule, permutation, handoff, optimizer, checkpoint, trust, metric,
information-boundary, and decision check remains mandatory.

A fixed positive contract must accept the corrected five-leaf route. A fixed
hidden-control corruption adds the 51-state Cox control as a sixth timed output
leaf; the auditor must reject its leaf set, `427819008` extra logical bytes,
compiler output bytes, and false canonical-work record. Separate corruptions
must reject a hidden-control count without the leaf, removed/reordered leaves,
wrong dtype/shape/bytes, false memory, swapped identity/actual route, identity
or boundary failure, timing imbalance, median/outlier corruption, false speed
decision, premature update, and any provenance or downstream-boundary breach.

The cap is one Phase-12 cell. An infrastructure failure or negative result
does not self-authorize a retry or another arm. Exact checksummed pull and
remote cleanup occur only after the independent audit passes.
