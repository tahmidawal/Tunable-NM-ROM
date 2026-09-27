# Burgers 1e-3 / 10x Phase-10 diagnostic preregistration

Status: prospective. Phase 10 permits exactly one train-only diagnostic cell,
`p10_d_r1`. It performs no optimizer or model update and cannot itself authorize
an architecture change, training submission, selection evaluation, weak/EQ,
scaling, model validation, or confirmation.

## License and immutable model

The diagnostic is licensed by valid P4--P7 evidence and the independently
accepted Phase-9 terminal full-field result. Phase-8 remains descriptive and
invalid for promotion. All Phase-9 capacity work remains retracted as
nonportable, unaccepted, and unusable for any decision or license.

The fixed learned model is the exact Phase-9 G1/q19 terminal generator in
checkpoint SHA-256
`90e9df6388bf3c05905d52c5ff073f331728ffd493a965e4376e4df285bb07d9`.
Its immutable recovery JSON and NPZ are
`8d86ab90c1d293e4810c3d3a9391db48f635f8c77a7a8c30e71809a112bc7f8d`
and `d6e5001d8dcb5ba7fb269241b989492533379807bcbc4cb65a384f7d429c773e`.
The accepting audit-only JSON, work NPZ, LOCAL, and staged-manifest SHA-256
values are respectively
`b7bb908addaeb54c81293624c4433c61388c03faf9f514dbe5c8f3ec9d281377`,
`66ef97a0daeb2036e0aa6ccc3f31863e6861702baaa9711311d1839541cd556b`,
`65e546fa0226fa0cc50a9d4f5953f53f1b51995a1741be656481621571a35a65`,
and `9173714f5a03feef10a9f22f21b4c0c6466caad34e710ccb3131ebb655730edc`.
The original final work checkpoint remains bound at
`dd7b5fecc07e9021c6264febf5c7187004daaa73a50381921d5190eede639ce0`.
The generator, all other model trees, q, normalization, and source artifacts
are read-only and byte-checked before and after the diagnostic.

## Information boundary and objective

Only the locked seed-0 full training population is regenerated: all 51 times
for N64 cases `0:512`, N128 `0:128`, and N256 `0:64`, totaling 35,904
snapshots. All 16 immutable P5 target chunks and the P4--P8 dependency chain
are checked. Train-only output normalization is regenerated exactly. Selection
cases/fields/coefficients, model validation, confirmation, seeds 29/47,
weak/EQ, scaling, and downstream rollout data remain sealed.

The G1 generator is frozen. For each train snapshot the variable is q directly,
bounded componentwise in `[-1,1]^19`. The sole start is exactly
`q0=tanh(final_q_raw)` from the accepted terminal checkpoint. There is no
encoder, predictor, free-target, zero, random, or restart control. The sole
objective is the Cox full-grid discrete value

```
ell(q) = ||u_theta(a,q;G1)-u_FOM||_2^2
         / max(||u_FOM||_2^2,1e-300).
```

Every grid point enters every objective. K3 is charged only at initial and
terminal full-cohort verification as an identity control; it never seeds or
changes the optimization.

## Fixed globalized-q protocol

Each snapshot receives at most 40 matrix-free Gauss--Newton/LM attempts. CG
solves `(J^T J + lambda I)p=-J^T r` from zero for at most 19 iterations at
relative tolerance `1e-12`. The step is radius-scaled, then projected into the
box. Fixed values are `Delta0=0.25`, `Delta in [2^-20,1]`, `lambda0=1e-6`,
and `lambda in [1e-12,1e12]`.

A step is accepted only with finite work, no CG breakdown, positive predicted
and actual decrease, and defined `rho=actual/predicted >=1e-4`. If predicted
decrease is nonpositive, store finite `rho=0`, `rho_defined=false`, and reject;
no update branch reads undefined rho. Rejection or defined `rho<0.25` divides
Delta by four and multiplies lambda by ten. Accepted `rho>0.75` divides lambda
by three and doubles Delta only when the projected step norm is at least
`0.9*Delta`; other accepted steps retain both. Accepted relative objective or
relative step change at most `1e-12` terminates that row. Exhaustion at minimum
radius and maximum damping, breakdown, nonfinite work, false rho/decision, or
any transition/work-count mismatch is unhealthy.

Persist q/objective/Delta/lambda/active at attempts 0--40 and all attempted,
trial, predicted, actual, rho/rho-defined, acceptance, termination, gradient,
step, CG, JVP/VJP, bound, exhaustion, and elapsed health work. Progress and a
terminal work checkpoint expose no scientific metric while running. The
independent negative-aware audit regenerates data/normalization and recomputes
initial and terminal Cox/K3 arrays, trajectory aggregation, every trust
transition, work count, immutable binding, and decision.

## Gates and interpretation

At every N and pooled, initial and recovered trajectory mean/worst use the
locked aggregation and must be finite, have zero boundary violations, and K3/
Cox identity at most `2e-14`. The unchanged recovered train accuracy gate is
mean `<=2e-4` and worst `<=7e-4` at every N and pooled.

- A healthy gate pass establishes only that the fixed trained G1 map can
  represent the training truth after globalized q recovery. It permits a new
  prospective proposal for exactly one corrected G1 alternating-optimization
  arm; it does not authorize implementation or submission by itself.
- A healthy miss establishes only that this fixed trained G1 map misses. It
  does not prove G1 architectural insufficiency and cannot license G2 or any
  richer architecture. It hard-stops Phase 10 pending a separately audited
  architecture-justification proposal.
- An unhealthy diagnostic is invalid and hard-stops without interpretation.

No result is a deployable inference or speed claim. The original pure-nonlinear
`1e-3`/`10x` objective and the proven P6 Cox-weak/K3-full route remain unchanged.

## Resources and cap

The sole real cell is cluster-only on one fixed H200, 8 CPUs, 96 GiB host RAM,
f64, highest matmul precision, and 16 hours. The 35,904-by-40 one-start work is
about 3.14 times P8-D's 5,712-by-40 two-start attempt count; P8-D completed in
33m35s on an H200, making 16 hours conservative without a cross-job timing
claim. Local execution is limited to one excluded synthetic smoke under 60
seconds through the mandated GPU wrapper. There is no retry or second D cell.

## Prospective portability amendment after zero-attempt r1

Job `2738710` reached GPU/f64/highest and independently regenerated the initial
full-field control, then stopped before trust attempt 1 because the driver used
exact Python dictionary equality against the accepted metric summary. It wrote
no JSON, NPZ, work checkpoint, or audit; performed zero optimizer updates; and
touched no selection, capacity, or downstream data. The exact failure bundle is
preserved with LOCAL SHA-256
`ac033e5fcff3afced17886831307c3bb6913fbda7c73607c878c939086e8f3a1`.
This is an unconsumed infrastructure/portability failure, not the one Phase-10
diagnostic outcome. Exactly one fresh `p10_d_r2` lifecycle is permitted after
root audit; there is no further retry.

Before attempt 1, r2 persists an immutable `initial_control.npz`. The driver and
independent auditor compare every required recomputed initial Cox/K3 floating
metric and floating array with the accepted immutable recovery control using
`rtol=2e-13, atol=2e-14`, finite values, and identical shapes. They persist
maximum absolute differences and exact-versus-portable classification. Metric
keys/types, Boolean health, integer boundary counts, q/affine ordering, source
model/q/normalization hashes, and the unchanged zero-boundary and K3/Cox
`<=2e-14` identity gates remain exact. No trust parameter, objective, data,
model, loss, scientific gate, decision branch, or method changes.

## Prospective audit-only amendment after completed r2

Job `2739690` completed all 40 fixed trust attempts and sealed the scientific
JSON, NPZ, initial control, and work checkpoint with zero optimizer updates.
It then exited `1:0` only because the independent repeat imposed nonportable
host/device exact predicates. The exact failed-audit bundle is preserved at
commit `1f16217deb1f982fa643302afd798a7d0a25325f`; its `LOCAL.sha256` digest is
`36076277caf1372eb4d02735219bda76a50ace2f8d0e5d1a5ab6dd3ba88e6990`.
The scientific artifacts and original failed AUDIT remain immutable and no
scientific rerun is permitted. One separate audit-only H200 cell may decide
whether the completed r2 result is valid.

The repaired auditor independently compares the persisted driver
`initial_control.npz` directly with the accepted immutable source. Floating
metrics and arrays require finite, identical-shape values satisfying the fixed
componentwise bound
`abs(repeat-reference) <= 2e-14 + 2e-13*abs(reference)`; Boolean health,
integer boundary counts, q/affine ordering, and source-model/normalization
hashes remain exact. A separately regenerated full-field/data repeat uses the
same bound for floating arrays and reported metrics and persists, per field,
the maximum absolute difference, maximum relative difference, and maximum
bound-normalized difference. Shapes, dtypes, finiteness, categorical values,
zero boundary, source identities, and the unchanged `2e-14` K3/Cox gate remain
exact.

Every trust transition, acceptance, work count, cap, and decision rule remains
unchanged. A recorded device termination predicate must equal the host
recomputation except when an accepted row's recomputed relative-improvement is
within `4e-16` of the fixed `1e-12` termination threshold; only that predicate
may follow the persisted device Boolean inside the band. The audit persists
the mismatch indices and distances and rejects any mismatch outside the band.
Corruption controls must reject a larger floating perturbation, a categorical
or boundary change, and a termination mismatch beyond the band. The audit-only
cell performs no optimizer/scientific update and opens no selection, capacity,
weak/EQ, scaling, model-validation, confirmation, or downstream data.

## Prospective audit-only schema correction after audit r1

Audit-only job `2747981` passed H200/GPU/f64/highest preflight but exited
`1:0` before producing an audit artifact because the auditor looked for
`coefficient_mean` at the top level of the accepted terminal checkpoint.  The
immutable checkpoint stores train-only normalization under the exact nested
schema `normalization.{mean,scales,feature_mean,feature_scale}`.  The verified
zero-output failure is infrastructure-only and does not consume the one audit
decision or alter completed r2 science.

Exactly one fresh audit-only r2 may replace only that invalid lookup with the
existing nested schema.  It must bind all four nested arrays bitwise to the
corresponding immutable recovery NPZ arrays and to their recorded recovery
report hashes; the Phase10 mean/scales arrays and hashes remain independently
bound as already specified.  A real-checkpoint-schema regression and a
corrupted nested-array/hash regression must pass before staging.  No driver,
scientific artifact, trust rule, gate, tolerance, model, data split, or
information boundary changes, and no scientific rerun, are permitted.
