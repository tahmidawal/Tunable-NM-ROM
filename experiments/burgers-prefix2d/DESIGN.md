# Bounded Burgers 2D prefix initialization

Prospective experiment authorized 2026-10-27. No numerical result yet. The user approved only this 2D pilot, with an early stop gate and a total ceiling of two allocated GPU-hours. No 3D, repeated correction cycles, training, physical mesh changes, sealed populations, or merges are authorized.

## Frozen question and target

Can one bounded width-initialization stage reduce complete query cost while returning the same admitted fine solve? This is nested initialization/subspace optimization, not multigrid.

Use consolidated fork `93f399faad67f94ea2b56fe25f13003143e64694`, code-only parent `5a64899fdcfbeb6cfe5b3371af1a8d8a3d2b568f`. Read-only historical qualification: `jcp-qualify/checks/analysis.json`, keys `256|R384|LSPG|CNR|1` and `256|R384|gauss128|LSPG|CNR|1` (P1 imported). Freeze the P1 `frozen` test-mode policy, checkpoint, rotation and initial projection. Target: L256 intervals, r384 of R512 dn256b, M1536, Gauss128 squared, CNR dt .005, T .25, two BE startup steps. Original weak analytic-advection/discrete-diffusion formulation and its step-dependent row weights are preserved, including its documented deviation from the general upwind rule.

Direct comparator is byte-preserved `jcp-qualify/p1core.py`: previous/linear/quadratic residual-minimizing predictor, damping carry, exact analytic Jacobian, damped normal equations, trust clipping and adaptive LM. Production gtol .001 and tolf 1e-9, budget600, all original decisions unchanged. Tight/tighter anchors use gtol1e-7/1e-8 and tolf0. The copied `prefixcore.py` may only add a bounded candidate before this unchanged fine LM and diagnostics. Timings compare candidates against the original fused query, not a slower instrumented baseline. Recorded direct parity checks all internal states, decisions and coefficients.

## Candidate arms

Let $P=[I;0]$ insert r128 coordinates into r384. Hold every fine residual row, quadrature point, weight and accepted fine history $H$ unchanged.

$$\phi_f(a;H)=\tfrac12\|r_f(a;H)\|^2.$$

- `direct`: actual original warm fine solver.
- `affine2`: at each actual fine predictor $w$, at most two coarse LM corrections minimizing $\phi_f(w+P\delta;H)$ from zero. Cache offset fields and retain nonlinear cross terms. Coarse tolerance is the same numerical gtol/tolf, on the restricted Jacobian, only to stop unnecessary coarse work; it is never a final certificate. Coarse step trust uses the fine trust bound, and starts from current fine damping. Its updated damping is discarded.
- `pure2`: at most two corrections minimizing $\phi_f(Pc;H)$ from the leading predictor coefficients. Its initial truncation is intentionally distinct from exact coarse-to-fine injection. Guard against the ORIGINAL fine predictor.

For both candidates, finite proposed state and objective reduction by at least a relative1e-8 against the original fine predictor are required. Otherwise restore the original predictor. Fine LM receives the original fine damping in either case; its output updates the ordinary history/damping carry. Every attempt is counted. A coarse solve may hit its work limit because this is an approximate initialization, but final fine admission is mandatory. No separate coarse trajectory supplies history.

## Early stop gate

Only dev6 is eligible initially; regenerate descriptors and IC fields from existing recorded seeds on the cluster. No references or validation cases are loaded in this phase. Reproduce original baseline, verify exact prefix transfer and restricted analytic Jacobian against independent autodiff, then run direct, affine2 and pure2 complete trajectories at identical production settings. Seven interleaved timing blocks with a frozen RNG order, GPU burn-in before every block, synchronization and raw arrays. Times include input projection, all evolution/candidate/fallback/guard work and identical full six-field output; outputs stay on device equally. Compilation/setup and offline certification are reported separately, included in resource use.

Measure direct residual/Jacobian and narrow-Jacobian kernels on fixed common-history states, fine versus coarse corrections/Jacobian counts, and actual complete-query ratios. These diagnose cost, not replace fused timings or establish a theoretical lower bound. Record field-metric fraction of the direct predictor-to-solution correction captured by the prefix; energy ordering is not a frequency ordering. A capped20-correction prefix diagnostic on fixed development steps may be added only before validation and is a local diagnostic, not a global oracle.

Transfer or original-baseline reproduction/certification failure stops the entire screen. Otherwise gate each candidate separately: it must achieve median paired query ratio <=.85 on dev6 and pass every own production certificate/timing identity. A failed candidate is rejected; its failure does not reject another qualified candidate. Stop without val32 if neither qualifies. Publish every case and fallback; do not tune widths, work budget, mesh, target or tolerances after seeing the gate. No success/generalization claim can follow from dev6 alone. A failed early screen establishes only that these bounded arms failed this development gate, not that all multilevel methods fail.

If a candidate passes, freeze it, implement/review a carried-fine-Jacobian/chord control using the same final fine target/admission, and evaluate direct/candidate/control on all exposed val32 with the same seven-block timing protocol. A separate equal-width subspace control is required only for an ordering-specific claim, which is not intended. Before validation run independent code review and ensure remaining reservation covers all cases; stop incomplete instead of selecting inexpensive cases. No extra user approval is needed for this already-approved bounded 2D validation, but any changed target or scope requires discussion.

## Numerical and cost gates

Independent fresh autodiff gradients for every internal step in each evaluated trajectory. Preserve existing returned-state stationarity/residual rules and record termination reasons. Original certificate agreement <=1e-7 times recorded gamma +1e-12. Production-to-tight maximum common-field distance <=2.5e-4, tight-to-tighter <=2.5e-5, fixed initial-field norm. All internal states and saved output times are checked. Common-history nonlinear-step objectives are compared separately from independently evolved trajectories. Tight anchors do not enter production timing. Tight/tighter failure stops acceptance, rather than loosening tolerances. Exact synthetic directional-derivative, transfer and nonzero-tail tests must precede GPU pilot.

If early cost gate fails, stronger physical accuracy/quadrature validation is unnecessary and must be labelled not run; no matched-accuracy speed conclusion is then permitted. If it passes, all candidates need own tight/tighter trajectories, common-history endpoint checks, independent final objective/gradient checks, Gauss160/192 successor witnesses and existing unsealed E0 reference scoring with explicit reference eligibility/uncertainty. A lower objective alone cannot certify the same nonconvex minimum.

Final success requires median candidate/direct total online ratio <=.85; a case-level paired confidence interval excluding1; all admission and field-agreement gates; no resolved physical-error regression under saved reference uncertainty; p95 slowdown <=10%, with all ratios and maximum published. Separate repeated-run noise from case variation. Also report comparisons to chord, setup/first-query/amortized costs and the query count needed to repay added setup. An internal improvement is not superiority over POD/spectral competitors.

## Operations and budget

Unique namespace `/cluster/tufts/paralab/tawal01/bprefix2d-20261027`. Exactly one job/directory. f64, highest precision, gpu partition, GPU preflight. First reservation20 minutes, cumulative reservation plus settled actual <=7200 GPU-seconds. Every failure/compile/diagnostic included. Local numerical smoke, if used, must be <60 seconds through jaxrun and charged to the cap. No bare scancel. Check queue before/after submissions. SCP source directly to final paralab directory, hash all source/inputs, regenerate case data from seeds, checksum returned results then remove remote duplicate directory. Preserve unique local arrays. Commit/mirror every checkpoint. Independent exact-job review before submission and result review before conclusion. Main canonical log closes the session; ask merge question at completion.

## Glossary

Bank: frozen spatial features. Prefix: first ordered features of the same bank. r: active coefficient count. M: test-mode count. Quadrature: points and weights approximating integrals. L: intervals per physical axis. CNR: Crank–Nicolson with the inherited two backward-Euler startup steps. LM: damped Gauss–Newton nonlinear solver. Warm start: temporal prediction from accepted prior states. Fine target: exact residual, time history and acceptance rules being solved. Admission: numerical consistency checks, not physical accuracy. dev6/val32: already-exposed development/validation populations; neither is sealed confirmation. Fused query: complete timed initial projection/evolution/output call. Chord: classical reuse of an earlier Jacobian. Gate: predeclared condition to continue or stop. Allocated GPU-hours: reservation elapsed time on one GPU, including failed work.
