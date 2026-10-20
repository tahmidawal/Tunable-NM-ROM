**Overall: NEEDS-RESTATEMENT.** A1 resolves several round-1 issues, but the square-system certificate, negative-control fixtures, and inherited 3D timing machinery still have consequential gaps.

Read-only audit completed. No files changed, jobs submitted, arrays copied, or scratch files created. Checks included source inspection, archive member inspection, and a small NumPy calculation of shell-completed mode counts. Draft lane code was context only, not accepted as audited implementation.

References below use:

- **D** = [lane DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/DESIGN.md).
- **L/** = files in this lane’s `experiments/jcp-dims/`.
- **Q/** = predecessor `2026-10-18-jcp-qualify/experiments/jcp-qualify/`.
- **S/** = predecessor `2026-10-14-jcp-speed3d/experiments/jcp-speed3d/`.
- **P/** = governing `reports/2026-10-17-full-experiment-plan/`.

**Round-1 ranked blockers**

| Blocker | Verdict on A1 | Evidence and remaining issue |
|---|---|---|
| 1. Timing-job certification | **NEEDS-RESTATEMENT** | D:427–436 correctly restores 2D in-job certification, anchors, Q2 and internal-state identity. D:437 inherits 3D `rollouts`/ADT and “hash identity of timed outputs”; the inherited path hashes only six output coefficient vectors, not internal states and stopping records. S/s3ctx.py:150–166,221–232. |
| 2. Unsupported controls | **WRONG** | Withdrawing production GAL=LSPG agreement and separating adverse/unavailable outcomes are correct, D:440–454. But the replacement square-system G-cert compares different residual norms and leaves incompatible tight-solve stopping tolerances unchanged, D:442–447; L/p1core.py:135–151,192–218; L/p1common.py:40–44. |
| 3. Explicit 3D modes | **CORRECT** | D:466–482 specifies propagation, identities, nondefault checks, exact shell-breaking 512, and maximum 4097. Independently recomputed shell counts agree. Implementation remains subject to code audit. |
| 4. Table lifetimes | **NEEDS-RESTATEMENT** | D:484–491 fixes the 2D design and pilots simultaneous footprints. For 3D, `Setting.drop` alone cannot release tables captured by timing-subject closures; the inherited panel constructs and warms all subjects. S/s3common.py:130–131; S/s3ctx.py:181–208,221–232. |
| 5. Qualification/decision precedence | **CORRECT**, with a separate Shapley correction | D:497–538 resolves the principal missing/invalid/unverified, empty-population, intersection, bootstrap and confirmation-precedence issues. The normalized **cost** Shapley threshold still uses an accuracy uncertainty; see below. |
| 6. GAL coverage and core/defer sets | **CORRECT at design level** | D:540–557 includes adaptive 3D GAL and LSPG, preserves shared cells, and explicitly makes the cost-free reporting deferral void. The low-level solver integration is feasible; the runner needs adaptation. |
| 7. Manifests and negative fixtures | **WRONG** | D:559–575 fixes the manifest, ladders and BE comparator coverage. But neither specifically named bad-vector archive fixture exists, and the establishment predicate does not establish the expected rollout-gate result. D:456–465; Q/configs/acc1024.json:129–184; Q/configs/q3d.json:78–184. |

**Every round-1 numbered item**

| Item | Verdict on A1 | Assessment |
|---|---|---|
| 1. Brief coverage | **CORRECT** | GAL coverage and unique-configuration deferrals now address the objections, D:540–557. |
| 2. Q1/Q2 inheritance | **NEEDS-RESTATEMENT** | State-machine and signed-weight fixes are sound, D:494–508. Square-system anchors and 3D timing identity remain unresolved. |
| 3. Decision rules | **NEEDS-RESTATEMENT** | Most edge cases are resolved, D:510–538. The normalized cost-Shapley threshold is dimensionally inappropriate, D:525–530. |
| 4. Controls | **WRONG** | Scientific outcomes are now distinguished correctly, but square certification and bad-vector establishment remain defective, D:439–474. |
| 5. Factorial arithmetic/rule names | **CORRECT** | Concrete manifest identities and successor links are required; BE no longer claims an \(m^*\), D:559–573. Original factorial arithmetic remains valid. |
| 6. 3D rounds/ladders/M | **CORRECT at design level** | Explicit-M propagation, completed counts, concrete family ladders, reconsideration and BE comparator ladders are supplied, D:476–482,565–575. |
| 7. Timing/elasticity | **NEEDS-RESTATEMENT** | 2D timing and regression safeguards are corrected, D:427–436,578–580. The 3D inherited identity contract remains insufficient. |
| 8. Budget/memory | **NEEDS-RESTATEMENT** | Actual-footprint pilots and local CBC accounting improve the plan, D:484–495. 3D closure lifetimes need an explicit bounded schedule; smoke-derived repricing must include the added GAL/BE/certification work. |
| 9. Misleading interpretations | **NEEDS-RESTATEMENT** | Bounded-search, selected-winner, adverse-control and historical-retraction wording is corrected, D:448–455,531–538,581–584. The bad-vector gate can still mistake an unsupported expected outcome for an infrastructure requirement. |

**Square-system certificate: the root predicate is meaningful, but its stated implementation contract is wrong.**

For a nonsingular square step Jacobian, the stationarity-ratio lower bound is valid. A1 should retain that qualification: squareness alone does not imply nonsingularity. Qualify explicitly says this at Q/DESIGN.md:507–509.

The solver evaluates and records the weighted residual
\[
r_w=D\mathcal R,
\]
whereas A1’s certificate uses
\[
\eta=\frac{\|\mathcal R\|}{\|Ac_n\|}.
\]

At L/p1core.py:192–218,
\[
\mathrm{tol}=\mathrm{tolf}\,\mathrm{scale},
\qquad
\mathrm{tolratio}=\frac{\|D\mathcal R\|}{\mathrm{tol}+10^{-300}}.
\]
Consequently, `tolratio × tolerance` recovers approximately **the weighted norm**, not the unweighted norm requested at D:446–447. Equality of exact root sets does not make these finite residual norms equal.

This can reject a correctly implemented solver step. For illustration, if \(D=0.01I\), `scale` and \(\|Ac_n\|\) are one, and \(\|\mathcal R\|=5\times10^{-8}\), the production weighted residual is \(5\times10^{-10}\), below its \(10^{-9}\) tolerance, but A1 rejects it. This is an algebraic example, not a measured campaign result.

Conversely, passing \(\eta\le10^{-8}\) does not imply satisfying the solver’s weighted tolerance or its stationarity threshold. Nor does a small residual guarantee small state error for an ill-conditioned Jacobian. Retaining solver validity and tight-anchor checks matters.

More seriously, inherited LSPG **tight/tighter** tolerances are `(1e-7, 0)` and `(1e-8, 0)`—L/p1common.py:40–44 and S/c3common.py:150–154. For a nonsingular square Jacobian whose ratio lower bound exceeds those thresholds, a nonzero floating-point residual cannot satisfy stationarity; the alternative residual stop requires exact zero. Changing the certificate does not change this stopping rule. A1 can therefore make a well-solved square trajectory fail its mandatory anchor.

**Required restatement:** define square-case production, tight and tighter stopping rules explicitly; distinguish weighted-norm recording consistency from the unweighted scientific certificate; retain raw norms and scales, including a precise zero-scale convention. Do not silently substitute GAL tolerances or relabel anchor failure as degradation.

**The \(M=r\) outcome logic itself is now sound.**

D:448–454 distinguishes:

- qualified improvement: control fired;
- qualified “does not improve”: adverse scientific result;
- unqualified or unresolved comparison: unavailable.

This closes the major round-1 loophole. It cannot operate reliably until square-case qualification is repaired. With multiple 3D forms now present, the manifest should identify which form’s 1→4 result controls each B6 verdict; the original 3D paragraph predates the adaptive-form panel.

**The bad-vector establishment fails both archive availability and discrimination.**

I inspected the actual archive filenames and NPZ member names:

- `Q/runs/acc1024/archive/output/qacc_acc1024/` contains `R_R384_LSPG_CNR_2.npz` and `_2.5.npz`, **not `_1.npz`**. The R384 configuration lists only factors 2 and 2.5 for LSPG-CNR: Q/configs/acc1024.json:150–162. Q/runs/ARCHIVES.sha256:20–21 corroborates this. A factor-1 file is listed under `smkA` at line 118, but that is a different fixture and must be pinned explicitly if selected.
- `Q/runs/q3d/archive/output/q3run_q3d/W_q2.npz` has **no member** for `W512|lat16384|fs|CN|0.01|K1`. The q3d configuration instead includes W512 lattice **QN-CN**, and W512 **gl24 FS-CN**: Q/configs/q3d.json:119–122,144–147. The exact requested member is absent from both the result-member map and NPZ keys.

CPU establishment is technically feasible with chunked independent bank evaluation once real fixtures are pinned. I did not substitute fixtures or claim their \(\rho_{\rm tgt}\) values.

Even then,
\[
\rho_{\rm tgt}\ge1
\]
on six frozen states does **not imply** a rollout distance greater than \(2.5\times10^{-4}\). It measures relative advection error, not accumulated trajectory error. Absolute advection magnitude, viscosity, tested directions and solver response all matter. A small target norm can produce a large relative error with little trajectory effect.

Thus A1 independently establishes a bad **integrand evaluation**, but demands failure of a different **rollout classifier**. Either establish the rollout predicate independently, or make this a control of an independently checked advection-disagreement predicate. Also define the outcome when fixture establishment fails; it must not trigger outcome-driven fixture tuning.

**The proposed 3D GAL mathematics and data interfaces are compatible.**

The necessary pieces match:

- `nl3(c,d)` returns \(M\) tested values; `nlJ3(c,d,None)` returns those values and an \(M\times r\) Jacobian: S/c3common.py:141–147.
- `p1core.make_evolve` accepts those signatures: L/p1core.py:114–118.
- `form_data` adds `hmask`, `Qm`, `Rm`, and `QtLA`: L/p1core.py:247–255.
- The existing adaptive wrapper demonstrates mesh projection, scale and decode: S/c3common.py:180–188. Projection/decode require `GTb` and `L`, preserved by the lean data path; L/vendor/quad3d/vendor/burgers3d-retry/tables.py:120–133.

However, this is **not a drop-in use of the current `Ctx` adapter**. Its adaptive dispatch hard-codes LSPG, lacks recorded internal states, and `rollouts` skips certification for `ad` specifications: S/s3common.py:190–200; S/s3ctx.py:155–161. The lane must adapt form identity, output shapes, recording, certification and timing identity. That is implementation work consistent with A1, not evidence that GAL is mathematically unavailable.

The 3D counts are correct: completed requests 1024, 1536, 2048 and 4096 yield **1027, 1538, 2052 and 4097**. Request 512 completes to 513, so the control must deliberately slice exactly 512 as A1 specifies.

**Timing and memory require one more explicit contract.**

The 2D certification and bounded-reference plan is implementable. Adjacent-rung comparisons can retain small coefficient records while releasing large tables.

For 3D, two inherited details invalidate a literal “unchanged” implementation:

1. `rom_subject` returns only saved-output `W`; its expected hash is also of `W`. Internal trajectory and stopping changes can escape detection. S/s3ctx.py:166,221–232.
2. Each subject closes over `d`; `time_panel` holds and warms the subject list. Removing `_d[rule]` does not remove these references. S/s3ctx.py:181–208,224–232.

Specify reference-plus-bounded-subject timing blocks, release subject closures and compiled references between blocks, and validate complete expected invocation counts across blocks. Pilot that actual implementation. `Setting.drop` by itself is insufficient.

The amended core/defer sets otherwise match the brief’s intent: shared A4 configurations survive, B6 ratios 1/2/4 survive, and reporting-only deferral no longer pretends to save GPU time. D:548–557; P/campaign/briefs/jcp-dims.md:35–36.

One additional decision-rule correction: D:528 compares \(|\Delta|\) with reference **error** uncertainty even when \(\Delta\) is a **cost** difference. Use separate accuracy and cost near-zero rules before reporting normalized Shapley shares.

**Overall: NEEDS-RESTATEMENT. Ranked remaining blockers:**

1. **Repair square-system LSPG certification and stopping together**, including weighted/unweighted G-cert consistency and attainable tight/tighter anchor criteria.
2. **Replace the nonexistent archive fixture references and establish a discriminating bad-vector predicate** that matches the gate actually being tested.
3. **Specify 3D timed/scored internal-state and stopping-record identity**, rather than inheriting six-output hashes.
4. **Specify bounded 3D timing-subject lifetimes**, including closure/cache release and a matching memory pilot.
5. **Separate accuracy and cost normalization thresholds for Shapley shares**, and explicitly map the expanded 3D forms to their \(M=r\) control outcomes.