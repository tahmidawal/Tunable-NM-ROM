**Overall: NEEDS-RESTATEMENT.** A2 repairs most round-2 objections, and both replacement fixtures are real and correctly pinned. Three design gaps remain: QN timing-record names, precedence between the state-frozen gate and rollout qualification, and zero-scale G-cert recovery.

Read-only audit completed. No files changed, jobs submitted, scratch files created, or arrays copied to disk. Draft code was inspected for interface evidence, not accepted as audited implementation.

References:

- **D** = [lane DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/DESIGN.md).
- **L/** = this lane’s `experiments/jcp-dims/`.
- **Q/** = `2026-10-18-jcp-qualify/experiments/jcp-qualify/`.
- **S/** = `2026-10-14-jcp-speed3d/experiments/jcp-speed3d/`.

Your task explicitly asks whether **A1** resolves the round-2 findings. The tables therefore distinguish that literal assessment from the current **A1+A2** contract.

**Round-2 ranked blockers 1–5**

| Blocker | A1 alone | Current A1+A2 assessment |
|---|---|---|
| 1. Square-system stopping and certification | **WRONG** | **NEEDS-RESTATEMENT.** D:590–605 fixes weighted versus unweighted norms and supplies positive tight/tighter residual tolerances. D:599–600 still gives incorrect raw-norm recovery at zero scale. |
| 2. Fixture availability and discriminating predicate | **WRONG** | **NEEDS-RESTATEMENT.** Both replacements at D:616–622 exist with matching hashes and keys. D:613–629 now tests the established advection predicate, but does not explicitly override A1’s Q1 prerequisite at D:522–524. |
| 3. 3D timed/scored internal identity | **NEEDS-RESTATEMENT** | **WRONG as the stated interface.** D:632–634 requires `it` and `rej` for QN; QN returns neither. S/qn3.py:228–240; S/s3common.py:245–246. The intended identity requirement is sound. |
| 4. Bounded 3D timing lifetimes | **NEEDS-RESTATEMENT** | **CORRECT at design level.** D:637–643 explicitly releases subject closures, query-cache entries and rule data; adds a memory-release check, complete invocation counts and a matching pilot. |
| 5. Shapley thresholds and expanded control mapping | **NEEDS-RESTATEMENT** | **CORRECT at design level.** Separate cost/accuracy thresholds at D:645–647; form-specific control mapping and the FS exception at D:604–608. |

**Every row of the round-2 “Round-1 ranked blockers” table**

| Row | A1 verdict | Current disposition and evidence |
|---|---|---|
| 1. Timing-job certification | **NEEDS-RESTATEMENT** | Still needs the QN identity correction. D:427–437,631–635. |
| 2. Unsupported controls | **WRONG** | Scientific-outcome handling is fixed, but square G-cert needs its exceptional-case correction. D:440–455,590–608. |
| 3. Explicit 3D modes | **CORRECT** | No new design objection. Explicit M, maximum 4097, hashes and nondefault checks remain specified. D:466–482. |
| 4. Table lifetimes | **NEEDS-RESTATEMENT** | Resolved prospectively by A2.4. D:484–495,637–643. |
| 5. Qualification/decision precedence | **CORRECT**, for the principal state machine | A2 introduces an unresolved interaction with the cross-family gate’s Q1 prerequisite. D:497–538,610–629. |
| 6. GAL coverage and core/defer sets | **CORRECT** | Coverage remains explicit; A2 removes square FS cells and maps their control to adaptive LSPG. D:540–557,604–608. |
| 7. Manifests and negative fixtures | **WRONG** | Manifest corrections stand; replacement fixture identities now pass inspection. Gate precedence still needs restatement. D:559–575,616–629. |

**Every row of the round-2 “Every round-1 numbered item” table**

| Row | A1 verdict | Current disposition and evidence |
|---|---|---|
| 1. Brief coverage | **CORRECT** | D:540–557,604–608. No new coverage blocker. |
| 2. Q1/Q2 inheritance | **NEEDS-RESTATEMENT** | Square exceptional-case G-cert and QN timing identity remain. D:597–600,631–635. |
| 3. Decision rules | **NEEDS-RESTATEMENT** | Shapley correction resolved at D:645–647; new cross-family precedence ambiguity remains at D:522–524,613–629. |
| 4. Controls | **WRONG** | Real fixtures and matched predicate repair most objections; zero-scale G-cert and gate availability remain. D:590–629. |
| 5. Factorial arithmetic/rule names | **CORRECT** | D:559–573 remains sufficient at design level. |
| 6. 3D rounds/ladders/M | **CORRECT** | D:476–482,565–575; A2’s FS square exclusion is explicit at D:604–608. |
| 7. Timing/elasticity | **NEEDS-RESTATEMENT** | Elasticity safeguards stand; QN hash schema is wrong. D:578–580,631–635. |
| 8. Budget/memory | **NEEDS-RESTATEMENT** | Resolved at design level by bounded timing blocks and smoke-derived repricing including added work. D:637–650. Actual feasibility remains a smoke gate. |
| 9. Misleading interpretations | **NEEDS-RESTATEMENT** | Bounded-search and adverse-outcome wording remain sound. Bad-vector control availability still needs unambiguous precedence. D:448–455,581–584,623–629. |

**A2.1: square stopping is substantially repaired; zero-scale G-cert is not.**

The revised positive-scale contract matches `p1core`:

\[
r_n=\|D\mathcal R_n\|,\qquad
\mathrm{tol}=t_f\,\mathrm{scale},
\]

with validity requiring stationarity or residual convergence, plus the existing exit checks. See L/p1core.py:135–151,192–199. The new tight/tighter pairs remove the inherited exact-zero residual requirement. They do **not** guarantee those anchors converge in floating point; unsuccessful anchors must remain qualification failures, as A1.5 requires.

The weighted certificate, reported literal stationarity ratio, conditioning flag and retained anchor comparisons are coherent. A small weighted residual alone is not a state-error guarantee, but the design does not rely on it alone.

The remaining error is explicit:

- D:599–600 recovers the recorded norm as `tolratio × tol`.
- L/p1core.py:213,218 records
  \[
  \mathrm{tolratio}=\frac{r_n}{\mathrm{tol}+10^{-300}}.
  \]
- At zero scale, `tol=0`; the stated recovery always returns zero for finite `tolratio`, even when the solver recorded a nonzero residual.

For example, a raw residual of \(10^{-10}\) gives a finite recorded ratio of \(10^{290}\). Multiplication by zero loses that evidence. The separate certificate should reject the step, but the **G-cert consistency test is still wrong**.

**Required correction:** recover the raw norm using `tolratio × (tol + 1e-300)`, or explicitly inherit the zero-scale branch from Q/DESIGN.md:533–537 and Q/qcert.py:84–95. Require finite recovered values; overflow must fail G-cert. Preserve raw norm and scale in the evidence.

**A2.2: fixture identities pass; numerical admission remains unestablished.**

I independently streamed both files through SHA-256 and inspected their NPZ members.

| Fixture | Verified result |
|---|---|
| 2D `Q/runs/smkA/archive/output/qacc_smkA/R_R384_LSPG_CNR_1.npz` | SHA-256 exactly `3b00ba0f24261e4792fa05468313f7a6e1ab349ce883b53ce18fdd65a3c93dee`. `case_ids[0]` is `dev6|0`; `W` has shape `(6,6,384)`. The required six output states are `W[0]`. |
| 3D `Q/runs/q3d/archive/output/q3run_q3d/W_q2.npz` | SHA-256 exactly `0ca5015571184c2feba922cf014d18bdcbc7feba99fc009e60fa863989583f5b`. The exact member at D:620 exists, shape `(64,6,512)`, with matching `cases|…` member containing cases 0–63. All its coefficients are finite. |

This resolves the nonexistent-fixture finding. The 2D prefix construction is also explicitly labelled as a kernel fixture rather than an R128 reached trajectory, D:618–619.

Independent CPU establishment is feasible:

- 2D has hand-propagated NumPy bank derivatives and tested-advection evaluation: L/vendor/quad2d/npbank.py:24–42; L/npcheck.py:43–61.
- 3D has independent NumPy bank derivatives, mode ordering and tested advection: Q/vendor/quad3d/npcore.py:26–52,67–79,103–109.
- Both computations can accumulate the six tested-advection vectors over point chunks without retaining full rule tables or writing large arrays.

The existing `npcheck.tables` and `npcore.offmesh_adv_np` helpers materialize tables; the establishment runner should stream chunks rather than call them on the entire finest rule.

The revised discrimination argument is now valid **conditionally**:

\[
\rho_{\rm tgt}\ge1 \quad\Longrightarrow\quad \rho_{\rm tgt}>0.116.
\]

It tests the same quantity as component (b), unlike A1’s unjustified implication from advection disagreement to rollout disagreement. I did **not** compute the full establishment values and do not claim either fixture is admitted.

The unresolved problem is **gate availability**:

- A1 says the cross-family gate is evaluated only when both finest-rung and target rollouts are valid and Q1-qualified on every case: D:522–524.
- A2 says an admitted fixture must make component (b) flag `badlat`: D:628–629.
- Component (b) uses frozen states and does not mathematically require a successful `badlat` rollout.

Thus, an independently established bad integrand with a Q1-failing rollout can encounter either “gate unavailable” or a mandatory flag, depending on how the amendments are interpreted.

**Required correction:** give (a) and (b) separate availability rules. Evaluate finite, established component (b) independently of rollout Q1; a failing available component should flag the family even if the other component is unavailable. A family should pass only when every required component is available and passes. Also state the zero-target/nonfinite convention for establishment explicitly; nonfinite computation should not admit a fixture merely because an inherited helper returns infinity.

“Ten times the bar” at D:626 is also numerically inaccurate: \(1/0.116\approx8.62\). This wording issue is not itself a blocker.

**A2.3: FS records exist; the QN record contract names nonexistent arrays.**

The actual interfaces differ:

| Path | Actual returned evidence |
|---|---|
| FS | `W_all`, `reason`, `it`, `rej` exist. L/fs3.py:165–167; S/s3common.py:167–182,204–212. |
| QN | `W_all` and `reason` exist; **`it` and `rej` do not**. Records include `q`, `n_exec`, `n_ref`, `n_bt`, `n_clip`, `n_mu`, `refresh`, etc. S/qn3.py:228–240; S/s3common.py:245–246. |
| Inherited adaptive query | Uses `t2core.make_evolve` without the P1 recording extension; it does not supply the proposed internal `states`. S/s3common.py:190–200. A1.6 already prospectively requires replacing this path with recording `p1core`. |

Furthermore, the inherited timing subject returns **only `W`**, and the timer hashes that array: S/s3ctx.py:193–199,221–232. A2 correctly requires changing this behavior; it is not already implemented by inheritance.

**Required correction:** freeze solver-specific hash payloads. For QN, use its actual per-step work/stopping records—at least `reason`, `q`, `n_exec` and the relevant rejection/backtracking/refresh records—or explicitly define new aliases and their semantics. Include shape/dtype/key boundaries in deterministic serialization.

The timed wrapper must return the full identity payload, synchronize it before stopping the timer, and hash afterward. Preserve the full-field query contract. The adaptive implementation must explicitly request sufficient state and step-record buffers.

**A2.4: bounded timing blocks are a sound design repair.**

A2 now addresses the actual ownership problem: deleting `Setting._d[rule]` alone cannot release data held by closures. Its closure/cache/data release sequence and matching pilot are sufficient prospectively.

Two implementation checks remain for the code audit:

- Remove every reference, including warmed-subject lists and compiled component handles, while preserving A.
- Measure against a fixed warmed baseline or enforce the overall live-memory ceiling continuously. Allowing nearly 1 GB of unreleased memory relative to each immediately preceding block can accumulate across many blocks.

Neither observation requires rejecting the bounded-block design itself. A2.6 correctly requires repricing certification, anchors, added GAL and BE ladders before production.

**A2.5: Shapley threshold repair passes.**

The cost criterion is now dimensionless, requires an interval excluding zero, and leaves absolute contributions available when normalization is unstable. Accuracy normalization retains its accuracy-scale threshold. Together with A1’s common populations and frozen selections, this is coherent as a selection-conditioned descriptive analysis. D:525–530,645–647.

**Overall: NEEDS-RESTATEMENT. Ranked remaining blockers:**

1. **Correct the 3D timing identity schema:** QN has no `it`/`rej`; specify actual solver-specific records and the wrapper that returns them.
2. **Resolve cross-family gate precedence:** component (b) must have explicit availability independent of a bad rule’s rollout Q1, with finite/zero-denominator handling.
3. **Correct square-system zero-scale G-cert recovery:** include the denominator’s \(10^{-300}\) term and explicit nonfinite/overflow failure behavior.