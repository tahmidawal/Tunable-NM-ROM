**I would fix the acceptance and held-out workflow before launching.** I found no algebraic error in E1–E3, no broken head monkeypatch, and no definite JSON/key crash in the supplied development configs.

This was read-only: no files changed, no GPU execution, no cluster actions. Validation included source tracing, AST comparison, config regeneration **in memory**, and NumPy-only projection checks.

**BLOCKERS — must fix before running**

1. **The held-out audit cannot produce the promised confirmation report.**  
   [make_configs.py:175](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/make_configs.py:175) disables certificates and includes only the selected ROMs and two parent picks. However:
   - [audit_b2speed.py:395](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/audit_b2speed.py:395) still requires the parent q0 arm to reproduce its **dev6** error. That parent arm is absent at 256/1024; at 512 it is tested on a different cohort.
   - At line 404, selection requires certificates, which the held-out job deliberately skips. Consequently `accurate`/`fast` are empty and the Table-1 comparison is not produced.
   
   Add an explicit held-out audit path that consumes the frozen development selection, reports those picks regardless of outcome, and does not apply development-cohort reproduction checks.

2. **The neighbour gate repeats the case-mix problem recorded in the lab log.**  
   [b2speed.py:670](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/b2speed.py:670) pools raw times across cases when comparing long versus short predecessors. Random ordering does not ensure identical case mixtures, especially with only three FOM repetitions. A slow-case imbalance can manufacture—or conceal—an order effect. [audit_b2speed.py:309](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/audit_b2speed.py:309) duplicates that estimator.
   
   Additionally, subjects with fewer than three observations in either group disappear from the gate; passing does not establish coverage of every subject/phase. Use a case-controlled estimator and require explicit coverage or report “not evaluable.”

3. **FOM compile-mode parity is substantially weaker than the stated guarantee.**  
   ROM pairs are generated in [make_configs.py:93](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/make_configs.py:93); no FOM pairs are generated. [audit_b2speed.py:356](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/audit_b2speed.py:356) checks only equality of the **single worst error** across modes. Different fields, different cases attaining that maximum, and different Newton iteration histories can pass.
   
   Compare FOM fields and per-step integer diagnostics across modes before treating their minimum timing as iterate-preserving engineering.

4. **The parity gate does not establish “same iterates.”**  
   [b2speed.py:510](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/b2speed.py:510) retains six output fields and timestep diagnostics; line 543 compares one norm over those six fields. It does not compare the 51 stored timestep states, initialization diagnostics, or rejected-step counts. Divergent intermediate states that reconverge at output times can pass.
   
   Moreover, only R′=128/384 and q0 have parent twins. New widths and cap=1 compare engineered default against engineered graphs, which cannot expose a shared engineering error. Add direct E2 residual/Jacobian checks and capped-versus-parent-budget-1 checks; compare all timestep states for the claimed parity scope.

5. **The injected error control does not demonstrate rejection by the real audit gate.**  
   [audit_b2speed.py:179](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/audit_b2speed.py:179) accepts restricted/full error ratios in `[0.5, 1.05]`, but lines 459–464 declare the 0.1% perturbation detected using a separate absolute-gap threshold. An unperturbed ratio of 0.86 already has a gap of 0.14; the control “passes” even though the perturbed record remains accepted.
   
   Run injected records through the actual acceptance predicate and require acceptance before injection and rejection afterward. Otherwise the mandatory independent-audit control is misleading.

**SHOULD-FIX**

6. **Resolve selection-rule discrepancies before registration is frozen.**  
   [audit_b2speed.py:360](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/audit_b2speed.py:360) excludes every FOM setting with any unconverged step, while DESIGN §6 specifies eligibility by error alone. This extra filter can select a slower comparator and inflate speedup relative to the stated rule. Either disclose that requirement or implement the written rule.
   
   At lines 398–410, fast selection uses the historical `pb['percent']`, not the same-job q0 error `got` promised by the design. The reproduction tolerance makes the discrepancy small, but boundary candidates can differ.

7. **The general-path comparison needs narrower claims and an accuracy criterion.**  
   [b2fast.py:297](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/b2fast.py:297) correctly applies `varpro.make_block_lm` with `K=Rp, q=0`: every coefficient receives LM damping and the trust restriction; there is no undamped correction block.
   
   It is a **reconstruction of the general solver on the linear rung**, not a replay of the historical nonlinear-head query. The comparator also changes clipping, damping carry, and predictor, so its factor includes solver-policy changes. [audit_b2speed.py:443](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/audit_b2speed.py:443) always computes X without testing “same error.” Predefine that accuracy condition before making the paper claim.

8. **Qualify the certificate interpretation and independence.**  
   [DESIGN.md:78](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/DESIGN.md:78) correctly excludes k=0 from the *accepted backward-Euler endpoint* condition. But “never evaluates” is too strong: [b2fast.py:240](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/b2fast.py:240) evaluates predictors including the initial state, and LM evaluates intermediate trial states.
   
   The certificate covers stored endpoints, not all online advection evaluations. For x1, primary k≥1 additionally tests an exact-step endpoint; actual EQ endpoints begin at k=2.
   
   The reused eqcert populations—including their “confirmation” draw—also participate in candidate selection. They are certification/selection data, not fresh confirmation of the final winner. I found explicit cohort-disjointness checks, but disjointness does not undo previous inspection.

9. **Held-out hashing weakens the stated repetition guarantee.**  
   [make_configs.py:179](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/make_configs.py:179) sets full hashing only every eighth invocation. [b2speed.py:636](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/b2speed.py:636) checks a spatial subsample otherwise. Those invocations cannot be described as full-output SHA-identical.

10. **Freeze and retain the complete selection provenance.**  
    [make_configs.py:170](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/make_configs.py:170) reads a working-tree selection without checking that it is committed. [stage.py:20](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/cluster/stage.py:20) stages committed execution inputs, which is good, but omits the selection artifact and audit source. Also specify whether held-out FOM comparators are frozen or reselected from the fixed grid: the selection writer currently records no chosen FOM setting/mode.

**NOTES — checks that passed and remaining limits**

- **E1/E3 are correct algebraically.**  
  [b2fast.py:53](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/b2fast.py:53) matches `hops.lattice_rule`’s i-major ordering, equal weights `(L/s)²`, and coefficient `2/L`. Both contractions preserve paired mode indices. NumPy-only checks at all three target meshes gave E1 discrepancies below `6×10⁻¹⁶`; E3 versus `hops.sep_project` was below `1.2×10⁻¹⁵`. These are algebra checks, not GPU parity evidence.

- **E2’s stencil, boundaries, layout, and scaling are correct.**  
  [b2fast.py:198](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/b2fast.py:198) uses `(c,xp,xm,yp,ym)`, matching `advect_points`; zero-padding matches `engines.spatial`. Bank construction is i-major, giving zero-based row `i*(L−1)+j`. The derivative is correctly
  `A + dt*S*project(dAdv)`: multiplying the linear derivative by `S` cancels `1+dt*nu*lam`. Reassociation can still change floating-point branch decisions, hence the need for parity.

- **LM semantics are correct, with an execution-cost caveat.**  
  The uncapped body is AST-identical to `hfast.make_fused_lm`; initialization, stopping condition, and return values match. At [b2fast.py:135](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/b2fast.py:135), cap=1 retains at most one attempted iteration, including rejection. However, `body(s)` executes even when `go` is false; selection masks its results afterward. Thus an already-converged input can report zero iterations while still paying for a solve and another residual/Jacobian evaluation.

- **The head monkeypatch works on both XF paths and is restored.**  
  [b2speed.py:436](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/b2speed.py:436) patches the module lookup used during query construction. Both XF’s delegated path and its exact-step path construct their LM closures while patched. `finally` restores the original even on failure; later JIT tracing uses those captured closures.

- **No demonstrated long-job serialization/key failure.**  
  All three JSON configs match their generator exactly; all parity names exist; Python sources parse. FOM `__graphs` names are consistently used in quick-hash and invocation keys. Driver repetition counts correctly require 10/3 development and 4/1 held-out ROM/FOM invocations per case. Current recorded values appear compatible with `clean()` and strict JSON serialization.

- **Gf memory is material but not an established OOM.**  
  [b2speed.py:206](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/b2speed.py:206) caches each width separately: **2.054 GiB total at 512**, plus **0.996 GiB** for the rotated bank. Padding, derivatives, compiled executables, and graph buffers add to that. This deserves a peak-memory budget, but the persistent Gf arrays alone do not justify predicting an A100 OOM.

- **Minimum-of-two timing is disclosed and symmetric, but still selected timing.**  
  Both ROM and FOM receive both modes. That is fairer than optimizing only ROM, but taking minima retains selection optimism; the differing repetition counts also imply differing uncertainty. Report both mode medians and use the frozen-mode held-out measurement for confirmation.