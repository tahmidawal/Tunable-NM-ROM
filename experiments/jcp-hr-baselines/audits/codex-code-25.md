**NOT READY for (B).** Audited `c7ebe67f9`: the checkout is now `93010a6e2`, but the scoped files and DESIGN are unchanged between those revisions. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **CORRECT — nonblocking — round-24 item 3: all members missing.**  
   `analyze.py:107–108` obtains uncertainty independently from `B['eps_ref']`; `195–200,224–229` handles absent members before selection. An in-memory probe returned **“role incomplete (missing)”**, zero assessed pairs, and no exception.  
   **Fix:** retain these checks.

2. **CORRECT — nonblocking — round-24 item 4: unfinished tuning inventory.**  
   `analyze.py:110–113,141–153` reconciles the tuning inventory independently of frozen membership and distinguishes unfinished, construction-failed and deferred configurations. A probe with an unfinished registered Q-DEIM configuration propagated the tuning-coverage restriction.  
   **Fix:** retain this reconciliation; address the conclusion-formatting escape below.

3. **WRONG — blocking — round-24 item 6: the pair inventory bypasses withholding and qualification.**  
   Roles and cell lines use `qualify`, but `analyze.py:255,268–269` writes raw substantive labels into `pairs`. With inadequate basis evidence, a probe produced a correctly withheld role alongside **`"B_cost: neither dominates"`** in `pairs`. These conclusions also bypass diagnostic-only batch wording. A35.1 requires restrictions on every emitted conclusion.  
   **Fix:** apply the shared formatter to pair outcomes, or store structured outcomes with explicit qualification, restrictions and withheld status. Do not expose raw labels as unrestricted conclusions.

4. **WRONG — blocking — round-24 item 7: interrupted accounting still lacks its promised upper bound.**  
   Comparator qualification, construction totals and per-pick construction information are now included (`analyze.py:67–88,250–252`). However, `accounting()` never reads the ledger, identifies the relevant allocation, or exports an upper bound. A truncated-job probe returned a V lower bound and active phase only. This falls short of A35.2.  
   **Fix:** bind each attempt to its ledger record and export measured allocation bounds; explicitly mark unavailable bounds unknown. Preserve the existing T/V phase classification.

5. **CORRECT — nonblocking — round-24 item 8: completed anchor/comparator evidence.**  
   `stageb.py:257–263` checkpoints each anchor immediately. Comparator arrays precede their metadata checkpoint (`204–207`), and active phases are saved before execution (`105–108`). Timing attempts remain checkpointed individually (`75–78`).  
   **Fix:** retain this ordering.

6. **NEEDS-RESTATEMENT — nonblocking under the prior A12.4 treatment — round-24 item 9: ECSW validator improved, but is not equivalent to the complete validator.**  
   Terminal node checks, residual-norm reproduction, checkpoint binding, optimum aliasing and artifact hashing are present (`freeze_stageb.py:90–108,169–170`). However, its KKT check (`97–100`) omits finite-value and nonnegative-tolerance checks present in `smoke_accept.py:118–119`. A synthetic certificate with both gradients and tolerance equal to infinity passed with `ok=True`.  
   **Fix:** reuse the complete KKT predicate, including finiteness and tolerance validation, before describing this as the full validator.

7. **WRONG — blocking — round-24 item 10: memory admission still does not bound the executed path.**  
   A35.3’s “one comparator object” rule is violated during qualification: `stageb.py:184–196` caches all distinct 3D comparator objects; the 2D path builds all admitted settings together (`180`). Moreover, `cell.set_basis` retains full `Phi`, `LPhi` and `DPhi` (`stageb.py:136`; `pipeline.py:114–118`), whereas the model uses full-basis width plus three *membership-maximum* prefixes (`freeze_stageb.py:193–199`). It omits comparator storage, retained reference fields and full-cohort working records. `peak_tune` is recorded but does not affect acceptance.  
   Staging accepts the fixed `memory_ok` flag without comparing it with `--gpu` or `--mem` (`cluster/stage.py:49–50,143–145`).  
   **Fix:** release comparators between qualification runs, model actual simultaneous allocations and workspaces, and enforce admission against the requested GPU and host memory.

8. **NEEDS-RESTATEMENT — blocking for resource certification — remaining round-24 item 10 time claim.**  
   The projection now includes setup/checkpoint allowances and staging compares against wall minus four minutes. Nevertheless, `freeze_stageb.py:184–197` uses fixed multipliers and allowances, calling the doubled result `hours_certified` without establishing a bound on compilation, certification or retry workload.  
   **Fix:** label it an estimate until supported by workload-specific measurements/bounds; include the omitted workload explicitly. Keep Slurm wall limits and ledger reservations as the separate hard GPU-hour controls.

9. **CORRECT — nonblocking — round-24 item 11: unique counting and missing-pair inventory.**  
   `analyze.py:174–175,224–234,239–271` supplies pair inventories on incomplete branches and counts unique IDs only after a substantive label. A single configuration filling both roles counted as **one** assessed pair; all-missing evidence counted as zero.  
   **Fix:** retain counting logic while repairing item 3’s outcome formatting.

10. **WRONG — blocking — additional failure path: failed replay silently reduces the candidate set.**  
    `analyze.py:114–115` exempts `replay_failed` from missing-member restrictions; `204–206` excludes it without marking roles incomplete. A two-member probe with one replay failure still filled a role and emitted **“no observed domination among the 1 pairs assessed”** without a failure restriction. The frozen candidate set was not successfully reproduced, contrary to the blocking replay gate in A32.6.  
    **Fix:** treat failed replay as blocking/incomplete evidence for the affected candidate set; withhold favourable or neutral conclusions and retain the failure reason.

11. **WRONG — blocking — additional invalid-evidence path: nonfinite scores reach selection.**  
    `analyze.py:217–222,235` passes eligible-case errors into selection without validating completeness and finiteness. Although `pair_label` subsequently rejects invalid inputs, `274–277` can still emit a favourable absence statement. A NaN-error probe produced **“no observed domination among the 0 pairs assessed”** without withholding. Missing error keys instead raise before a verdict is written.  
    **Fix:** validate every candidate’s full eligible-case score inventory before selection; turn missing/nonfinite evidence into an explicit incomplete restriction and withhold absence-of-domination conclusions.

12. **CORRECT — nonblocking — retained timing, batch, spending and cohort safeguards.**  
    Retry exhaustion remains checked (`analyze.py:49–60`); timing is X-outer with the registered member order (`stageb.py:308–329`); role/cell formatting preserves batch qualification (`analyze.py:158–172`). Stage-B staging retains atomic spending reservations (`cluster/stage.py:159–162`). Cohorts remain `dev6`/`val32` or `val64` (`stageb.py:139–149`).  
    **Fix:** retain these safeguards; extend qualification consistently to the pair inventory.

All probes were CPU-only and in memory, with writes intercepted. No files, scratch, jobs, GPU work, or `tests/test_core.py` were used.