**Overall: NEEDS-RESTATEMENT.** A2 fixes the main 3D claim, but budget/defer semantics and exceptional GAL evidence remain incomplete.

Read-only: no files changed, scratch created, arrays copied, or jobs submitted.

References: `D` = `experiments/jcp-qualify/DESIGN.md`; `Brief` / `Common` = campaign briefs; `P1` / `Combined` = the respective lane experiment directories.

| Previous blocker | A2 disposition | Judgment and evidence |
|---|---|---|
| 1. False complete 3D front | **RESOLVED** | **CORRECT.** Always labels a tested subset; computes its front using Q1/Q2-passed candidates only. Open candidates are checked against **every** passed candidate, avoiding the previous transitivity assumption. `D:474–480`. |
| 2. Global barrier / core reservation | **PARTIALLY** | **NEEDS-RESTATEMENT.** Mesh-local readiness and deferred terminal states fix the global barrier. Outstanding core reservation is added, but its accounting and optional-work classification remain ambiguous; see blocker 1 below. `D:484–490`. |
| 3. Missing 3D successors | **RESOLVED** | **CORRECT.** Missing successor/check rules explicitly produce “unverified”; subset wording also survives early budget deferral. `D:474–482`. |
| 4. Positive controls / TH06 coverage | **PARTIALLY** | **NEEDS-RESTATEMENT.** Designated GAL fixture replaces “every arm must pass”; full TH06 is scheduled before B4 interpretation. However, the cited evidence does not establish every claimed control predicate, and postponement conflicts with the governing smoke gate. `D:492–499`; `Common:18–22`. |
| 5. Timing memory | **RESOLVED** | **CORRECT.** Pilot now includes both banks, main tables, sequential certification tables and R512 fallback tables; failure explicitly means core incomplete. This is a design resolution, not demonstrated feasibility. `D:501–505`. |
| 6. GAL mathematics / exceptional evidence | **PARTIALLY** | **NEEDS-RESTATEMENT.** Singular-Jacobian correction and boundary fixtures are correct. Zero-scale solver-side raw-residual recovery remains unspecified. `D:507–515`; `P1/p1core.py:192–218`. |

**Ranked remaining blockers**

1. **NEEDS-RESTATEMENT — reservation and defer order are not yet executable unambiguously.**  
   A2 calls all `$256²$` and “larger steps” optional, although the prescribed defer list names only **256² larger steps**, not its entire panel or 1024² larger steps. Clarify whether this broadens permissible deferrals; otherwise preserve the exact list. Also exclude the proposed core job from “not-yet-submitted core jobs,” or its time limit is counted twice. Define reservation for *possible* core fallbacks before their need is known. `D:487–490`; `D:357–359`; `Brief:43–44`.

   Mesh-local timing still waits for optional accuracy/fallback work **within that mesh**. This is consistent with a single final panel, but is not immediate core-only readiness. State that limitation explicitly. `D:485–487`.

2. **NEEDS-RESTATEMENT — zero-scale G-cert lacks its independent comparator.**  
   A2 records the **certifier’s** raw residual, but never specifies how the **solver’s** raw residual is stored/recovered. The unchanged P1 solver stores `rn/(tol+1e-300)`, not `rn`; with zero scale this can overflow. Comparing the certifier’s raw norm with itself would be vacuous. Specify solver-side storage or guarded recovery, including overflow behavior. `D:54–56,510–514`; `P1/p1core.py:193,218`.

3. **NEEDS-RESTATEMENT — positive GAL evidence and failure attribution overclaim.**  
   The cited Combined table’s `38/38` entries count **unresolved error-budget components**, not verified solver runs. Small production/tight trajectory distances alone do not establish per-step residual convergence, G-cert agreement, or valid exits. Cite the actual convergence evidence and matching configuration. A fixture failure blocks GAL qualification; it does not uniquely prove “the certifier is wrong”—driver, solver or provenance failures remain possible. `D:495–497`; `Combined/report.md:273–284`.

4. **NEEDS-RESTATEMENT — full TH06 control moves past a binding gate.**  
   Running six cases in `order` repairs coverage, but Common requires every control demonstrated in the real-size smoke before subsequent production submission. Explicitly amend that sequencing contract, or place the six-case control in smoke. `D:498–499`; `Common:18–22`.

**Additional undefined case:** B4 takes `log e` without defining behavior for zero error or a nonpositive error difference. Register “not interpretable” for undefined logs/slopes and add a zero-error fixture. `D:247–256,432–434`.

No remaining 3D completeness counterexample defeats A2’s deliberately restricted label. The blockers are now chiefly execution-contract and evidence-definition issues.