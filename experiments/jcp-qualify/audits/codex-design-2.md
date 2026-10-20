**Overall verdict: NEEDS-RESTATEMENT.** A1 closes several original issues, but the 3D completeness claim remains wrong and the execution sequence can obstruct the never-dropped core.

Read-only: no files changed, scratch created, arrays copied, or jobs submitted.

Citation shorthand: `D` = `experiments/jcp-qualify/DESIGN.md`; `Brief` = campaign `briefs/jcp-qualify.md`; `P1` / `Speed3D` = their respective experiment directories.

Original audit items:

| Item | A1 disposition | Judgment and evidence |
|---|---|---|
| 1. Matrix/reuse | RESOLVED—retained | **CORRECT.** Expansion and pooled qualification remain explicit. `D:77–84`. |
| 2. Larger-step selection | RESOLVED | **CORRECT.** Rounding and conditional-on-history wording corrected. `D:465–466`. |
| 3. LSPG Q1 | RESOLVED—retained | **CORRECT.** Unchanged inherited predicates. `D:186–200`. |
| 4. GAL certificate | PARTIALLY | **NEEDS-RESTATEMENT.** Replacement threshold and zero denominator addressed; new “equals 0 only at an exact zero residual” claim is false for singular Jacobians. `D:413–424`. |
| 5. GAL G-cert | PARTIALLY | **NEEDS-RESTATEMENT.** Ordinary-scale reconstruction remains correct. Zero-scale raw-residual comparison lacks its recording/recovery and agreement-tolerance specification. `D:212–213,420–421`; `P1/p1core.py:218`. |
| 6. Q2 predicates | RESOLVED—retained | **CORRECT** for the original configurations. Expanded 3D rounds introduce a missing-successor case below. `D:143–155,217–224`. |
| 7. Fallback timing | RESOLVED | **CORRECT** on same-job pairing. Global sequencing introduces a separate blocker. `D:406–411`. |
| 8. Initial 3D screening list | RESOLVED—retained | **CORRECT.** Initial list unchanged. `D:123–145,399`. |
| 9. 3D negative fixture | RESOLVED—retained | **CORRECT.** Same-comparator triangle argument retained. `D:319`. |
| 10. Qualified 3D front | PARTIALLY | **WRONG.** Uncertified admission removed, but “no untested exposed member” does not establish completeness under non-transitive dominance. `D:396–404`. |
| 11. B1 populations | RESOLVED | **CORRECT.** Explicit R512 mesh tests and consistent 38-case transitions. `D:440–445`. |
| 12. B4 interpretation | RESOLVED | **CORRECT.** Finite-range limitation stated beside the label. `D:435–438`. |
| 13. B4 controls/buffers | PARTIALLY | **NEEDS-RESTATEMENT.** Non-vacuous classifier checks and buffer sizing added; the smoke inventory still cannot demonstrate the required five-of-six control. `D:340,429–436`. |
| 14. Primary timing | RESOLVED—retained | **CORRECT.** Qualification and identity requirements retained. `D:266–279`. |
| 15. Dominance implementation | RESOLVED | **CORRECT.** Exhaustive pairwise comparison and separate fixtures specified. This does not repair A1.1’s stopping rule. `D:458–463`. |
| 16. Replayable handoff | RESOLVED | **CORRECT.** Mode assets, rule hashes, reference identity, entry points, identity evidence and freezing commit added. `D:448–453`. |
| 17. Control evidence/blocking | PARTIALLY | **NEEDS-RESTATEMENT.** Motion establishment and B5-3D propagation fixed; requiring every smoke GAL configuration to pass confuses scientific qualification failures with certifier failures. `D:422–426,467`. |
| 18. Resources/memory | PARTIALLY | **NEEDS-RESTATEMENT.** Sum and explicit core-incomplete outcome fixed. Mitigation only names `acc4096`, although the failing pilot includes timing’s reference bank. `D:341,454–456,467–468`. |

Original ranked blockers: **1 PARTIALLY; 2 RESOLVED; 3 PARTIALLY; 4 PARTIALLY; 5 RESOLVED; 6 PARTIALLY.**

Ranked blockers remaining:

1. **WRONG — A1.1 can falsely declare the complete qualified front.**  
   Under `D:281–284`, take zero reference uncertainty and synthetic `(cost,error)` points:
   `A=(1,1)`, `B=(1.04,0.96)`, `C=(1.0816,0.9216)`. Then **B dominates A; C dominates B; C does not dominate A**. Only C is initially exposed. If C passes Q2, A1.1 stops without testing B or A. But if B fails and A passes, the qualified front is `{A,C}`.  
   **Fix:** establish enough qualification statuses to prove completeness, or always use a tested-subset label. Recompute that subset’s front using only qualified candidates. `D:399–404,458–463`.

2. **NEEDS-RESTATEMENT — global accuracy barrier threatens core timing and handoff.**  
   A1.2 explicitly waits for **all** accuracy and fallback jobs before staging **any** timing job. Thus optional GAL/larger-step work and unrelated meshes gate completed core panels. The ledger reserves queued/running jobs, not outstanding mandatory timing; optional work can consume the remaining budget first. Deferred jobs also have no explicit terminal state satisfying “all … complete.”  
   **Fix:** mesh-local readiness, explicit deferred states, and budget reservation for remaining core accuracy/fallback/timing before optional submissions. Preserve the prescribed defer order. `D:350–359,407–410`; `Brief:39–44`.

3. **NEEDS-RESTATEMENT — expanded 3D universe contains an undefined Q2 ladder.**  
   `W1024|lat65536|fs|BE|0.01|K1` is eligible with a valid screening panel (`Speed3D/checks/analysis-grid.json:133704`). Its successor is `lat131072`, but its required second successor is beyond the registered ladder. A1.1 can expose this candidate while promising the “same predicate.”  
   **Fix:** register the additional rule and memory/cost pilot, or explicitly mark missing-successor candidates unverified and the result incomplete. Also define subset labels for budget deferral before round three. `D:144–152,398–403`.

4. **NEEDS-RESTATEMENT — positive-control requirements conflict with legitimate outcomes and smoke coverage.**  
   “Every GAL production configuration of the smoke must pass” prevents an honest configuration-level failure from being treated as an experimental result; no independent evidence establishes that all those configurations should pass. Separately, the required TH06 control needs ≥5 interpretable cases, while `smkA` specifies a one-case B4 mini.  
   **Fix:** independently establish a designated positive GAL fixture; permit other arms to fail qualification. Schedule the full TH06 control before reading B4 verdicts. `D:312–313,340,422–434`.

5. **NEEDS-RESTATEMENT — memory mitigation does not cover the required timing panel.**  
   Splitting `acc4096` does not demonstrate that `time4096` fits with A’s bank and the required paired subjects. Running timing subjects in separate jobs would violate A1.2.  
   **Fix:** specify and pilot sequential subject/role lifetimes within the single timing job, including fallback rules; explicitly report core incomplete if this fails. `D:266–279,341,454–456`.

6. **WRONG / NEEDS-RESTATEMENT — GAL exceptional mathematics and evidence remain incomplete.**  
   With singular `J=diag(1,0)` and nonzero residual `r=(0,1)`, `Jᵀr=0`, so γ is zero without a root. Singular steps are explicitly contemplated by Q3. Qualify the claim with nonsingularity. For zero-scale G-cert, specify raw-residual storage/recovery and comparison tolerances; boundary fixtures should include exact threshold equality and zero-denominator cases. `D:210–215,417–424`.

**A1 is not ready for implementation without another restatement.**