Read-only audit completed. No files changed, jobs submitted, or lab-log entries written. Verification used source inspection, CPU bookkeeping, and staged-file hashes; no GPU solves.

1. **CORRECT — round 2 item 7: certificates regenerated.** Both published certificates exactly match recomputation from the current `resources.py` and smoke outputs:

   | Job | Certificate GPU-h | Required limit, ×1.5 | Staged limit |
   |---|---:|---:|---:|
   | ref100 | 0.4186862735 | 0.6280294102 h | 0.75 h |
   | ref10 | 0.1377292775 | 0.2065939163 h | 0.50 h |

2. **CORRECT — round 2 item 8: qualification-count bound fixed.** [resources.py:195](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/resources.py:195) now sums each group’s most expensive permitted attempts without deduplication. This covers the previous screening-order counterexample. Spectral qualification compilation charges remain separate. This validates the accounting correction, conditional on the smoke-derived cost model.

3. **CORRECT, conditional — resume evidence and equivalence.** The resume path imports the adjacent `stages.npz`, preserving reused L1 evidence when that archive is present. A9.1 correctly withdraws identical-result claims because fresh screening timings can change rankings. **Operational qualification:** archive loading is conditional on existence; L2 preparation must verify that the archive accompanies `result.json`. The code does not reject a missing archive.

4. **CORRECT — budget completeness repaired as a conditional plan.** A9.3 bounds pooled fallback at seven cases, adds a separate two-case test-fallback reserve, and requires regeneration/deferment when certificates exceed reserves. Arithmetic checks:
   - Spent plus remaining: **7.2520155510 GPU-h**.
   - Headroom below eight: **0.7479844490 GPU-h**.

   **NEEDS-RESTATEMENT:** “fits” means the provisional allocations fit. L2, timing, and test execution still need their promised certificates; completion of the entire phase within eight hours is not yet demonstrated.

5. **CORRECT — staged job configuration and execution safeguards.** Both jobs request one H200, `gpu` partition, 64G host memory, and sufficient time under the certificate rule. They use the absolute cluster interpreter, enforce GPU backend and f64/highest precision, check the input manifest, and write logs, caches, temporary files, and outputs beneath their distinct `/cluster/tufts/paralab/tawal01/imp-varcoef/<job>` directories.

   **NEEDS-RESTATEMENT:** 64G is a requested host-memory allocation, not measured production-memory validation. The earlier ladder-memory qualification remains open.

6. **CORRECT — staged provenance.** For **each job**, all **156 provenance entries** match their staged bytes, recorded sizes/hashes, and Git blobs at `4bde13457c82c6dddb4211649fe8f12e1e926b9a`. Every manifest entry verifies. Configurations and regenerated certificates are included. This verifies local staging; remote transfer remains a submission-time check.

7. **CORRECT — production reference populations and levels.** Each configuration contains exactly eight development and 24 validation cases, with no test cases. Both use A7.3’s `(512, 0.005)`, `(1024, 0.0025)`, `(2048, 0.00125)` levels, cross-check development cases 0–3, and enable reference controls. Cases are regenerated from recorded seeds.

8. **CORRECT — reference acceptance and eligibility evidence.** `vref.py` applies E0 acceptance and the `ε_ref ≤ 10⁻³` threshold. Its seven-run cross-check implements A2.5 as subsequently amended by A4.5/A5.5, including adjacent spatial/temporal differences, validity, quadrature and tolerance checks. `elig.py` rejects missing, stale, or failed required cross-checks, checks the exact pooled population, and applies the 24-of-32 coverage threshold.

9. **NEEDS-RESTATEMENT — what these outputs answer.** These jobs supply reference fields and evidence needed to freeze the eligibility mask **after permitted fallback attempts and before ladder scoring**. They do not themselves freeze that mask or answer whether the fastest accuracy-matched FOM costs at least twice the ROM proxy. That requires the later qualified ladder and timing results. Reference estimates remain conditional; conclusions apply to the eligible population.

**NOTED — forgery-only:** deliberate substitution of checkpoint or provenance contents is outside the blocking findings.

**Overall verdict: CLEAN — no WRONG findings; qualifications above remain binding.**