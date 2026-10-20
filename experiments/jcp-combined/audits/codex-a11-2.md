**Overall: WRONG — important fixes landed, but extension acceptance and smoke certification still have defects. Job gate: SUBMIT the H200 smoke; HOLD production.**

**(a) `select3d.py` — WRONG overall; reproduction and status checks substantially fixed.**

Correct now:

- All four gl80 arms are required, with ordered case IDs and matching `(64, 6, 1024)` production shapes.
- Production requires cohort 923801 and cases 0–63.
- Bank, rank, startup and numerical settings are compared.
- A10 pair statuses are recomputed from trajectories and row validity; disagreement prevents extension acceptance.
- Accepted smoke inputs propagate `smoke_inputs=true`; mixing smoke and production archives is rejected.

**Blocking defect:** [select3d.py:151](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/select3d.py:151) compares `refined_ref` strings literally. [Staging:105](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/cluster/stage.py:105) produces different paths:

```text
…/c3descW1024/refs/ref_923801.npz
…/c3extW1024/refs/ref_923801.npz
```

Thus a correctly staged extension fails the contract assertion even with identical reference contents. Bind reference identity by its recorded SHA256 and metadata, rather than its job-specific path.

Two remaining hardening gaps: the extension branch does not explicitly restrict acceptance to W1024, and only gl80 receives the explicit shape check; successor arrays can undergo NumPy broadcasting.

**Synthetic test — CORRECT as limited machinery evidence.** I verified both recorded input hashes, exact gl80 equality for all four arms, and exact `gl96 = gl80*(1+1e-9)` arrays. Its output records successful reproduction, matching recomputed/job statuses, convergence at gl80, and downstream selection of `lat65536`. It exercised the acceptance path. However, it copied the original reference path, masking the real staging defect; it did not exercise gl96 selection through the second extension pair.

**(b) `inventory.py::smoke_gates` — WRONG.**

The timing-panel exemption works, but the claimed completeness checks do not. At [inventory.py:248](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:248):

- Empty `pairs` passes through vacuous `all(...)`.
- Rows are checked for validity, without requiring every case/rule/arm combination.
- Rule-list prefix acceptance does not verify that skipping the final rule was justified.

Executing the actual extracted function on the local extension result still returned **True** after deleting every pair, and separately after retaining only one row. Require the complete expected inventory, required pair/arm records, and justified conditional omission.

**(c) `make_budget_configs.py::FINER` — CORRECT.**

Both required successors are present: `gl80 → gl96` and `gl96 → gl112`. Round-1 item 12 is fixed.

**(d) Restated A11 and A12 — NEEDS-RESTATEMENT.**

The round-1 wording corrections are sound: approximately 15× rather than orders of magnitude; W768’s unresolved original axis; conditional escalation after fail **or unresolved**; reapplying the selection procedure; and limiting the local run to control-flow evidence.

However, A11’s statement that the smoke machinery gate requires every planned run and every finite pair overstates the implementation identified in **(b)**. Its contract-binding description also needs the reference-identity correction in **(a)**. A12 accurately describes the implemented relative formula.

**(e) A12 relative B0 negative control — CORRECT for the pinned fixtures, with a qualification.**

Ignoring the small unmodified closure defect, the relative mismatch is

\[
99\left(\frac{e_P}{e_\star}\right)^2.
\]

Recomputing from the saved fixture terms gives:

| Fixture | Absolute defect | Relative defect |
|---|---:|---:|
| 2D pinned ST | \(5.2450\times10^{-7}\) | \(1.377219\) |
| 3D pinned | \(1.5711\times10^{-4}\) | \(0.678144\) |

This genuinely introduces inconsistent normalization. The \(10^{-6}\) detection threshold is sensible: four orders above the \(10^{-10}\) B0 tolerance, and comfortably below both observed defects.

It does **not** guarantee failure for arbitrary data: if \(e_P=0\), the mutation changes nothing; sufficiently small \(e_P/e_\star\) is also non-discriminating. Qualification requires approximately \(e_P/e_\star>1.005\times10^{-4}\). Both pinned fixtures satisfy this.

The saved `bud3-smk.json` still reports the old **absolute** defect; its stored terms support the relative value above.

**(f) JOB GATE — smoke CORRECT / SUBMIT; production configuration CORRECT / HOLD.**

- **`smk3ext.json`: SUBMIT on H200.** W1024, cases `[0,1]`, all four arms, and unconditional gl96/gl112 correctly test the largest rule’s memory and rate. No timing panel is required. Staging inputs match HEAD; manufactured-test source hashes and the reference checksum pass. Capacity remains unverified until this smoke completes.
- **`c3extW1024.json`: correctly specified, but HOLD production.** It covers all 64 validation cases, runs gl80/gl96, and runs gl112 unless the first pair passes A10. Before production, fix **(a)** and **(b)** and obtain the H200 resource certificate from complete smoke evidence.

Operationally, the smoke is not yet staged, and the current untracked `audits/codex-a11-2.log` trips staging’s cleanliness assertion. The SUBMIT verdict approves the smoke configuration; ordinary staging must resolve that guard first.

No files were created, modified, or appended by my audit actions.