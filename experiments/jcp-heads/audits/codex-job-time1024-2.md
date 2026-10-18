**time1024: SUBMIT for acquisition. Results acceptance remains blocked by analysis defects.**

The original evidence-loss blocker is fixed in staged commit `578e58129022ddd1e64c8db18aa69a1c560cbb56`. No files changed, generators executed, jobs submitted, or lab-log entries written.

| Round-1 finding | Reassessment | Current state |
|---|---|---|
| 1. Accuracy coverage | **CORRECT** | 54 arm/scheme records; rules and successors match `RULES.json`. |
| 2. Timing coverage | **CORRECT** | 42 subjects × (6×3 + 32×1) = **2,100 split/fused pairs**. Reference A and doubled-work target remain correct. |
| 3. Staging and sbatch | **CORRECT** | Both manifests pass. All 138 provenance-listed files in each staging directory match their respective committed blobs. time1024 now pins `578e58129`. |
| 4. Resources | **CORRECT**, retaining the restatement | **22 arm objects**, not 42. Eight-burn correction reproduces **1.643384 h** projected runtime; **2.5 h** scheduler limit. This remains an estimate, not a bound. |
| 5. Analysis incomplete | **CORRECT** | Several repairs landed, but material defects remain below. |
| 6. Timing evidence discarded | **CORRECT historically; NEEDS-RESTATEMENT now** | All-case scoring trajectories and subject identity hashes are now persisted. The submission blocker is resolved. |

The restaged configuration matches the accuracy arms’ kinds, heads/ranks, rules, successors and schemes. The batch script uses H200 on `gpu`, the absolute cluster venv, GPU preflight, x64/highest precision, a dedicated paralab directory, input checksums and output checksums.

[htime.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/htime.py:127) now saves `T_case_ids`, `T_W`, `T_w0`, `T_n0` and `T_ident` for all 38 cases, including reference A. Split attempts retain B/A1/A2 hashes; fused attempts retain B hashes. Thus the measured subject trajectories can be linked to recoverable scoring trajectories. Fused A1/A2 hashes remain represented only by the combined boolean—a residual auditability limitation, but not the original subject-trajectory blocker.

**Required before accepting results:**

1. **Repair timing qualification.** [The new predicate](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/analyze.py:156) omits the Q2 **rho ≤ 0.116** gate. It trusts JSON certification booleans/distances without independently recomputing the saved-array checks. Require the complete dev6 population, independent distance/rho verification, and separate Q1/Q2 outcomes, including “unverified.”

2. **Make consistency establish what it claims.** [The consistency check](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/analyze.py:161) correctly decodes `T_W` and scores it against E0, but compares only scalar errors. Equal error does **not** establish equal trajectories or transfer solver qualification. Compare decoded timing and accuracy trajectories directly; retain timing-job errors for joint accuracy–cost results. Also:
   - require exact case coverage and matching configuration/decoder identities;
   - validate saved metadata rather than supplying hard-coded scoring metadata;
   - reject missing accuracy records and nonfinite values;
   - verify attempt hashes against `T_ident`, rather than trusting `valid`;
   - retain per-case scores and differences. Currently missing consistency can still yield `usable=True`.

3. **Fix control propagation.** [Doubled-work acceptance](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/analyze.py:147) passes vacuously if every repetition fails drift, or none exists. Require expected coverage and valid evidence. The split-panel validity result is merely exported; enforce the applicable identity, finite-positive-ratio and drift gates. [H-cost-N](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/analyze.py:327) also bypasses timing qualification/equivalence/doubled-work gates by consuming `raw_ev` directly.

4. **Finish per-seed comparisons correctly.** Pooled CN-R finalization is fixed, with F5 still BE-only. However, [finish_lh](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/analyze.py:315) still requires `len(S_prime) >= 2`, so every single-seed Q-LH-cost result becomes unavailable. Use the single-seed criterion for that sensitivity. Per-seed H-acc-K and Q-LH outputs also remain BE-only; complete the registered CN-R sensitivities.

5. **Finish provenance enforcement/export.** The new E0 store pin matches the current store, and ST reference hashes/version now survive export. However, the manifest’s metric/checkpoint/rotation fields are not enforced by that pinning code; metric metadata and corresponding S-reference provenance remain omitted from per-case exports.

The per-head Jacobian gate is fixed. The remaining defects can be repaired against the evidence this job will preserve; they do not justify withholding acquisition.

**Final disposition: SUBMIT time1024; do not accept scientific results until these analysis repairs pass review.**