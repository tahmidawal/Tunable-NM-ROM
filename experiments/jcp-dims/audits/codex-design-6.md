**Overall: NEEDS-RESTATEMENT.** A6’s adjustment is defensible but its supporting numbers need correction. A8 leaves a deferral ambiguity and an implementation gap.

| Amendment | Verdict | Finding |
|---|---|---|
| **A5** | **CORRECT** | Smolyak’s removal as a must-fail gate already occurred in A1.2, before round-5 acceptance. Production release is supported by the coordinator’s recorded decision. Neither introduces a new scientific verdict rule. |
| **A6** | **NEEDS-RESTATEMENT** | The floor is explicitly post-smoke and numerically reasonable, but the quoted smoke maxima and margin are inaccurate. A6.2 openly reduces independent verification coverage; this is a weakened verification requirement, although the production Q2 bars remain unchanged. |
| **A7** | **CORRECT** | Revising runtime estimates after observed execution costs does not change accuracy or qualification predicates. The cancellation and charged hours are disclosed; estimates remain distinct from scheduler bounds. |
| **A8** | **NEEDS-RESTATEMENT** | Splitting and retrying accuracy work are sound in principle. The deferral language must preserve A1.6’s non-deferrable items, and the analysis loader must recognize the split jobs. |

**A6.1: justified floor, incorrect evidence summary.** Recomputing from the archived `smk2` GAL and square-LSPG arrays using their stored tolerances gives:

- Maximum absolute discrepancy: **1.2564×10⁻¹³**, exceeding the stated **7×10⁻¹⁴**.
- Maximum scale-normalized discrepancy: **5.1947×10⁻¹⁵**, exceeding **4.3×10⁻¹⁵**.
- **413 of 1,100 steps** fail the old comparison; none fail the amended comparison.
- The scale-floor margin is therefore approximately **19×**, rather than 23×.

These magnitudes support a round-off explanation, although they do not independently prove evaluation order as its cause. Keep the explicit post-smoke label, correct the numbers, and describe the causal explanation as consistent with the evidence. This **does relax a registered G-cert predicate after smoke data**; it does not alter the residual-certification or Q2 thresholds. See [A6.1]( /home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/DESIGN.md:709).

**Retry merging is conditionally safe.** Whole-arm accuracy archives can be combined across attempts with unchanged numerical code, configuration, inputs and H200 execution. This does not authorize combining cross-job wall-clock measurements. The staged `f2b`/`f2br1` differences leave the `dacc` execution path unchanged, consistent with the earlier job audit. Preserve complete case/anchor coverage, provenance and failed outcomes; choose duplicates by the declared attempt order, never by accuracy. Future retries still require these checks—“H200” alone is insufficient.

**A8 needs two concrete corrections:**

1. **Preserve the allowed deferral set.** A8.4’s “non-core … otherwise deferred” is broader than [A1.6](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/DESIGN.md:548), which explicitly makes κ=3, `b3f`, the fixed-(M,m) panel and `t2` non-deferrable despite being non-core. Splitting κ=3 and κ=8 into `f2kB` must not permit dropping both. Prioritizing core work is consistent with the brief; treating every remaining item as deferrable is not.
2. **Recognize the split archives.** I verified that `f2kA ∪ f2kB` exactly reproduces the original settings. However, [analyze2d.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:104) discovers `f2k` and numbered retries only, excluding `f2kA` and `f2kB`. The current implementation cannot deliver the promised merged inventory.

No files changed, jobs submitted, or lab-log writes performed.