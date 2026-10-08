**The J1 numerical results pass this audit.** I found **no numerical mismatch** in the 2D H1/H3 tables or plots. Several interpretations need qualification, especially the ST-specific “useful trim” and any claim that the flat ST median proves a time-error floor. No files were modified.

I recomputed directly from [result.json](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/runs/j1/archive/output/result.json), independently of `make_report.py`, against [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/DESIGN.md), and compared with [report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/report.md). All accuracy statements below remain **provisional and benchmark-relative**.

**1. H1 — CORRECT**

Recomputed errors at \(\kappa=4\), in percent; each cell is **worst / median**:

| \(R'\) | Deployed rule | Deployed ST | Deployed S | Converged ST | Converged S |
|---:|---|---:|---:|---:|---:|
| 128 | fib4181 | 4.636916 / 1.035481 | 3.448908 / 0.412136 | 4.636994 / 1.035475 | 3.451109 / 0.412131 |
| 256 | gauss96 | 3.058787 / 0.909683 | 1.494880 / 0.092386 | 3.058809 / 0.909687 | 1.494712 / 0.092346 |
| 384 | gauss128 | 2.746547 / 0.906332 | 1.049270 / 0.050798 | 2.746552 / 0.906333 | 1.045759 / 0.050795 |
| 512 | gauss192 | 2.837376 / 0.906660 | 1.599920 / 0.059497 | 2.837279 / 0.906660 | 1.599488 / 0.059514 |

Every entry matches the report after rounding.

Applying **A2-7 precedence** to the deployed \(512\)-versus-\(384\) comparison:

- Both deployed arms are available.
- ST worst increases **0.090829 pp**, exceeding the 0.05 pp unresolved band; relative increase **3.307%**.
- S worst increases **0.550650 pp**; relative increase **52.479%**.
- Neither meets the improvement bar. Both verdicts, and the combined verdict, are **“does not meet the registered bar.”**

The report’s paired ST statement also matches: **12/38 improve**. Its rounded “median improvement −0.000 pp” hides a small adverse change: median paired **increase 0.000371 pp**. A2-7 also requests relative-percentage changes, which the report omits. Those are presentation omissions, not verdict errors.

**2. H3 — CORRECT numerically; NEEDS-RESTATEMENT for broader accuracy claims**

I calculated each final-panel ratio as prescribed: median within each `(case, phase)` for each setting, divide trim by baseline, then take the median of the **12 paired ratios**. Setting-panel ratios use the separate panel medians.

| \(R'\) | \(\kappa\) | Δ worst ST, pp | Δ median ST, pp | Δ worst S, pp | Acceptable | Final paired ratio | Setting ratio | Useful |
|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 128 | 3 | +0.150890 | +0.003510 | +0.206549 | No | 0.912610 | 0.907931 | No |
| 128 | 2 | +1.004206 | +0.003991 | +1.301557 | No | 1.001119 | 0.971863 | No |
| 256 | 3 | +0.045930 | +0.003993 | +0.192424 | Yes | 1.242593 | 1.244151 | No |
| 256 | 2 | +0.420434 | +0.018862 | +0.875458 | No | 0.999855 | 0.992383 | No |
| 384 | 3 | +0.003784 | +0.000116 | +0.418449 | Yes | 0.752900 | 0.750253 | **Yes** |
| 384 | 2 | +0.517715 | +0.004490 | +1.499525 | No | 1.209940 | 1.201318 | No |
| 512 | 3 | +0.053592 | +0.000381 | +0.424525 | No | 0.780591 | 0.780302 | No |
| 512 | 2 | +0.523847 | +0.016284 | +1.167124 | No | 0.563788 | 0.563314 | No |

All match the report.

The useful \(384,\kappa=3\) trim reduces paired query time by **24.71%**, but worst S error rises **1.049270% → 1.467718%**. Therefore “useful under the registered ST criterion” is supported; “preserves accuracy” without naming ST is not.

Also, \(512,\kappa=3\) **fails** acceptability: its increase is 0.053592 pp, not within the 0.05 pp bar.

**3. Controls, gates and selection — CORRECT within the recorded scope**

- **Coverage:** 9,120 rows = 12 settings × 20 arms × 38 distinct cases. Per-time maxima and recorded eligibility agree with independent recomputation.
- **Controls:** all **24 control/setting combinations fail both criteria**, even at secondary \(\tau=10^{-3}\). Across controls, minimum worst distance is **0.146380**, and minimum \(\rho_{\max}\) is **1.428472**. Thus both the original stronger expectation and amended discrimination rule hold.
- **K-conv:** all settings pass; largest check distance **\(1.3054\times10^{-8}\)** versus **\(2.5\times10^{-5}\)**. Both converged/check rollouts are eligible on every case.
- **K-target:** all recorded check/flux comparisons pass; largest **\(9.6506\times10^{-8}\)** versus **\(10^{-5}\)**. Target norm minima are positive.
- **Floors/identifiability:** all floor consistency checks pass; largest discrepancy **\(7.86\times10^{-14}\)**. All sampled-bank, \(A\), and recorded reached-state Jacobian ranks equal \(R'\).
- **Selection:** independently reproduced both families’ primary/secondary \(m^\star\), distance-only and \(\rho\)-only choices, and deployed-family choices. No mismatch.
- **Cohorts:** every population contains exactly the expected **38 × 50** reached states. Selection uses dev6 and val32; timing uses dev6. The code constructs test64 descriptors for overlap checks but performs no test64 rollout or outcome-based selection.
- **K-eval:** the registered G1–G4 evaluator suite is explicitly **3D-scoped**; it should not be described as a J1 gate that passed.

I reran `audit_w.py` without an output argument: **1,824 scalar error comparisons passed**, maximum relative disagreement **\(4.38\times10^{-15}\)**; both injected faults were detected.

**Coverage qualification:** saved fields support every arm on the two audit cases and `gref` on all cases—not independent field-level reconstruction of every deployed case. Full-mesh distances and complete target evaluations were checked through their retained records, not rerun on a GPU. The audit’s `accepted` flag excludes A6; I separately checked those gates.

**4. Job integrity — CORRECT**

- Log explicitly contains **`jax_backend=gpu x64=True precision=highest`**.
- Job **5012763**, A100 80GB PCIe, host `pax007`; GPU UUID recorded.
- All **13 staged provenance entries** match their hashes and Git content at **`97add8fc5a12b6e3ffbb5fe92a91b55de1852367`**.
- Archived output checksums pass. Checkpoint hash is unchanged.
- Reference manifest and pinned files match; **76/76 references accepted**; cohort hashes agree.
- Log reaches `W2D COMPLETE` and `ALL-DONE`; stderr is empty.
- Recomputed timing drift spans **0.985975–1.023563**, within the registered interval. All **5,184 timed ROM invocations** record matching outputs; repetition coverage is complete. **`timing_valid_jobwide=True` is supported.**

**5. Interpretation challenges**

**“512 is worse than 384 against S” — CORRECT on this cohort; not a single-case artefact.**

Worst-case identities below are **zero-based val32 indices**, shown as **ST / S**. Deployed and converged identities agree in every cell.

| \(R'\) | \(\kappa=2\) | \(\kappa=3\) | \(\kappa=4\) |
|---:|---:|---:|---:|
| 128 | 19 / 19 | 19 / 19 | 19 / 5 |
| 256 | 19 / 19 | 19 / 5 | 19 / 19 |
| 384 | 19 / 5 | 19 / 5 | 19 / 5 |
| 512 | 5 / 5 | 19 / 5 | 19 / 5 |

At \(\kappa=4\), **the same S case, val32[5], is worst at both widths**. Removing it still leaves worst S error increasing **0.987630% → 1.212601%**. Moreover:

- **28/38 cases worsen**, 10 improve.
- **11/38 worsen by more than 0.05 pp**.
- Median paired S increase is **0.008026 pp**.
- Converged rollouts show the same pattern.

This supports degradation across multiple cases in this validation cohort, not a generalization claim beyond it. Meanwhile, the S projection floor improves **0.687% → 0.605%**: the worse rollout does not establish a worse representational span.

**“ST median is flat because of time error” — NEEDS-RESTATEMENT.**

From \(R'=256\) onward, ST medians stay near **0.91%**, while S medians are **0.092386%, 0.050798%, 0.059497%**. Independently, the S–ST reference separation has median **0.913799%**.

That strongly supports **a time-discretization-related plateau**. It does not isolate causality or supply an additive error decomposition; a matched time-step study is needed for that stronger claim.

**“\(m^\star\) depends on \(M\), or alternatively only on \(R'\)” — neither exclusive claim is supported.**

Primary Gauss \(m^\star\), in increasing \(\kappa=2,3,4\):

| \(R'\) | \(\kappa=2\) | \(\kappa=3\) | \(\kappa=4\) |
|---:|---:|---:|---:|
| 128 | 9,216 | 9,216 | 9,216 |
| 256 | 16,384 | 16,384 | 9,216 |
| 384 | 36,864 | 16,384 | 16,384 |
| 512 | 36,864 | 36,864 | 36,864 |

At fixed width, increasing \(M\) sometimes **reduces** the required rule size. At fixed \(M=1024\), \(R'=256\) needs **9,216**, while \(R'=512\) needs **36,864**. The observed dependence is joint and ladder-dependent. For all settings, primary Gauss \(m^\star=m_d\): rollout distance determines the selected size; the \(\rho\)-only requirement is smaller.

**6. Plots — CORRECT**

I inspected all four PNGs and traced their plotted quantities:

- Error plot matches deployed worst ST/S errors and S floors.
- Memory plot matches the registered byte formulas at selected rules.
- \(m^\star\)-versus-\(M\) plot matches the **Gauss-only** primary ladder.
- Cost plot matches **final-panel** medians. Its differences from the trim-grid’s **setting-panel** times are intentional.

The memory plot represents advection storage, not peak device memory or a demonstrated linear scaling law.

| Audit item | Verdict | Finding |
|---|---|---|
| H1 numbers and A2-7 verdicts | **CORRECT** | No numerical mismatch |
| H3 numbers and classifications | **CORRECT** | Only \(384,\kappa=3\) is useful |
| Controls, applicable gates, selection | **CORRECT** | All pass within stated audit coverage |
| Job provenance and timing validity | **CORRECT** | Hashes, logs and invocation records agree |
| S degradation attributed to one case | **WRONG** | Persists after removing the shared worst case |
| Useful trim described as accuracy-preserving generally | **NEEDS-RESTATEMENT** | Acceptance is ST-specific |
| Time error asserted as proven plateau cause | **NEEDS-RESTATEMENT** | Strong supporting evidence, not causal isolation |
| \(m^\star\) attributed exclusively to \(M\) or \(R'\) | **NEEDS-RESTATEMENT** | Joint dependence; distance criterion controls selection |
| Four 2D plots | **CORRECT** | Match their specified quantities |

Statements safe to report:

- On these 38 development/validation cases at \(1024^2\), the \(512,2048\) setting fails H1 against both first-order references.
- Its S degradation relative to \(384,1536\) affects multiple cases and persists without the shared worst case.
- The \(384,\kappa=3\) trim is useful under the registered ST criterion, with a 24.71% paired time reduction and a 0.418449 pp increase in worst S error.
- The nearly flat ST median is consistent with time-discretization error masking smaller spatial-error differences.
- Selected quadrature size depends on the joint setting and tested ladder; reducing \(M\) does not reliably reduce \(m^\star\).
- All applicable recorded J1 gates pass, and the report’s 2D numerical tables and plots match the retained data.