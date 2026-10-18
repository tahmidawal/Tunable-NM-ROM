**Overall verdict: NEEDS-RESTATEMENT.** A2 repairs most round-2 findings, but the synthetic control is not reproducibly specified, and P1c still transfers internal-step certification through output agreement.

References below are to [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/DESIGN.md).

| Round-2 blocker | Disposition | Verdict | DESIGN evidence |
|---|---|---|---|
| 1. Broken fit initializer | PARTIALLY | NEEDS-RESTATEMENT | 574–582: initializer executes and flat-cost handling improves. The mandatory synthetic outcome inherited from 540–542 depends on unspecified noise-array ordering. |
| 2. Unsupported component-sum gate | RESOLVED | CORRECT | 584–591 explicitly withdraw the gate, acknowledge the AM-5 deviation and restrict the claim to component-kernel scaling. |
| 3. Output-only qualification transfer | PARTIALLY | NEEDS-RESTATEMENT | 593–602 adds same-job Q1/Q2 and requires both labels, but certification still uses a separate recording run linked only by six-output agreement. |
| 4. Q2 zero-denominator cases | RESOLVED | CORRECT | 604–606 distinguishes trivial zeros, infinite discrepancy, non-finites and empty indicator populations. |
| 5. Historical-statistic mismatch | RESOLVED | CORRECT | 608–616 separates H1’s three-mesh spread from H4’s historical two-mesh spread, prohibits qualification labels and withdraws the cancellation claim. |

**CPU fit check.** I used the specified six ranks, NumPy `default_rng(0)`, case bootstrap seed 20261017, and **2,000 resamples instead of 10,000**. Each sampled case carried both fixtures and all ranks.

The initializer executes. However, the design does not freeze the random-number array’s axis order:

| Noise generation order | Point estimate $\hat p$ | Bootstrap 95% interval | Declared 3 | Declared 2 |
|---|---:|---:|---|---|
| `(case, fixture, rank)`, shape `(6,2,6)` | 2.017128 | [2.010949, 2.025817] | FAIL | **FAIL** |
| `(rank, case, fixture)`, shape `(6,6,2)` | 1.997286 | [1.976174, 2.019168] | FAIL | PASS |

The first construction had **zero failed/boundary bootstrap fits**. Rescaling its optimization residuals also left 2 outside the interval; this is not simply premature optimizer termination.

**Ranked remaining blockers**

1. **The mandatory synthetic positive control can reject a correct implementation.**  
   A1.9’s fixture, retained by A2.1, specifies seed and dimensions but not which draw belongs to which observation. Both constructions above satisfy that prose. Freeze executable fixture generation, including axis order and bootstrap indexing. The rank-major construction reproduces the intended fail/pass outcomes. Do not adjust the statistical rule merely to force a noisy positive fixture to pass.

2. **A2.3 does not establish certification of the timing run’s internal trajectory.**  
   Same-job anchors, successor replays and requiring both population labels are sound and executable. But lines 594–597 compare the recording run with the unrecorded scoring run only through $W$ at the six output times. Intermediate states and validity outcomes can differ while those outputs agree; the recorded trajectory’s pooled stationarity certificate then does not certify the timed trajectory. Requiring the accuracy-job label cannot close that gap.

   Require an auditable identity check over **every internal state and relevant validity flag**, or certify the exact execution being scored/timed. If instrumentation changes the executable, explicitly establish the transfer between those executions. Apply tight-distance and Q2 predicates to the actual scoring outputs rather than silently substituting nearby certified outputs.

A2.2 honestly records its AM-5 deviation. Independent component timings can support **component-kernel scaling** without a component-sum gate; they cannot establish total solver-time scaling in $r$. I recommend carrying this narrowed scope into the coordinator’s governing amendment and verifying component arithmetic against the solver through numerical equivalence checks, without imposing an unsupported wall-time band.

A2.4 and A2.5 are total and faithful for the defects they address. I found no additional blocking WRONG item beyond the two routes above.

Read-only audit completed: no files changed and no jobs submitted.