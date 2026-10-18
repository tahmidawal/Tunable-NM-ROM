**Overall verdict: NEEDS-RESTATEMENT.** A3 resolves the statistical specification and closes the certification gap in principle, but the timing implementation does not enforce the promised identity gate.

| Item | Verdict | Evidence |
|---|---|---|
| Round-3 blocker 1: reproducible synthetic control | **RESOLVED** | A3.1, `fit_power`, `h_cost_r`, and `t_h_cost_r` now specify relative residuals, array ordering, aggregation and case bootstrap. CPU reproduction confirms the stated results. |
| Relative-residual weighting | **Legitimate pre-data specification** | Equal relative weighting is defensible for multiplicative noise and costs spanning a large range. Round 3 used an unspecified weighting; this is not evidence of tuning against measured lane results. The claim that this was always the intended weighting cannot be independently established. |
| Round-3 blocker 2: certification transfer | **PARTIALLY** | Scoring and timing use the recording executable; `ident()` covers W, all recorded step states, exit reasons and iteration counts. Certification, anchor distances and ρ use the scoring trajectory. However, failed identity checks do not block timing acceptance. |
| A3.3 scope | **HONEST** | Component-kernel scaling does not establish total solver-time scaling. Numerical equivalence is an appropriate arithmetic check; the promised smoke check remains to be demonstrated. |
| Other WRONG design items | **None additional identified** | Read with amendment precedence, the remaining blocker is implementation enforcement below. |

CPU-only `python -B -c` reproduction, **10,000 bootstrap resamples**, without running `test_p1.py`:

| Noise ordering | Point exponent | 95% interval | Declared 3 | Declared 2 |
|---|---:|---|---|---|
| Case × fixture × rank | 1.999191 | [1.996274, 2.002278] | FAIL | PASS |
| Rank × case × fixture | 2.002504 | [1.997624, 2.006844] | FAIL | PASS |

Both had zero failed/boundary bootstrap fits. An absolute-residual fit gives approximately **2.017**, confirming the weighting explanation.

**Remaining blocker: identity is recorded, not enforced.**

In [p1time.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/p1time.py:213):

- `hashes_ok` and `all_hashes_ok` are saved, but `valid` at line 216 depends **only on drift**. A mismatching execution can therefore remain marked valid.
- Fused-triplet A-flank hashes are discarded at lines 219 and 222.
- Doubled-work checks cover the two B executions, but omit the single-execution flanks `o1` and `o2`.
- No current downstream consumer enforces `all_hashes_ok`; the planned analysis layer is absent.

Make any identity failure block the affected timing claims, including retries and every reference flank, and demonstrate that with an injected mismatch. The trajectory identity approach itself is sufficient; no further output-based transfer argument is needed.

No files changed or jobs submitted.