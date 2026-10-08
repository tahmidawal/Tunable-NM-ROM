**NO-GO for ev1 as currently configured.** Most numerical fixes are sound, but items 9–10 remain incomplete, and the smoke does not establish the six-hour resource envelope. Waiting for tr1b does not resolve these issues.

Paths below are relative to `experiments/jcp-smooth-bank/`. Labels assess the revised implementation.

| Item | Assessment | Evidence and required fix |
|---|---|---|
| 1 | **CORRECT** | Imports and input paths agree; timing follows successful workers: `bankeval.py:34–45`, `cluster/stage.py:29–56,71–74`, `jobs/ev1.json:38–51`. **Fix:** none, provided tr1b outputs are complete and verified before staging. |
| 2 | **CORRECT** | Lane trust radius and cold codes use rotated LS coefficients; deployed behavior remains intact: `bankeval.py:70–85,260–261`. **Fix:** none. |
| 3 | **CORRECT** | Reference restriction, normalization, evolved maxima and pinned R1 arithmetic agree: `bankeval.py:301–327`, `make_report.py:258–271`. **Fix:** acceptance propagation still needs item 10. |
| 4 | **CORRECT** | Common-support D2/D4 are stored, used for H1 and now displayed: `bankeval.py:365–379`, `make_report.py:206–209,409–413`. **Fix:** none. |
| 5 | **CORRECT in code; smoke verification incomplete** | Full/320-test norms are now persisted and checked; descriptive interpolation exists: `bankeval.py:413–419`, `make_report.py:124–127,158–172`. **Fix:** retain a smoke from this revision—the supplied JSON lacks these new norm fields and the directory contains no norm NPZs. |
| 6 | **CORRECT** | Spectrum keys, unresolved counts, individual envelopes and conditional aggregates are implemented: `bankeval.py:185–191,484–499`. **Fix:** none to computation; supplied smoke lacks the NPZ artifacts needed to verify persistence. |
| 7 | **CORRECT** | Full-field finiteness/distances, C4 normalization and minimum-over-steps C5a follow the amended controls: `bankeval.py:251–252,302,323–327,420`. **Fix:** none. |
| 8 | **NEEDS-RESTATEMENT** | Smoke proves execution, not full-cohort A100 feasibility. Timing retains six banks simultaneously: `banktime.py:49–85`; evaluation scales to 4,028 states per setting: `bankeval.py:387–403`. **Fix:** measure representative A100 throughput and peak memory before accepting six hours; see below. |
| 9 | **WRONG — partially repaired** | Empty timing sets now fail, and `valid=False` suppresses costs. But reporting never checks `complete`, independently verifies coverage, or checks invocation finiteness: `banktime.py:117–120`, `make_report.py:370–373,460–469`. **Fix:** require complete, finite, exact expected subject/case/repetition coverage before publishing. |
| 10 | **WRONG — partially repaired** | Counts replace exact population validation; selected m* rollout coverage, artifacts and several finite diagnostics remain unchecked: `make_report.py:110–154`. Global failure clears headline verdicts but leaves nested `H2[…]['passed']=True`: `364–369`. C6 filtering still considers only gref failures: `92–97,455–458`. **Fix:** validate identities/artifacts and all required diagnostics, invalidate nested verdicts, and report ST/S with/without the union of failed cases for every arm. |

Two isolated, in-memory fixtures confirmed item 10: duplicating the same case to nominal counts passes `complete_inputs`; injecting a gauss32 failure records C6 failure but leaves `C6_failed_cases` empty.

**Smoke evidence.** The saved evaluation has `complete=True`, took **2,053.25 s**, and records GPU/f64/highest precision. That flag means execution finished; the report correctly rejects its insufficient population.

- **R1 fails both settings**, as expected: lat64-state Gauss64 ρ is **0.002049 / 0.005623**, versus pinned **0.018554 / 0.023300**.
- **C4 passes:** largest recorded discrepancy across populations is **2.60×10⁻⁹ / 9.85×10⁻¹⁰** for acc/fast.
- **C5a passes:** minimum error **2.35×10⁻⁷**.
- **C6:** all supplied rollouts have zero nonaccepted exits. Tight-solver distances are **9.09×10⁻⁷ / 1.09×10⁻⁵**.
- The generated report displays these controls, completeness failures and R1 failure, with no ranked winners. However, its headings falsely say **38 cases / 64 spectrum states** despite using **1 / 50** (`make_report.py:432,444`). Generate those counts from inputs.
- No timing output is supplied, so timing suppression is verified statically, not by this smoke.

**Runtime versus six hours.** A reliable A100 point estimate is not identifiable from this smoke: its timings combine compilation, setup and execution, and the two logged variants take **2,053 s versus 809 s**. Multiplying everything by 38 would be misleading.

A conservative scaling scenario—38× ordinary work, 6× dev6-only controls, two sequential variants—gives **32.7 GB10-equivalent hours per worker**, before timing and additional L=1024 mesh costs. It overcounts reusable setup/JIT:

| Assumed effective A100 acceleration | Evaluation scenario, before timing |
|---|---:|
| 4× | 8.2 h |
| 8× | 4.1 h |

Thus this conservative scenario needs **more than 5.45× effective acceleration** to fit, with additional headroom for timing. Neither acceleration is measured here. Most quadrature work does **not** grow 16× with mesh area; bank construction and decoding do.

At L=1024, the six retained acc mesh banks alone occupy about **18 GiB**, before quadrature blocks, temporaries and executables. **40GB is unverified; 80GB offers more memory margin but does not establish runtime.** Calibrate on the intended A100 and revise the wall limit or execution layout accordingly.

No repository files or lab-log entries were modified.