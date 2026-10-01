1. **CORRECT — numerical tables reproduce.** Read-only execution with the generator’s write intercepted reproduced the report exactly. Python independently recomputed worst errors, medians, outlier counts, timing medians, eligible comparators and speedups for **all 48 rollout rows**. No numerical mismatch found. Here are 18 checked values from the six run summaries:

   | Cohort/mesh | Head worst, % | Timing median, ms | Speedup |
   |---|---:|---:|---:|
   | Development 32³ | 1.022266 | 5.711230 | 0.516315 |
   | Development 64³ | 1.253355 | 7.978341 | 2.395331 |
   | Development 96³ | 1.266824 | 16.055946 | 5.736321 |
   | Test 32³ | 1.084756 | 5.718247 | 0.934935 |
   | Test 64³ | 1.265049 | 7.842085 | 3.035185 |
   | Test 96³ | 1.279171 | 15.448011 | 7.923684 |

   Sources: `rollouts.*.errors`, `timing.fast.*.repetitions`, `cnab2.*`. **NEEDS-RESTATEMENT:** “every number is read from JSON” is literally false: unknown counts use hard-coded ranks plus three; numerous configuration numbers and thresholds are literal generator strings.

2. **CORRECT — all bar verdicts and comparator selections.** Against DESIGN §5/A4:
   - **(a) FAIL:** floor/POD ratios **8.050003, 9.645528, 9.750405**, all above 1.5.
   - **(b′) PASS:** head/floor ratios **1.018121, 1.007988, 1.007884**; development 96³ speedup **5.736321**.
   - **Original (b) FAIL:** development 96³ error **1.266824% > 0.25%**.
   - **(c) FAIL:** identical bank hashes; head max/min **1.239231 passes**; floor max/min **1.251819 fails**; pre-Löwdin deviations **0.0190047, 0.00136, 0.00149** fail at every mesh.
   - All 48 comparators independently match the fastest finite CNAB2 setting with worst error ≤100% and ≤the arm’s error. Ratios use same-job repetition medians.

3. **CORRECT — A4 disclosure; NEEDS-RESTATEMENT — registration history.** The opening, bar sections and retrospective disclose the post-pilot user decision, changed accuracy target, cancelled fits and unchanged bar (c). However, `report_plan.wrong[1]` conflates the original five-hour plan with A3’s later ten-hour/band-penalty amendment. State that sequence explicitly. “Opened once” should say **once for this lane’s coordnet evaluation**: DESIGN A2 explicitly records earlier use of this test cohort by parent lanes.

4. **CORRECT — narrow performance claims.** `answer[0]`’s “within a few per cent” holds for **cohort evolved-worst / bank evolved-worst**: excesses range **0.509–1.812%**. It does not establish per-case closeness. “More than 5× on both cohorts” correctly applies specifically to **96³**, with **5.736×/7.924×**. Shared bank, per-mesh heads and failure to match POD accuracy are supported.

5. **NEEDS-RESTATEMENT — causal and general claims.**
   - `wrong[2]` appropriately labels aliasing’s effect on bar (c) as an inference; `answer[2]` then states it as established causation and generalizes to **every resolving mesh**. Only three meshes were evaluated; even 64³/96³ fail the orthonormality criterion.
   - Shrinking autodiff/FFT disagreement supports under-resolution; successful spectral/shift checks support the implemented frame derivative. These do **not** isolate the cause of the floor discrepancy.
   - `answer[1]`/`open[1]`: training loss **5.1948e−5 versus 3.7295e−7** establishes a gap to unrestricted rank-64 POD. It does not distinguish optimization limits, network capacity and sampling effects. “Fitting shortfall” is defensible only in that broad sense.
   - `open[0]`: cancelled fits **would have tested**, not necessarily “would have answered,” the aliasing explanation.
   - Hand-written numerical assertions remain in `report_plan.json`: training durations, ranks, approximately 1%, 0.25%, 5×, 1.25 and “more than ten per cent.” JSON storage does not make these source-generated.

6. **CORRECT — recorded controls and integrity gates.** Fixed-frame floors are **70.1592%** at all meshes; pilot no-centring is **97.5855%**; frozen-frame rollouts are **40.19–42.57%**, exceeding 5%. All six mesh audits pass and reject their perturbed copies; timing bounds, LM parity and timed-output agreement pass recomputation. No-centring was checked only at 32³, correctly shown with dashes elsewhere. **NEEDS-RESTATEMENT:** “pilot gates fired” means diagnostic **flags**, not failed abort gates. “Converged LM reference” exceeds the recorded evidence: parity is recorded, convergence certification is not.

7. **CORRECT — successful-run provenance; WRONG — cancelled-job description.** Printed IDs, GPUs, summary hashes and bank/manifest hashes match their sources. Recorded source-file hashes match the stated Git commits; the frozen manifest exists identically in the test commit; successful logs contain `jax_backend=gpu`.
   - **WRONG:** “Jobs that never computed” includes cancelled jobs whose logs show thousands of optimization steps. Likewise, “longer fits or larger ranks were not run” should say **not completed/evaluated**.
   - **WRONG literally:** “three architectures, 30 min each.” Pilot logs record approximately **1669/2362/2522 seconds** for A/B/C training. Thirty minutes was the nominal budget.
   - **NEEDS-RESTATEMENT:** the malformed-submit anecdote is not independently supported by the specified JSON evidence.

8. **NEEDS-RESTATEMENT — report-rule compliance.** H1/status, LaTeX display equations, coloured Mermaid and an ending glossary are present. Glossary coverage is incomplete: unknowns, GPU ms/timing case and repetitions, derivative/shift/tensor checks, restriction, provenance columns and several acronyms remain undefined. Some prose math remains outside LaTeX. Two definitions overclaim: equal worst/floor does not prove “the solve loses nothing”; mesh consistency alone does not imply an identity Gram without normalization.

**Required fixes:** qualify causal/general claims; distinguish original registration, A3 and A4; generate numerical prose from sources; correct cancelled-job and pilot-duration wording; qualify test reuse and LM convergence; complete/correct the glossary and math formatting. **The numerical tables and reported bar verdicts need no correction.** No files modified, GPU jobs run or pushes made.