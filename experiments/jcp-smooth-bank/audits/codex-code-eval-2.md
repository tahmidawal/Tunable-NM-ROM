**NO-GO for `ev1` as currently implemented**, even after `tr1b` finishes. The deterministic crash is fixed, but invalid timing and incomplete evaluations can still yield published results or passing verdicts.

Paths below are relative to `experiments/jcp-smooth-bank/`. Labels assess the **current implementation** against each previous item.

1. **CORRECT — imports/staging.** `bankeval.py:34–45`, `cluster/stage.py:29–56,71–74`: dependencies are staged and timing follows successful workers on one GPU. `jobs/ev1.json:38–51` correctly uses `runs/tr1b/archive/output/<arm>`. **Fix:** none required for those paths; stage only after complete, verified training outputs exist.

2. **CORRECT — trust radius/cold codes.** `bankeval.py:70–85,260–261`: lane variants use rotated LS coefficients; deployed evaluation retains the original trust calculation. **Fix:** none.

3. **CORRECT — reference metrics and R1 arithmetic.** `bankeval.py:301–314` preserves restriction, initial-norm normalization and evolved maximum. `make_report.py:211–228` now implements the pinned R1 comparisons. **Fix:** enforce R1 in exported verdicts, as covered by item 10.

4. **CORRECT — P2 calculation/storage.** `bankeval.py:365–379` saves common-support D2 and per-state D4 errors; `make_report.py:163–166` uses both common-support stencils for H1. A5 explicitly drops the secondary training-mesh projection. **Fix:** expose common-support D2 in the report table (`make_report.py:357–361`), which still displays full-interior D2.

5. **NEEDS-RESTATEMENT — primary denominator loophole closed; evidence incomplete.** `bankeval.py:413–415` records invalid full-target norms, and `make_report.py:119–120` rejects them. However, `bankeval.py:406–408` still clamps denominators, including the unchecked first-320 target; only norm summaries are retained. Descriptive interpolation remains absent (`432–450`). **Fix:** persist and validate per-state denominators, including the 320-test diagnostic; generate the specified interpolation separately.

6. **CORRECT — spectra crash/persistence fixed.** `bankeval.py:185–188,482–495`: epsilon keys agree, unresolved classifications are counted, individual envelopes and resolution diagnostics are saved, and aggregates are explicitly conditional on resolution. **Fix:** none.

7. **CORRECT — revised controls.** `bankeval.py:302` checks full fields/internal states; `323–327` supplies full-field distances for dev6 controls; `416` matches the target-check denominator. `251–252` correctly uses the **minimum error across steps**, as A2.5 specifies. **Fix:** none; retract the previous demand to gate exclusively on step `1e-5`.

8. **NEEDS-RESTATEMENT — resource feasibility remains unverified.** `jobs/ev1.json:4–7` still allocates six hours for two sequential variants per worker plus timing. `bankeval.py:336` bounds rule caching, but `banktime.py:49–85` retains all timing subjects simultaneously. **Fix:** obtain representative phase timings and peak-memory measurements before accepting this resource envelope. Static inspection does not establish an inevitable OOM or timeout.

9. **WRONG — timing validity is computed but ignored by reporting.** `banktime.py:97–117` now saves coefficient differences, hashes and exits; numerical comparison appropriately replaces mandatory bitwise equality. However, `make_report.py:297–302,403–410` publishes timings regardless of `valid` or `complete`. An empty invocation set also passes `all(...)` at `banktime.py:117`. **Fix:** require complete expected invocation coverage and valid comparisons; suppress invalid timing tables/plots and display coefficient agreement rather than hash equality as acceptance evidence.

10. **WRONG — acceptance/completeness still permit false passes.** Noise, two-setting H2 direction checks and ranking are repaired (`make_report.py:145–206`). Remaining failures:
    - `105–129` never checks evaluation completion, exact case/state populations or required artifacts. An isolated fixture with `complete=False` and one case still returned **H2 PASS/useful winner**.
    - `318–319,430–450` uses failed global gates only for a prose warning; PASS flags and ranked winners remain in `report_numbers.json`.
    - `92–95` excludes only gref-failed cases; that filtered metric is not printed, and corresponding ST/all-arm reporting is absent.

    **Fix:** validate complete inputs and finite diagnostics before decisions; propagate global acceptance failure into every verdict/ranking and exported JSON; report ST/S errors with and without all C6-failed cases.

Read-only review completed using static inspection and isolated report-function fixtures. No repository files—including the lab log—were modified; `bankeval.py` was not run.