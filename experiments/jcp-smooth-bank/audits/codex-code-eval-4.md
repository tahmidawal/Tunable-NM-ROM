**NO-GO for staging ev1: item 9 remains unresolved.**

- **8 — A6 implemented:** six GPUs, one variant each, 10 hours, 12 CPUs. Runtime and peak memory on A100 remain unverified; the allocation is **60 GPU-hours**.
- **9 — Still faulty:** in-memory fixtures confirm incomplete timing, short invocation counts, and nonfinite durations are suppressed. But **duplicate invocations replacing missing coverage pass**, as does missing-case coverage with an adjusted `expected_invocations`. [make_report.py:384](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/make_report.py:384) trusts counts instead of independently validating exact subject/case/repetition coverage.
- **10 — Requested regressions pass:** duplicate evaluation cases and non-dev6 `gauss768` rows are rejected; a `gauss32` C6 failure enters `C6_failed_cases` and excludes that case from both S/ST filtered maxima; failed acceptance gates clear every nested H2 pass.
- **Invocation arithmetic is correct:** two timed directions × unique rules × six cases × repetitions, summed across bank/settings. With six banks, two settings, three repetitions: **432–864 invocations** for one–two rules each.

Fix exact timing coverage validation before staging. No repository files were modified.