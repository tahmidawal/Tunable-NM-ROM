**SUBMIT — s1c smoke attempt.** No blocking findings.

- **Anchors match hr-baselines:** native damped Gauss–Newton; tight `1e-7`, tighter `1e-8`; unchanged scheme/time-step schedule; correct `certify(level=...)`. The existing round-off-stall allowance is preserved.
- **Production Q1 remains independent:** production still uses common LM, fresh certification, validity checks, ≤1% nonstationary steps, and both distance bounds. Changing anchors cannot bypass these checks. The s1b zero-budget mutation had 300/300 nonstationary steps and remains rejected.
- **s1c is consistent:** all 146 manifest entries and 139 provenance entries verified. Staged fix/design/solver files match current sources. Current and staged smoke-config hashes match `resources-s1c.json`; `S1_ATTEMPT` is `s1c`.

This approves the smoke submission, not its eventual results. No files changed or jobs submitted.