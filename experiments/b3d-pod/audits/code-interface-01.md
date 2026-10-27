# Independent parameterized POD interface review

Reviewer: `/root/independent_review`, read-only Codex agent.
Verdict: **READY for Stage F reuse**.

Reviewed source SHA256:
`464c79e3ebd2b3f6d3c4cadfcdc42689deba23e435a4ac2dd0b03ba52b128b3c`.
The explicit buffer is consistently passed to evolution, output schedule and fresh
certifier; the schedule rejects more steps than its buffer. The inherited global
buffer constant is unchanged. Tiny controls show exact old/new CN coefficients,
consistent fine BDF2 output slots and finite fresh certificates.

This approves the equations and buffer plumbing, not scientific qualification of
every schedule. Finer POD subjects require their own complete Q1 anchors and a frozen
Stage F manifest before a comparative claim. The frontier driver must call
`functions_for` for finer arms, rather than the old fixed-step `setup` query.
