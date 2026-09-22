# Handoff — ns3d-shift

Updated 2026-09-22. Branch `exp/2026-09-22-ns3d-shift-decoder`, forked from
`exp/2026-09-21-ns3d-grok` @ 8852b7cd.

## Idea

The translation is solved **online** as an unknown of the reduced least-squares
problem, in the co-moving (freezing) form: `u(x,t) = v(x - c(t), t)`, `v = G a`
in a fixed centered bank, so the weak residual gains one term linear in
`delta = c^{n+1} - c^n` and **nothing is shifted at run time**. See `DESIGN.md`.

## State

Design written, audited twice by `codex exec -m gpt-6-astra` (four blockers
accepted and fixed; see `results/codex-design-audit-pass{1,2}.md` and the audit
disposition section of `DESIGN.md`). Local n=8 smoke passes every check.

## Cluster

Namespace `/cluster/tufts/paralab/tawal01/ns3dshift_20260922/`, one directory per
job. Submit with `cluster/submit.sh <jobname>`. Budget: <= 1 running, <= 6 total.

## Closed seeds

202609203 and 202609211 are closed. Training 202609201, development 202609202.
The sealed cohort, if the bars pass, is 202609221 and has not been drawn.
