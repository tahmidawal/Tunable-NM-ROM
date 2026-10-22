**Verdict: NEEDS-RESTATEMENT — §8 is not yet a complete freeze.**

- **CORRECT — E10-s:** Adding factor 0.125 permits certification at 0.25; the bound \(1+7+3=11\) matches Phase B’s unchanged candidate chain. Preserve Q2-before-step precedence and evaluate round 11 fully.
- **WRONG — E10-q “single change”:** Phase B’s `next_candidate` ends at gauss256; gauss384 is its certification successor. Allowing gauss384 production additionally enlarges the ladder. Fix: retain gauss256 or explicitly register this second change and a matched control.
- **NEEDS-RESTATEMENT — Local escalation:** Freeze repeated re-solving from the same pre-step history until ρ passes; specify nonfinite evidence, failed retries, maximum-rule failure, and which rejected attempts enter cost/certification records.
- **NEEDS-RESTATEMENT — Outer state machine:** State whether full-rollout Q2 failure still advances the starting candidate at unchanged factor. Local ρ checks cannot replace that transition; freeze the resulting round bound.
- **NEEDS-RESTATEMENT — Full-rollout Q2:** “Same policy with every rule used replaced” is ambiguous between replaying the recorded schedule and independently adapting. Freeze one, including identical initialization, successor maps for every reachable rule, and valid-run requirements.
- **NEEDS-RESTATEMENT — Q1/step comparisons:** Define adaptive tight/tighter and half-step schedules, promotion semantics, and per-step certification using the actual active rule. Phase B’s fixed-rule executor does not specify these.
- **CORRECT — E10-s control:** SM-r FC1|0 at gauss256, factor 0.5 has half-step distance 0.005294 > 0.00025. Pin that rule/state and evidence identity.
- **WRONG — E10-q control justification:** SM-r’s forced fib4181 escalation followed rollout distance 0.004762, while recorded maximum ρ was only 0.032637 < 0.116. It does not establish the new local trigger. Fix: identify an actual violating step, plus passing and threshold/nonfinite fixtures.
- **NEEDS-RESTATEMENT — F13:** All eligible FC1 cases passing adaptive Q2 is a reasonable operationalisation, but define Q2 at the scored terminal run, require the population minimum, and distinguish missing certification from measured failure.
- **NEEDS-RESTATEMENT — Endpoint precedence:** “Otherwise STOP” conflicts with inherited INCONCLUSIVE outcomes for inadequate populations/unavailable prerequisites and nonfinite K1 median. Restate the complete precedence explicitly.
- **CORRECT — Reuse:** Reusing K1 and references supports descriptive comparisons without timing claims. Label E10 developmental: these cases informed its design; they provide no independent confirmation.
- **NEEDS-RESTATEMENT — Budget:** 0.4 + 0.8 + 0.6 = 1.8 GPU-h fits 10, but ceilings do not establish completion. Require smoke-based adaptive certification costs and admission accounting for spent, running, pending, remaining work and retries against E10’s cap.
- **NEEDS-RESTATEMENT — Supersession:** Explicitly supersede §§1, 3, 5–7 where they retain no-release language, new-seed confirmation, rescoring, and E10-c scope.

Read-only audit completed; no files, scratch, jobs, main or LAB-LOG changes.