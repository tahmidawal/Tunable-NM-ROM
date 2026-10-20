**NO-GO for the complete smk1 → smk2 sequence as currently gated.** The remaining blocker is smoke acceptance: it still accepts missing required evidence. I found no new smoke-driver execution blocker. A7 adequately bounds the accepted account-cap race and the pilot’s interpretation.

Audited HEAD `1de4244da`. No files changed, no jobs submitted, no scratch used. The pre-existing `HEARTBEAT.json` modification remains.

**Disposition of round 3’s four ranked fixes**

| Round-3 fix | Verdict | Resolution |
|---|---|---|
| 1. Account-cap enforcement | **CORRECT under A7** | [submit.sh:38](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/cluster/submit.sh:38) now fails closed when the locked queue query fails. Shell mocks confirmed failure and eight-job responses never reach mock submission. [A7:649](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/DESIGN.md:649) explicitly accepts the race with nonparticipating lanes; it does not claim campaign-wide exclusion. |
| 2. Smoke acceptance | **WRONG — partially resolved** | [smoke_accept.py:65](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:65) enforces reproduction/Jacobian identities; line 163 binds timing subjects/cases to the supplied config; line 169 rejects nonfinite fused measurements. Other required evidence can still disappear without rejection, detailed below. |
| 3. Pilot working-set claim | **CORRECT as restated** | [A7:654](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/DESIGN.md:654) and [smoke_accept.py:194](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:194) explicitly call this a measurement probe, disclaim a proven bound, and name the memory-deferral outcome. |
| 4. Offline `fib4181` dispatch | **CORRECT; round-3 premise corrected** | [hscore.py:36](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hscore.py:36) rejects non-Gauss rules. Crucially, [A1.4:365](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/DESIGN.md:365) already replaced the frozen Fibonacci rule with a common Gauss policy. Round 3 incorrectly called `fib4181` part of the current registered production set. Rejecting it is the appropriate fix; implementing Fibonacci is unnecessary. |

This resolves round 3’s per-file **WRONG/NEEDS-RESTATEMENT** findings for `submit.sh`, `hpilot.py`, and `hscore.py`. `smoke_accept.py` remains **WRONG**.

**Confirmed remaining failure scenarios**

I constructed a passing timing fixture from the checked-in smoke configuration, then changed one item at a time. Only result-file reads were mocked; the actual `smk2()` acceptance function and timing validator ran.

| Mutation | `all_controls_pass` |
|---|---:|
| Passing baseline | `True` |
| Remove every split/fused equivalence record | **`True`** |
| Empty every subject’s certification list | **`True`** |
| Keep only one ordinary split invocation plus one injected-fault invocation | **`True`** |
| Keep only one of three doubled-work repetitions | **`True`** |
| Set timing `complete=False` | **`True`** |
| Replace fused ratios with finite positive `123`, retaining unit measured times | **`True`** |
| Set pilot peak to NaN while retaining `within_bar=True` | **`True`** |
| Remove the expected CN-R subject | `False` |
| Replace ordinary fused ratios with NaN | `False` |

The remaining defects are specific:

- **Vacuous equivalence and certification acceptance.** [Line 155](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:155) uses `all()` without checking expected equivalence identities. [Lines 184 and 198](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:184) similarly accept empty certification lists. Keeping subject names does not establish that their controls ran.
- **Split and doubled-work coverage remains self-declared.** [Line 192](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:192) validates only supplied split rows. [Line 181](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:181) requires a nonempty doubled-work list, without checking its configured subject/case and three repetitions.
- **Some readily recomputable evidence is trusted.** Fused ratios need only be positive and finite; their agreement with `t_fused / mean(tA1,tA2)` is unchecked. Pilot acceptance trusts `within_bar`, without validating the peak or recomputing the allowance predicate. Completion is not enforced by this checker.

These are acceptance defects, not evidence that the current drivers actually omit those records. They matter because the gate is intended to detect precisely such regressions or incomplete evidence.

**Per-file verdicts**

“CORRECT” means within the requested smoke scope, not that unrun GPU controls are guaranteed to pass.

| File | Verdict | Assessment and failure boundary |
|---|---|---|
| [hacc.py:237](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hacc.py:237) | **CORRECT** | Four analytic/autodiff fixtures per head; per-row comparisons reject NaNs. Both linear reproduction schemes run at line 284. Controls retain the evidence needed for their stated checks. |
| [htime.py:117](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/htime.py:117) | **CORRECT** | Both anchors retain internal states, certificates, budget counts, IC records and validity. Lines 233–303 retain split/fused retries and doubled-work evidence. Missing-evidence failures are in the consumer. |
| [htrain.py:235](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/htrain.py:235) | **CORRECT, wording qualification** | Reloaded compact parameters are compared with in-memory emitted head parameters; `COMPLETE` is written at line 259. The comment at line 241 still must not be read as verification by reloading the pickle. |
| [hpilot.py:60](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hpilot.py:60) | **CORRECT as an A7 probe** | Separate actual-K stand-ins and retained inputs/outputs exercise meaningful storage. Unrepresented compilation/allocation patterns remain possible; `within_bar` cannot authorize a production-memory guarantee. |
| [hcommon.py:44](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hcommon.py:44) | **CORRECT** | Rejects compact archives naming another source checkpoint; checks head dimensions and shared bank leaves. |
| [harm.py:86](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/harm.py:86) | **CORRECT** | Both head and linear `run` wrappers reject schedule/buffer mismatches; initial-fit veto remains enforced. |
| [hrule.py:75](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hrule.py:75) | **CORRECT** | Requires bounded `min_pass`; selection applies that threshold at line 143. Two-of-three tests pass. |
| [hscore.py:33](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hscore.py:33) | **CORRECT** | Gauss-only dispatch now fails closed. Accuracy-record scope and certificate provenance are explicit. `q3_flag` handles the valid nonempty spectrum contract, including rank-deficient, zero and nonfinite spectra. |
| [hstats.py:114](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hstats.py:114) | **CORRECT** | Holm decisions remain mandatory; missing rung decisions cannot produce the positive at-most-cost label. Supplied statistical fixtures pass. |
| [smoke_accept.py:151](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:151) | **WRONG** | Concrete missing-evidence and inconsistent-value counterexamples above. Round-3 fixture identities, missing-subject and nonfinite fused-ratio examples are closed. |
| [test_h.py:60](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/test_h.py:60) | **CORRECT, limited coverage** | Full suite passes, including seed threshold, Holm, Q3 and timing self-tests. It lacks acceptance omission/mutation regressions. |
| [ledger.py:82](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/ledger.py:82) | **CORRECT for retry/pending logic** | Base/retry exclusions both returned **4 hours**; empty mandatory inventory is rejected. Saved pending reservations are restored after failed attempts. |
| [cluster/submit.sh:24](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/cluster/submit.sh:24) | **CORRECT under A7** | Lane serialization, one-smoke-at-a-time and fail-closed queue handling are present. The explicitly accepted nonparticipating-lane race remains. |
| [cluster/stage.py:45](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/cluster/stage.py:45) | **CORRECT for smokes** | Enforces two-hour limit and stages committed blobs with manifests; GPU partition/preflight and f64/highest settings remain. Smoke staging bypasses production certificates. |
| [cluster/collect.py:47](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/cluster/collect.py:47) | **CORRECT** | Verifies original remote outputs, excludes pickles from transfer, retains original/filtered manifests, and verifies collected files against the filtered manifest. |
| [make_configs.py:43](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/make_configs.py:43) | **CORRECT for smokes** | In-memory regeneration exactly matched all five checked-in smoke JSONs. |
| [smk1train.json:4](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/configs/smk1train.json:4) | **CORRECT** | Identity plus short K=4/K=32 training arms. |
| [smk1acc.json:15](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/configs/smk1acc.json:15) | **CORRECT** | All38 incumbent/both schemes, short heads, linear arms and reproduction fixtures. |
| [smk1rule.json:31](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/configs/smk1rule.json:31) | **CORRECT as plumbing coverage** | Duplicate short-head archive is explicitly disclosed; independent-seed evidence is not claimed. |
| [smk2time.json:26](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/configs/smk2time.json:26) | **CORRECT** | Normal subjects, cohort-specific repetitions, doubled work and deliberate identity fault are configured. |
| [smk2pilot.json:32](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/configs/smk2pilot.json:32) | **CORRECT** | Twelve independent actual-dimension stand-ins plus incumbent and linear ladder, interpreted under A7. |

`resources.py` is **not a smoke dependency**. Its rewrite and the production configuration audit remain appropriately deferred.

**Ranked remaining required fixes**

1. **Close acceptance coverage.** Require expected unique equivalence subjects, certification subject/case pairs, split invocation identities and configured doubled-work repetitions. Require completed evidence. Preserve the distinction between a recorded invalid repeat and a missing repeat.
2. **Check numerical consistency where raw values are already available.** Recompute fused ratios and pilot memory predicates; reject nonfinite inputs and contradictory stored flags. Validate equivalence values alongside their flags.
3. **Add CPU acceptance regressions** for the confirmed counterexamples, including a complete passing fixture.

**Final decision: NO-GO for the current smk1 (A100-80G) → smk2 (H200) release sequence until these acceptance fixes land.** No additional campaign-lock implementation or proven pilot memory bound is required for these smokes under A7.