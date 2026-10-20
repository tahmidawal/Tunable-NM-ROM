**NOT-READY for smk1 acceptance. NO-GO for submitting smk2 (H200).** Remaining blockers concern incomplete evidence that can pass acceptance; forged contradictions are noted separately.

Audited commit `82b6a46ab18b74bea369de13d9510d3190224953`. No files changed, scratch created, jobs submitted, or lab-log writes. Checks used the prescribed Python with `JAX_PLATFORMS=cpu` and bytecode writes disabled. The smk1 archive is not yet present locally, so this is an acceptance-code audit, not acceptance of its results.

**Round-6 dispositions**

| Blocker | Verdict | Assessment |
|---|---|---|
| 1. Finish smk1 evidence validation | **WRONG — partially fixed** | Negative-control finiteness and successor summaries are repaired. Rule populations and underlying evidence remain insufficiently checked; accuracy evidence coverage is incomplete. |
| 2. Timing-record integrity | **WRONG — partially fixed** | Split/fused measurement integrity and retained identity failures are substantially repaired. Earlier doubled-work measurements remain unchecked. Forged aggregate contradictions are not independent blockers. |
| 3. Certification evidence and regression fixtures | **NEEDS-RESTATEMENT** | Timing certification fields now match the producer, including A’s exception. Regression coverage improved, but still lacks smk1 fixtures. Missing tests alone are not a separate blocker; the accepted omissions below are. |

The committed positive fixture and all **21 rejection tests pass**. A separate producer-shaped fixture with successful drift retries in split, fused, and doubled records also passes.

**Confirmed accepted omissions**

Every example below returned `all_controls_pass=True`.

| Path | Mutation | Classification |
|---|---|---|
| smk1 rules | Reduce every group’s population and check population to `dev6|0`; consistently shorten all corresponding vectors. | **Blocking:** a case-selection bug can produce internally consistent evidence for the wrong population. |
| smk1 rules | Remove `check_population` and per-arm `valid_q`, `valid_qp`, `valid_qpp`, `dist_check`, `rho_max`, and `max_dist`. | **Blocking:** missing numerical and validity evidence passes on summary flags alone. |
| smk1 accuracy | Remove anchor validity fields, tighter verification/agreement fields, successor validity fields, and all Q2 metrics from the short-head and linear rows. | **Blocking:** configured computations can lack recorded evidence while row-identity coverage still passes. |
| smk2 doubled work | Prepend a drift-invalid retry with `t_double` absent—or `null`—followed by a valid final attempt. Preserve correct aggregate identity/iteration flags and attempt count. | **Blocking:** incomplete retained measurements pass. No contradictory aggregate is required. |

Smk1 probes exercised the actual `smk1()` with an **84-row in-memory fixture**. File access, head loading, and expensive independent numerical calculations were stubbed; these establish acceptance-logic defects, not numerical results. Smk2 probes exercised the actual `accept_timing()` with the committed fixture.

The doubled-work `null` probe reflects the producer’s serialization convention: `QS.clean()` converts nonfinite measurements to JSON `null`.

**Producer compatibility**

I found no new required-key or label mismatch that rejects a complete, finite, passing producer record.

- **Timing:** labels match `mesh|name|rule|scheme`; five subjects and five equivalence records are expected, including `1024|A_R384|gauss128|CNR`. Two certification cases per subject produce ten certification rows.
- **Repetitions:** four B subjects × `(2 + 2 + 1)` produce **20 split and 20 fused records**. The configured doubled subject produces three records.
- **Retry schemas:** split attempts contain `attempt` and `ratio`; fused attempts do not. Doubled records have aggregate `attempt`, rather than `retries`. The current checks respect these distinctions.
- **Certification:** all subjects, including A, emit tight/tighter validity and verification fields. Only the four B subjects require configured successor evidence. **A correctly requires no `qp`/`qpp` fields.** Unverified anchors remain allowed recorded outcomes.
- **Pilot:** all **16 expected stages** match `hpilot.py`, including first-head-per-K rollouts, the largest-K level variants, `L384`, split programs, and A. Memory keys and the 10% allowance agree.
- **Smk1:** training metrics, compact-head dimensions, reproduction identities, Jacobian fixtures, 84 accuracy identities, and completion markers match the producers. Rule successor checks are conditional; skipped checks and legitimate ladder exhaustion must remain representable.

The five drivers and smk1 accuracy/rule configs are byte-identical to their staged smk1 copies.

**Noted, not blocking**

Changing selected `qp`/`qpp` while retaining the original rule trail passes. Likewise, a fabricated doubled aggregate `attempt=200` passes. These contradict deterministic producer construction and are **forgery-only** under COMMON-BRIEF §7.

**Minor:** the timing fixture omits emitted iteration arrays, so it is still not a complete producer-schema fixture. The existing identity hash includes iteration counts; this omission alone is not another blocker.

**Ranked required fixes — blockers only**

1. **Finish rule-evidence validation** at [smoke_accept.py:179](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:179). Bind population/check identities to configuration; require per-arm evidence at every rung; recompute rollout, rho, and conditional check predicates. Preserve legitimate failed-rung and exhausted-ladder outcomes.

2. **Require configured accuracy evidence across all arms** at [smoke_accept.py:165](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:165). Require applicable anchor and successor statuses/metrics, without requiring every anchor or arm to qualify. Add smk1 regressions for these omissions and the population bug.

3. **Validate every doubled-work attempt’s measurements** at [smoke_accept.py:321](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:321). Require finite positive times and recompute each attempt’s drift. Earlier drift failures should remain allowed; missing or corrupt measurements should not. Add missing/`null` retry regressions.

**Final verdict: NOT-READY for using `smoke_accept.py` on smk1; NO-GO for smk2 submission.**