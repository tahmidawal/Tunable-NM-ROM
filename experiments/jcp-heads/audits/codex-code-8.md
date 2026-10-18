**NOT-READY for smk1 acceptance. NO-GO for smk2 submission pending one remaining acceptance fix.** I found no new smk2-specific blocker.

Audited commit `3386ed5308235cf1e97367b90b50a518b75e3565`. No files changed, scratch created, jobs submitted, or lab-log writes. The smk1 archive is absent locally; this audits the checker, not the smoke results.

**Round-7 dispositions**

| Blocker | Verdict | Assessment |
|---|---|---|
| 1. Rule-evidence validation | **CORRECT** | Configured populations, every rung’s arm coverage, validity vectors, conditional check evidence, predicates, pass counts, and selection/exhaustion are now checked. |
| 2. Configured accuracy evidence | **WRONG — narrowly incomplete** | Most missing evidence is now rejected, but `tighter_valid` remains unchecked. |
| 3. Every doubled-work attempt’s measurements | **CORRECT** | Earlier attempts with missing, `null`, NaN, zero, or negative `t_double` are rejected. Successful drift retries remain accepted. |

**Remaining blocker: smk1 accepts missing tighter-anchor validity**

[hacc.py:183](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hacc.py:183) emits both `tighter_valid` and `tighter_verified`. However, [smoke_accept.py:121](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:121) requires only the tighter distance and verification flag.

Confirmed with the actual `acc_rows_ok()`:

- The committed 84-row positive fixture already omits `tighter_valid` from all **20 applicable rows** and passes.
- Adding producer-shaped `tighter_valid=True` fields, then deleting all of them, still passes.
- No later `smk1()` check accesses this field, so that omission does not affect its overall verdict.

**Classification: blocking, not forgery-only.** This is missing evidence without contradictory fields. An incomplete recorder can omit the validity status while retaining the distance and `tighter_verified=False`. That verification flag does not distinguish a valid but insufficiently converged anchor from an invalid solve.

Require `tighter_valid` wherever the tighter replay is configured. Preserve `False` as a reportable outcome; do not require every anchor to qualify.

**Producer compatibility and adversarial checks**

The committed regression suite passes: **22 timing rejection cases, seven rule rejection cases, and four accuracy rejection cases**, plus positive fixtures.

Source inspection and additional in-memory probes confirm:

| Area | Result |
|---|---|
| Timing labels | Match `mesh\|name\|rule\|scheme`, including `1024\|A_R384\|gauss128\|CNR`. |
| Coverage | Five subjects/equivalence rows; ten certification rows; 20 split and 20 fused records; three doubled records. |
| Retry schemas | Split attempts have `attempt`/`ratio`; fused attempts do not. Doubled records use aggregate `attempt`. Successful drift retries in all three paths pass. |
| Certification | A’s tight/tighter status fields are required. Removing each individually rejects. A correctly needs no successor fields; the four B subjects do. |
| Pilot | All 16 expected stages match `hpilot.py`, including first-head-per-K rollouts, largest-K level variants, `L384`, split programs, and A. Memory keys and allowance agree. |
| Smk1 | Training keys, compact-head dimensions, reproduction/Jacobian identities, accuracy populations, and completion markers match the producers. Conditional rule checks and finite failed-rung/exhaustion outcomes remain representable. |

The five drivers and smk1 accuracy/rule configs remain byte-identical to their staged smk1 copies. I found no new required-key or label mismatch that would reject otherwise complete, finite, passing producer output.

**Noted, not blocking:** a fabricated doubled aggregate `attempt=200` still passes. This contradicts the producer’s bounded retry loop and is forgery-only.

**Minor:** timing fixtures still omit emitted iteration arrays and the doubled aggregate attempt field. Completing those fixtures would improve schema coverage; this alone is not a blocker.

**Ranked required fixes — blockers only**

1. Require applicable `tighter_valid` evidence in `acc_rows_ok()`. Add a producer-shaped positive fixture and a rejection regression deleting that field.

**Final verdict: NOT-READY for using `smoke_accept.py` to accept smk1; NO-GO for submitting smk2 until that remaining acceptance gap is closed.**