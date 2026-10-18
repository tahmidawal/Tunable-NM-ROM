**NOT-READY for accepting the smk1 archive. NO-GO for submitting smk2 (H200).** Both acceptance paths still accept missing or contradictory evidence. I found no new driver blocker or producer/consumer schema mismatch that would spuriously reject an otherwise passing smoke.

Audited HEAD `ecf34ab370f8a5368f4fad7612df78df37060a99`. No files changed, scratch created, jobs submitted, or lab-log writes. Checks used the prescribed Python, `JAX_PLATFORMS=cpu`, and disabled bytecode writes.

The committed `test_accept.py` passes its positive fixture and all **17 rejection tests**. Additional in-memory probes below nevertheless pass acceptance.

**Round-5 dispositions**

| Item | Verdict | Assessment |
|---|---|---|
| Blocker 1: timing identity, retries, corrupt numerical evidence | **WRONG — partially fixed** | Ordinary retry hash failures and drift-only injected faults are rejected. Corrupt timings can still pass as invalid repeats; attempt/aggregate consistency remains incomplete. |
| Blocker 2: smk1 coverage and control consistency | **WRONG — partially fixed** | Accuracy identities, negative-control case coverage, wrong-mesh exceptions, and group names are checked. Rule evidence and successor-validity summaries remain insufficiently checked. |
| Blocker 3: pilot stages and individual peaks | **CORRECT** | Required stages match the committed producer/configuration. Every peak must be finite and positive; maximum and allowance are recomputed. |
| Blocker 4: duplicate equivalence, doubled drift, regressions | **NEEDS-RESTATEMENT** | Duplicate rejection and doubled final-drift consistency are correct. Regression coverage is incomplete: no smk1 fixture, incomplete certification rows, and no doubled attempt records. |
| Minor: nonfinite short-head means | **CORRECT** | Reconstruction and oracle means are checked for finiteness. |
| Minor: contradictory fused stored drift | **CORRECT** | Aggregate fused drift is checked against raw times. Attempt-level evidence remains a separate blocker. |

**Confirmed accepted counterexamples**

Every mutation below returned **`all_controls_pass=True`**.

For smk2, these call the actual `accept_timing()` using the committed passing fixture:

| Mutation | Remaining defect |
|---|---|
| Set an ordinary split’s final and retained-attempt `t_init=-1`, with `valid=False` | Corrupt evidence is accepted as an invalid repeat. Flag agreement does not establish record integrity. |
| Set an ordinary fused attempt’s `t_fused=NaN`, retaining its finite aggregate | Fused attempt times are unchecked. |
| Change split aggregate `t_evolve` and `ratio` from `0.5` to `50`, leaving its last attempt unchanged | Individually consistent aggregate numbers need not match the recorded measurement. |
| Add doubled attempts with `identity_ok=False` and `same_iterations=False`, retaining true aggregate flags | Doubled retry evidence is ignored. |
| Remove all attempts from the injected-fault split and fused records | False aggregate hash flags alone satisfy the identity-fault demonstration. |
| Use the committed certification fixture unchanged | It already omits anchor validity/verification and configured successor evidence while passing certification acceptance. |

The relevant logic is in [smoke_accept.py:207](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:207), [line 220](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:220), and [line 253](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:253).

For smk1, I exercised the actual `smk1()` with an in-memory fixture containing all **84 configured accuracy rows**. File reads, completion markers, head loading, and expensive independent numerical calculations were stubbed. These demonstrate acceptance-logic defects, not GPU results.

| Mutation | Remaining defect |
|---|---|
| Set an under-resolved row’s `succ_valid=False` and `check_valid=False`, retaining `successors_valid=True` | The decisive successor-validity summary is not recomputed. |
| Delete those two validity fields from every under-resolved row | Missing evidence also passes. |
| Replace every group’s trail with one entry containing the expected arm names mapped to empty dictionaries | Group/arm-name coverage substitutes for actual rule evidence. |
| Change a first-rung selection to “ladder exhausted,” with `chosen=None` | Exhaustion is accepted without traversing the selectable ladder or checking the existing passing evidence. |
| Delete `dist_qp` and `rho_max` from every accuracy row | Configured Q2 evidence disappears while row-identity coverage still passes. |
| Insert a NaN at the end of the last under-resolved `rho` array | Nested Python `max()` hides the NaN; the finite stored maximum still agrees. |

See [control acceptance:109](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:109), [accuracy summaries:144](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:144), and [rule acceptance:162](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:162).

**Producer compatibility**

Source inspection confirms the existing expectations match what the drivers write:

- **Timing:** five subjects and five equivalence records, including `1024|A_R384|gauss128|CNR`; two certification cases per subject, totaling ten rows.
- **Repetitions:** four B subjects × `(2 dev0 + 2 dev1 + 1 val0)` gives **20 split and 20 fused records**.
- **Doubled work:** three records for Hinc BE, `dev6|0`. The producer also writes retained attempts, which acceptance currently ignores.
- **Certification:** `htime.py` writes tight/tighter validity and verification fields for A as well as B. Successor fields apply to the four configured B subjects. **A has no configured `qp`/`qpp`; requiring those fields for A would be incorrect.**
- **Pilot:** the expected **16 stages** agree with `hpilot.py`: seven setup stages, first-head rollouts `P4s0/P8s0/P16s0/P32s0`, `L384`, two P32s0 level variants, split programs, and A. No separate incumbent rollout is expected. Memory keys and the 10% allowance agree.
- **Training/accuracy:** training means, compact-head dimensions, emitted-head comparison, completion markers, reproduction identities, Jacobian fixtures, accuracy identities, and three wrong-mesh exception strings agree.
- **Rules:** `hrule.py` writes populated trails and conditional successor-check evidence. Legitimate ladder exhaustion must remain acceptable; claiming exhaustion without its trail must not.

The five driver files and smk1 accuracy/rule configs are byte-identical between the current lane and staged smk1 copies.

Unverified tight anchors remain a reportable smoke outcome. A repair should require their recorded status, **without requiring every anchor to be verified**. Likewise, rule checks must respect the producer’s conditional execution of successor checks.

**Ranked required fixes — blockers only**

1. **Finish smk1 evidence validation.** Recompute `successors_valid`; validate negative-control numerical arrays before reductions; require configured Q2 evidence; validate rule populations, rung sequence, per-arm evidence, pass counts, and consistency of selection/exhaustion. Preserve legitimate numerical-failure outcomes.

2. **Separate timing-record integrity from timing validity.** Require finite positive measurements and internally consistent retained attempts regardless of `valid`. Bind aggregates to the final attempt; validate retry counts/order and hash/iteration aggregates across split, fused, and doubled records. Require actual failed-hash evidence for the injected fault. Continue allowing legitimate drift-invalid repeats.

3. **Require producer-shaped certification evidence and regression fixtures.** Check required anchor and applicable successor fields without imposing new convergence requirements or successor requirements on A. Add smk1 regressions and complete timing fixtures covering the counterexamples above.

The two round-5 **minor** items are fixed; neither independently blocks submission.

**Final verdict: NOT-READY for smk1 acceptance; NO-GO for smk2 submission.**