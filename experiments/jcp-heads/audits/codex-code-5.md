**NO-GO for submitting smk1 (A100-80G), and NO-GO for the subsequent smk2 (H200).** The round-4 counterexamples are closed, but both acceptance paths still admit missing or contradictory evidence. I found no new producer/consumer key mismatch that would reject an otherwise passing real smoke.

Audited HEAD `8a6e5680021dc9e8ffed9909d35d0a99d68c703b`. No files changed, jobs submitted, or scratch created. Checks used the prescribed Python with `JAX_PLATFORMS=cpu` and bytecode writes disabled.

**Round-4 fix dispositions**

| Ranked fix | Verdict | Assessment |
|---|---|---|
| 1. Close acceptance coverage and require completion | **WRONG — partially resolved** | Configured timing identities and completion are now checked. Equivalence duplicates are silently collapsed; smk1 accuracy/rule coverage and pilot stage coverage remain unchecked. |
| 2. Recompute numerical consistency | **WRONG — partially resolved** | Fused ratios and aggregate pilot allowance are correctly recomputed. Retry hash failures, doubled-work drift, individual pilot peaks, and contradictory control evidence still escape. |
| 3. Add a passing fixture and counterexample regressions | **CORRECT for the requested regressions** | `test_accept.py` passes all 12 mutations. Its fixture is not a complete producer-output fixture: notably, its pilot contains only a fictitious stage `x`, and timing attempts are absent. |

**Confirmed new counterexamples**

Every mutation below returned **`all_controls_pass=True`**.

For smk2, I called the actual pure `accept_timing()` function using the committed passing fixture. The retry-hash example was additionally checked with records matching the producer’s retry structure.

| Mutation | Defect |
|---|---|
| Every ordinary split record has an earlier failed hash attempt, a successful final hash, `all_hashes_ok=False`, and `valid=False` | The panel still passes because `panel_validity()` checks final `hashes_ok`, while flag agreement merely confirms that each record is invalid. |
| Injected-fault subject has **all hashes correct**, but drift is 2 and `valid=False` | The identity-fault control passes without demonstrating an identity failure. |
| Doubled times are `(t_single1,t_double,t_single2)=(1,3,2)`, ratio 2, stored drift 1, `drift_ok=False` | Actual anchor drift is 2×; acceptance trusts stored drift and ignores the contradictory flag. |
| Insert a failing equivalence row before the existing passing row for the same subject | Dictionary construction overwrites the failure. |
| Pilot contains only `bank_mesh`, with a positive below-bar peak and completion asserted | No heads, rollouts, level variants, or split-program evidence is required. |
| Append a pilot stage with peak `-1`, or a later NaN, after a finite peak | Checking only `max(stages)` hides invalid individual measurements. `-1` is also the producer’s memory-stat fallback. |
| Set an ordinary split `t_init=-1` and `valid=False` | Invalid numerical evidence is treated as an appropriately invalid repeat and the panel still passes. |
| Set fused stored drift to NaN | Raw drift is checked, but the contradictory stored value is ignored. |

For smk1, I exercised the actual `smk1()` function with an in-memory fixture containing all 84 configured accuracy rows. File reads, completion markers, and expensive numerical calculations were stubbed; these tests establish acceptance-logic failures, not GPU numerical results.

| Mutation | Defect |
|---|---|
| Replace rule report with `groups={}` | Completion alone accepts a rule probe with no evidence. |
| Retain only Hinc `dev6|0` for each scheme: **2 of 84 rows** | Neither the all38 incumbent coverage nor short-head/linear accuracy coverage is enforced. |
| Empty under-resolved control rows, retaining passing summaries | Missing negative-control evidence passes. |
| Empty budget-0 control rows, retaining pooled fraction | The pooled statistic’s evidence can disappear. |
| Set wrong-mesh `raised=[False,False,False]`, retaining `passed=True` | Explicitly contradictory control evidence passes. |
| Set short-head reconstruction/oracle means to NaN | Training output summaries are not checked for finiteness. |

Setting smk1’s accuracy `complete=False` **was rejected**, confirming the new completion check works.

**Producer compatibility**

The current timing expectations match [htime.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/htime.py:99):

- Five unique subjects, including **`1024|A_R384|gauss128|CNR`**, and five equivalence records.
- Two certification cases per subject: **10 certification rows**.
- Four timed B subjects, with two repetitions per dev case and one for the val case: **20 split and 20 fused records**.
- Three doubled-work records for the configured Hinc BE case.
- Certification field names, ratio formulas, label formats, and completion markers agree.

The reference A correctly has certification and equivalence records despite having no B triplets. It has **no configured `qp`/`qpp`**, so a repair must not require successor evidence for A.

[hpilot.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hpilot.py:78) writes the expected `stages[*].mem.peak`, `peak_bytes`, `within_bar`, and completion fields. Its allowance matches acceptance. Stage coverage must follow its actual behavior: it rolls out the **first head of each distinct K**, not every seed or the incumbent separately.

Smk1’s training keys, compact-head dimensions, reproduction identities, Jacobian fixtures, and completion markers agree with `htrain.py`/`hacc.py`. `hrule.py` permits either selection or exhausted-ladder completion; **exhaustion is a legitimate smoke outcome**, but missing groups/trails are not evidence of that outcome.

Unverified tight anchors also remain a reportable smoke outcome under DESIGN A1.3. Their failure alone is **not** a submission blocker; requiring every anchor to converge would overstate the smoke contract.

**Ranked required fixes — blockers only**

1. **Repair smk2 identity and retry acceptance.** At [smoke_accept.py:195](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:195), distinguish legitimate drift-invalid repeats from corrupt numerical evidence and identity failures. Ordinary split retry hash failures must not yield a passing panel. Verify the injected control fails identity, not merely drift. Check retained attempts against aggregate flags without requiring every drift-invalid repeat to pass.

2. **Require smk1 evidence coverage and consistent control summaries.** At [smoke_accept.py:116](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:116), enforce configured accuracy identities and rule groups/trails. Check negative-control case coverage and recompute available summaries, including wrong-mesh exceptions. Permit recorded numerical failures where the smoke explicitly reports outcomes.

3. **Require the actual pilot stages and valid individual memory measurements.** At [smoke_accept.py:221](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:221), bind coverage to the committed pilot configuration/driver contract. Validate every peak before taking the maximum, then recompute the existing allowance predicate. A7’s “measurement probe” qualification does not excuse missing probe stages.

4. **Close remaining timing-control consistency holes.** At [smoke_accept.py:178](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:178), reject duplicate equivalence identities rather than overwriting them. Recompute doubled-work drift from raw times and check its stored flag. Add regressions for these failures and the preceding blockers using producer-shaped fixtures.

**Minor:** reject nonfinite short-head summary metrics and contradictory fused stored drift. Neither is a separate reason for withholding submission.

**Final decision: NO-GO for smk1 → smk2 until the acceptance blockers above are fixed.**