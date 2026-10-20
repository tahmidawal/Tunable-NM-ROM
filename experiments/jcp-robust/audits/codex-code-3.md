**Overall verdict: NEEDS-FIX — not ready for smoke.**

Audited `f0d5e2735`. No files changed, scratch created, jobs submitted, or SSH/cluster commands run. Checks used CPU-only Python with bytecode writing disabled. Paths below are relative to `experiments/jcp-robust/`. **BLOCKING** means fix before the smoke.

**Round-two dispositions — original numbering preserved**

**1. NOT RESOLVED — WRONG — BLOCKING: 3D target-check disagreement still bypasses integrity precedence.**

`e1analyze3d.py:81–84` folds offline/in-job target-check disagreement into `tc_ok=False`, but never adds it to `mismatches`. Lines `138–139` then permit own-family inference, while `212` can still report integrity OK. This confuses corrupted evidence with an uncertified target.

The corresponding 2D disagreement handling and diagnostic G-cert separation are repaired (`e1analyze.py:102`, `140–145`).

**Fix:** Record target-check disagreement independently as an integrity failure before applying target-unavailable fallback.

**2. NOT RESOLVED — NEEDS-FIX — BLOCKING: registered populations and certificate completeness remain incompletely enforced.**

The new 2D production population registry and duplicate-production-row detection help (`e1analyze.py:31–40`, `74–78`, `118–122`). Remaining holes:

- Certificate length is compared with the output’s own `steps`, rather than an independently derived expected count (`e1analyze.py:122`).
- Target-check cases still come from supplied configuration metadata (`e1analyze.py:101`).
- 3D tighter cases come from `cf['tighter_cases']`, including production (`e1analyze3d.py:89`, `114`).
- 3D stationarity checks row count but neither unique case identities nor expected certificate length (`e1analyze3d.py:117–119`).
- The three-draw smoke unconditionally fails the 16-draw check (`e1analyze3d.py:90–91`).

**Fix:** Define independent production and smoke contracts covering exact cases, draws, anchors, unique identities, and expected steps.

**3. RESOLVED — CORRECT — MINOR: the identified target-margin and zero-eligible-case defects are repaired.**

`e1analyze3d.py:149–157` computes target contributions per case and initializes `econv` when no eligible cases exist. The 2D path preserves per-case target availability (`e1analyze.py:106`, `229`).

**Fix:** None for these specific defects.

**4. NOT RESOLVED — NEEDS-FIX — BLOCKING: normalization improved, but missing/non-finite evidence still has unsafe paths.**

`e1analyze3d.agree()` now handles `None`, and diagnostic indicators use `E.num()`. However:

- Missing output/archive files can still raise uncaught `FileNotFoundError`; the handler only catches `KeyError`, `ValueError`, and `IndexError` (`e1analyze3d.py:184–185`, `218–220`).
- The 3D E1 reproduction gate checks archive finiteness but not new-output finiteness (`195–199`). A later NaN in `rows` can be ignored by Python’s `max`.
- The smoke’s 3D replay repeats the unguarded reduction (`smoke_accept.py:153–154`).

**Fix:** Validate shapes and finiteness on both sides before reductions; convert absent evidence into explicit failed/unavailable checks.

**11. NOT RESOLVED — NEEDS-FIX — BLOCKING: reproduction is not integrated into a working smoke acceptance path.**

Production E2 3D replay now checks finite arrays and matching shapes (`e2analyze3d.py:195–203`). Nevertheless:

- `smk1` cannot pass its 3D integrity check because of item 2.
- `smk2` acceptance resolves trained-bank paths against the source checkout through `e2bank.use()`, not against the supplied collected output directory (`smoke_accept.py:118`; `e2bank.py:22–24`). The configs’ `../../output/train2d_s1/...` paths are correct inside the staged job, but point elsewhere during local acceptance.
- `smk3` uses a hard-coded archive without checking its owning-lane hash (`smoke_accept.py:147–154`).

**Fix:** Resolve collected artifacts explicitly, verify their recorded hashes, and provide smoke-specific replay acceptance without weakening production populations.

**12. NOT RESOLVED — WRONG — BLOCKING: resource certification still cannot cover the complete lane.**

Block-specific 3D populations and `dtime` inventory support are added. Remaining defects:

- `certify.INV` has no `e5time` handler (`certify.py:124`); certification of an E5 production step fails at `152`.
- Certificates record a commit (`certify.py:181–186`), but staging never verifies that commit against the staged code (`cluster/stage.py:64–83`). A certificate survives subsequent code changes.
- Smokes still carry neither execution inventory nor resource certificate (`cluster/stage.py:105–107`), despite A4.2’s every-job requirement.
- Collection omits staged resource certificates, inventories, and `RESERVATION.json` from its explicit tar list (`cluster/collect.py:63–64`).

**Fix:** Inventory E5, enforce code/certificate binding, supply the smoke inventory/reservation contract, and retain these artifacts through collection and cleanup.

**17. RESOLVED — CORRECT — MINOR: empty fixture admission is rejected.**

`establish_e1.py:63–64` requires exactly six entries in all three inputs. CPU probe: `admit([], [], [])` returns `admitted=False`.

**Fix:** None.

**21. RESOLVED — CORRECT — MINOR: target-free own-family qualification works.**

`e1stats.py:130–145` evaluates own-family evidence before requiring a target. A passing target-free CPU fixture returned:

```text
(None, True, 'target not certified')
```

**Fix:** None for the original defect.

**22. RESOLVED — CORRECT — MINOR: the identified H-acc-r stationarity/anchor omissions are repaired.**

The 2D predicate now includes configuration qualification and tighter-anchor evidence (`e1stats.py:152–163`). The 3D predicate includes pooled stationarity, validity, and tighter-anchor agreement (`e2analyze3d.py:48–56`).

Population and independent evidence-validation deficiencies remain under items 2 and 24.

**23. NOT RESOLVED — WRONG — BLOCKING: timing evidence can still enter fronts after acceptance failures.**

The 3D path now requires `tm['valid']`, and both dimensions require repeated observations per case. However:

- The 2D path collects surviving `valid` invocations without enforcing the subject-wide identity veto, split/fused equivalence, `counts_complete`, or memory rejection (`e2analyze.py:95–110`). `dtime` can emit `complete=True` while recording memory rejection (`dtime.py:327–343`).
- The 3D front does not bind the timing bank to the analyzed bank (`e2analyze3d.py:143–154`).
- Neither path independently reconstructs the complete timing acceptance contract before front assignment.

**Fix:** Validate the whole subject/panel, including all-attempt identity failures, expected unique repetitions, memory rejection, and checkpoint/rotation identity. Then assign costs.

**24. NOT RESOLVED — WRONG — BLOCKING: 3D selection’s distance agreement test accepts missing serialized evidence.**

`select3d.py:95–100` uses:

```python
abs(off - num(job)) > 1e-9 * abs(num(job)) + 1e-13
```

For finite `off` and `job=None`, this becomes `inf > inf`, which is false. The CPU probe confirmed that the disagreement is missed. The same pattern appears in `e2analyze3d.py:108`.

Production/tight/successor coefficient arrays also remain positionally compared without validating all their saved case mappings (`select3d.py:68–102`). The new job/backend/memory gating is an improvement.

**Fix:** Use a finite-aware agreement predicate and validate exact case identities for every participating array before comparing it.

**25. RESOLVED — CORRECT — MINOR: selected/deployed aliases are retained.**

`e2analyze.py:164–177` now supplies both identities when selected and deployed rules coincide, and records `alias_of` (`42`).

**Fix:** None.

**26. NOT RESOLVED — NEEDS-FIX — BLOCKING: trained-bank execution is added, but the promised selection-to-verdict smoke is incomplete.**

`smk2t/u` now load the emitted checkpoint and rotation. `smk3q` registers and executes `W1024pilot`; these repair the incumbent-only execution defect.

However:

- No smoke invokes the actual selection procedure and feeds its result into evaluation.
- `smk2u` hard-codes `fib4181` rather than consuming the selection result.
- Acceptance computes qualification labels but never exercises the promised H-acc-N calculation (`smoke_accept.py:114–122`).
- The pilot-bank 3D rung pair uses four cases, rather than the eight specified in §6 (`configs/smk3q.json`, third block).

**Fix:** Exercise selection and its downstream configuration/verdict path using an explicit smoke population, including legitimate “none/unavailable” outcomes.

**27. NOT RESOLVED — WRONG — BLOCKING: `smoke_accept.py` exists but is not a reliable acceptance gate.**

Beyond items 2 and 11:

- `smk4` calls `e5analyze` with `--expect A100-80G=1` (`smoke_accept.py:163`). When that job is valid, `analyze()` enters the all-complete branch and requests H200 quantities (`e5analyze.py:537–540`). An in-memory fixture reproduced **`KeyError: 'H200'`**.
- Conversely, a failed/blocked E5 job can produce a normal analysis report; smoke acceptance checks only process success and then records its summary as unconditionally OK (`smoke_accept.py:165–168`).
- `main()` never exits nonzero when `ALL=False` (`172–181`).

**Fix:** Implement a single-job smoke analysis mode, assert each required control/gate explicitly, and return failure status when acceptance fails.

**Additional findings**

**28. WRONG — BLOCKING: production E2 3D analysis cannot decode the second job’s widths.**

The generator splits W256/W512 into job `a` and W768/W1024 into job `b` (`make_configs_e2_3d.py:50–56`). Analysis installs only job `a`’s settings (`e2analyze3d.py:83`), and `e2score3.install()` replaces the entire decoder map (`e2score3.py:29`).

The CPU probe confirmed that this map causes `SC.glat('W768')` to raise **`KeyError: 'W768'`**. Thus the 512→768 transition cannot complete.

**Fix:** Install a validated union of both jobs’ settings and bank identities, or install the owning job’s context before each decode.

**29. WRONG — BLOCKING: target-unverified diagnostic picks enter selected-rule robustness counts.**

Both `keys_2d()` and `keys_3d()` convert any non-null selected rule into a definitive `rule:...` label without inspecting its diagnostic outcome (`e2verdicts.py:50–51`, `69–71`). A2.5 explicitly excludes target-unverified picks from selected-rule robustness counts.

**Fix:** Preserve the diagnostic rule for reporting, but emit a non-definitive state for cross-family selected-rule comparisons.

**30. WRONG — BLOCKING: E5 reproduction gates do not govern E5 acceptance or verdicts.**

- `load_job()` ignores P-cross `G-RR`, `G-FR`, and `A-valid` (`e5analyze.py:100–115`). The original copied `c2analyze.py:98` requires them.
- An in-memory job with all three gates false still loaded with `status='ok'`.
- Blocking **G-E5.1** is calculated only after pooled verdicts are constructed and never changes them (`e5analyze.py:583–598`). Omitting `--p1ref` skips it entirely.
- G-E5.1 compares whichever score files and intersecting cases exist, rather than requiring the registered complete set (`413–433`).

**Fix:** Require complete, finite, pinned reproduction evidence before marking a job acceptable or constructing its verdicts. Propagate failures through A4.1.

**31. WRONG — BLOCKING: the 3D smoke memory gate can omit required evidence and never checks host headroom.**

`smoke_accept.py:130–132` simply omits the training memory check if `run_record.json` is missing. It never checks host RSS × 1.25 against requested memory, although the design requires that gate and training records host RSS (`train3d/run_seed.sh:99`). The rung checks also accept empty validity lists via `all([])` (`smoke_accept.py:144`).

**Fix:** Require the record, finite device and host measurements, the registered headroom inequality, and exact nonempty rung populations.

**32. CORRECT — MINOR: the inspected E5 timing mechanics and source copies are consistent.**

All **35** E5 provenance entries matched their hashes. The copied `p1time.py` and `c2time.py` also matched their pinned historical source blobs. P-cost calls the unchanged P1 driver; P-cross retains synchronized A–B–A timing, preceding burns, and retry mechanics (`e5time.py:221–237`). The pure CPU drift-control test passed, as did injected/null estimator fixtures on an in-memory job.

These checks validate the inspected mechanics; they do not repair the acceptance failures above.

**Smoke staging and control coverage**

Static staging checks passed for all four specs. All **229** staged source files matched HEAD; referenced configs/scripts are included, shell-step markers match collection, and the inspected E5 imports are covered. The dirty files currently present are excluded by staging. **I found no immediate staging rejection.**

That establishes staging feasibility, not successful cluster execution, resource fit, or end-to-end acceptance.

| Smoke | Execution coverage present | Controls/gates it cannot currently demonstrate end to end |
|---|---|---|
| `smk1` | E1 2D/3D draws, bad-vector fixtures, identity rejection, permuted replay | Unified acceptance of **C-E1.1, C-E1.2, G-E1.1–3** is blocked by the 3D population defect. The §6 H200 E5 panel is absent, so this job cannot demonstrate **C-E5.1/G-E5.1**. |
| `smk2` | Incumbent replay, truncated training, rotation, trained-bank evaluations | **G-E2.1 selection run path** is not exercised; no selection-fed verdict calculation. Trained-bank acceptance has the collected-path defect. **G-E2.3** replay is present but does not make the whole smoke acceptable. |
| `smk3` | Training pilot, heavy incumbent evaluation, pilot-bank rung pair | Complete **memory gate** lacks host-headroom enforcement and mandatory records. Pilot-bank selection is not exercised. **G-E2.3** replay lacks pinned, finite-safe acceptance. |
| `smk4` | A100 P-cost/P-cross, doubled work, five cases × two reps, environment records | Valid-job analysis crashes in single-type mode. Acceptance does not require **C-E5.1**, real-data **C-E5.2** fixtures, or **G-E5.1** to pass. **C-E5.3** has a passing CPU test. **G-E5.2** cannot be demonstrated because no H200 E5 smoke runs. |

The submission lock, reservation checks, and completion-marker collection safeguards remain present. The blocking issues are the acceptance paths, evidence precedence, production decoder mapping, and incomplete certification—not an untracked-file staging failure.