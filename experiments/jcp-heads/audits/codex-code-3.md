**NO-GO for the current smk1 → smk2 submission sequence.** The fixes close several round-2 defects, but smoke acceptance still accepts incomplete evidence, and the submission lock does not enforce the account-wide cap.

Audited HEAD `e70f6ec83`. I made no file changes, submitted no jobs, and used no scratch files. CPU checks ran with bytecode and JAX compilation caching disabled.

**Disposition of the four ranked round-2 fixes**

| Fix | Verdict | Current disposition |
|---|---|---|
| 1. Smoke acceptance | **WRONG — partially resolved** | NaNs in reproduction/Jacobian rows and the original missing-fused-repeat counterexample are rejected. Fixture identity coverage, expected timing-subject coverage, and finite fused ratios remain unenforced. |
| 2. Account-cap race | **WRONG — partially resolved** | A shared lock exists, but other campaign submitters do not use it. The locked queue check also fails open when `squeue` fails. |
| 3. Pilot working-set bound | **NEEDS-RESTATEMENT** | More inputs, outputs and compiled variants are retained. The additional 10% allowance has no demonstrated upper-bound justification. This is a memory probe, not yet a production memory certificate. |
| 4. Holm, qualification contract, rule-probe labeling | **CORRECT for the requested fixes** | Holm decisions are mandatory; the qualification contract now explicitly covers accuracy archives; the duplicated-head probe is labeled as plumbing; the two-of-three unit checks pass. A separate production-only rule-dispatch defect was found in `hscore.py`. |

**Confirmed counterexamples**

1. **Missing evidence can still pass smoke acceptance.** At [smoke_accept.py:65](experiments/jcp-heads/smoke_accept.py:65), reproduction checks enforce counts, but not the expected unique `(rule, case)` or `(scheme, case)` sets. Jacobian checks similarly omit the expected `(case, step)` set.

   At [smoke_accept.py:156](experiments/jcp-heads/smoke_accept.py:156), timing coverage is derived from `result.json` itself. Removing an entire expected subject from the subject map, equivalence records and timing records removes its coverage obligation.

   In-memory checks produced:

   ```text
   duplicate_fixture_rows_accepted = True
   entire_CNR_subject_missing_smk2 = True
   nonfinite_fused_ratios_smk2 = True
   round2_all_fused_missing_rejected = True
   ```

   The nonfinite-ratio fixture retained `valid=True` while replacing ordinary fused ratios with NaN. Acceptance returned true and reported NaN medians. [Lines 169–174](experiments/jcp-heads/smoke_accept.py:169) check panel validity only for split invocations.

2. **The account cap remains unenforced across lanes.** [submit.sh:35](experiments/jcp-heads/cluster/submit.sh:35) explicitly acknowledges that nonparticipating submitters can race the lock. A search of worktree `submit.sh` files found this lock only in the heads lane. Seven queued jobs plus concurrent locked and unlocked submissions can still become nine.

   Additionally, the inner `bash -c` at line 38 lacks `pipefail` and explicit queue-command error handling. A failed `squeue | wc -l` yields zero and proceeds toward `sbatch`. A shell-only reproduction confirmed that branch without submitting anything.

3. **The memory allowance is a margin, not an established bound.** [hpilot.py:88](experiments/jcp-heads/hpilot.py:88) now retains dev6 inputs at both meshes, multiple outputs, tight/tighter K=32 queries and a split program. However, [htime.py:143](experiments/jcp-heads/htime.py:143) also executes autodiff certification, and its arm caches retain additional query variants. The pilot does not reproduce that full compilation/lifetime pattern. Multiplying its peak by `1.10` at [hpilot.py:116](experiments/jcp-heads/hpilot.py:116) does not demonstrate that the omitted allocations fit.

4. **Production-only: `fib4181` silently becomes Gauss-81 offline.** [hscore.py:36](experiments/jcp-heads/hscore.py:36) unconditionally calls `gauss_rule(int(rule[5:]))`. For the registered linear R128 rule `fib4181`, that integer is **81**. An intercepted dispatch confirmed this without allocating rule tables. Offline Q2 would therefore use the wrong quadrature. Current smoke configurations use Gauss rules, so this does not block their execution.

**Per-file verdicts**

“CORRECT” below means correct within the requested smoke scope; it does not predict that the unrun GPU controls will pass.

| File | Verdict | Evidence and failure boundary |
|---|---|---|
| [hacc.py:237](experiments/jcp-heads/hacc.py:237) | **CORRECT** | Four analytic/autodiff Jacobian fixtures run for each head. Per-row comparisons reject NaNs. Both linear reproduction schemes run at lines 284–297; tight/tighter evidence remains archived. The weaker acceptance consumer is the remaining defect. |
| [htime.py:117](experiments/jcp-heads/htime.py:117) | **CORRECT** | Both anchor trajectories, certificates, budget counts, initial-fit records and validity are retained. Lines 233–303 preserve retries, identity results and doubled-work evidence. |
| [htrain.py:235](experiments/jcp-heads/htrain.py:235) | **CORRECT, with a wording qualification** | Compact archive reload/comparison and `COMPLETE` are present. The comparison uses the in-memory emitted parameters; it does **not** reload the emitted pickle, despite the comment at line 241. |
| [hpilot.py:27](experiments/jcp-heads/hpilot.py:27) | **NEEDS-RESTATEMENT** | Independent actual-dimension stand-ins are correct. Retention is improved, but the 10% allowance does not establish the claimed production bound. |
| [hcommon.py:44](experiments/jcp-heads/hcommon.py:44) | **CORRECT** | Compact archives naming another source checkpoint are rejected. IC defaults and their documentation now agree with A1.3. |
| [harm.py:86](experiments/jcp-heads/harm.py:86) | **CORRECT** | Head and linear wrappers enforce allocated schedule length at lines 88 and 161. Initial-fit veto remains present. |
| [hrule.py:75](experiments/jcp-heads/hrule.py:75) | **CORRECT** | `min_pass` is required and bounded. Lines 142–143 apply it; the two-of-three helper tests pass. |
| [hscore.py:33](experiments/jcp-heads/hscore.py:33) | **WRONG for the registered production rule set; corrected round-2 contract** | Accuracy-only contract and certificate provenance are now explicit. `q3_flag` correctly handles the tested threshold, rank deficiency, zero and NaN spectra. The `fib4181` dispatch defect remains production work. |
| [hstats.py:114](experiments/jcp-heads/hstats.py:114) | **CORRECT for the round-2 fix** | Omitting Holm decisions no longer works; missing rung decisions cannot yield the positive at-most-cost label. |
| [smoke_accept.py:65](experiments/jcp-heads/smoke_accept.py:65) | **WRONG** | Count-only fixture coverage and self-declared timing coverage permit omissions; fused ratios lack independent finiteness checks. Original NaN-max and wholly invalid fused-subject examples are fixed. |
| [test_h.py:60](experiments/jcp-heads/test_h.py:60) | **CORRECT, limited scope** | Entire suite passes, including two-of-three, mandatory Holm and production Q3-classifier fixtures. It does not cover the remaining acceptance counterexamples. |
| [ledger.py:82](experiments/jcp-heads/ledger.py:82) | **CORRECT for requested retry/pending fixes** | Base and retry exclude the same pending reservation; empty mandatory inventory is rejected. Failed attempts with saved pending reservations restore them at lines 165–166. Checks again returned **4 hours for both `train` and `trainr1`**. |
| [cluster/submit.sh:38](experiments/jcp-heads/cluster/submit.sh:38) | **WRONG** | Shared-lock participation is incomplete; inner queue failure permits submission. Lane serialization and one-smoke-at-a-time checks are present. |
| [cluster/stage.py:45](experiments/jcp-heads/cluster/stage.py:45) | **CORRECT for smoke staging** | Two-hour limit, committed-blob staging, manifests, GPU partition/preflight and f64/highest settings remain present. Smoke staging bypasses production certificates. |
| [cluster/collect.py:47](experiments/jcp-heads/cluster/collect.py:47) | **CORRECT for requested archival behavior** | Original outputs, including pickles, are verified remotely. Pickles are excluded from collection; original and filtered manifests are retained; local verification uses the filtered manifest. |
| [make_configs.py:43](experiments/jcp-heads/make_configs.py:43) | **CORRECT for smoke generation** | In-memory regeneration exactly matched all five checked-in smoke JSONs. |
| [configs/smk1train.json](experiments/jcp-heads/configs/smk1train.json) | **CORRECT** | Identity plus short K=4 and K=32 training arms. |
| [configs/smk1acc.json](experiments/jcp-heads/configs/smk1acc.json) | **CORRECT** | Incumbent all38/both schemes, short heads, linear arms and both reproduction fixtures. |
| [configs/smk1rule.json](experiments/jcp-heads/configs/smk1rule.json) | **CORRECT as a plumbing probe** | Duplicate archive usage is now explicitly labeled in acceptance output; independent-seed evidence is not claimed. This resolves the round-2 restatement. |
| [configs/smk2time.json](experiments/jcp-heads/configs/smk2time.json) | **CORRECT** | Normal subjects, cohort-specific repeats, deliberate identity fault and doubled work are configured. |
| [configs/smk2pilot.json](experiments/jcp-heads/configs/smk2pilot.json) | **CORRECT** | Twelve independent actual-K stand-ins plus incumbent and linear ladder. The certification limitation lies in pilot execution and interpretation. |

`resources.py` is **not a smoke dependency**. Its rewrite and production configuration audit remain deferred as requested.

**Ranked remaining required fixes**

1. **Before smk1 submission:** make account-wide check-and-submit exclusive across participating campaign lanes or a single submission authority. Fail closed on queue-query errors.
2. **Before accepting either smoke:** bind coverage to the committed smoke configurations; require exact unique fixture and subject/case identities. Validate finite positive fused measurements and their identity/drift evidence independently.
3. **Before using smk2 to authorize 4096² production:** reproduce the timing driver’s missing allocation/compilation pattern or substantiate a conservative byte bound. Until then, label the pilot a measurement probe.
4. **Before production scoring:** dispatch `fib4181` to the correct independent quadrature implementation.

**Final decision: NO-GO for smk1 on A100-80G followed by smk2 on H200 as currently implemented.**