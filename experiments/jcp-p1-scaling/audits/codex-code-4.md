**READY-FOR-SMOKE.** I found no remaining job-side blocker in the specified A100 smoke or H200 pilot paths. Offline acceptance and accounting gaps remain; this verdict does not authorize production claims.

Audited HEAD: `b147edabb596b8ca92a4fe32ba01813d54f8fa3d`. Locations below are relative to `experiments/jcp-p1-scaling/`.

| Round-3 finding | Disposition | Evidence |
|---|---|---|
| **1. Order-anchor evidence overwritten** | **RESOLVED** | `p1core.py:166–170,217–220` allocates and indexes using `nstep`; `p1common.py:275` checks capacity. `p1acc.py:273–275` supplies 1,601 slots for the 1,600-step anchor. |
| **2. Doubled-work retry evidence discarded** | **RESOLVED for the identity veto** | `p1time.py:284–294` retains each attempt’s timings and checks, aggregating failures across attempts. `p1stats.py:309–310` consumes that aggregate. Earlier attempts’ raw iteration arrays still are not retained—only their check results; the final attempt’s arrays are saved at `p1time.py:292`. |
| **3. Incomplete pilot execution footprint** | **RESOLVED for the listed omissions** | `pilot.py:58–64` retains six dev inputs per mesh. `:85–106` exercises split evolution, certification, tight fused replay, separate decode, recording fused query, successor/check queries and component kernels. Accuracy extras remain at `:108–125`. Actual memory sufficiency still requires the pilot measurement. |
| **4. Qualification false passes / incomplete timing consumer** | **PARTIALLY** | Missing independent rho evidence now raises (`p1score.py:158–159`); decode disagreement fails qualification (`:140–141`); successor/check/rho shapes are checked (`:160–164`). Timing supplies independent tables (`smoke_accept.py:212–214`). Timing E0 scoring and the accuracy/timing qualification conjunction remain absent. |
| **5. Resource-estimator schema mismatch** | **RESOLVED** | `resources.py:87–90` consumes the pilot’s current schema and compares matching recording fused queries at 4096² and 1024² on the same H200. Historical-extra rows are safely excluded. |
| **6. Fingerprint, reproduction identity and settlement weaknesses** | **PARTIALLY** | Fingerprint limitations are now stated accurately (`p1common.py:186–190`). Recorded form, level, tolerances, budget, weighting and time tag are compared (`p1score.py:278–281`). Settlement checks both accounting and the live queue (`ledger.py:77–88`), but the residual gaps below remain. |

Remaining issues, ranked by operational impact:

1. **Accounting: two empty responses still release an ambiguous reservation.**  
   `ledger.py:89–90` treats successful-but-empty `sacct` and `squeue` responses as proof the job never existed. A completed job absent from the live queue while accounting is delayed remains indistinguishable from a failed submission. Retain the reservation until reconciliation establishes its disposition. This does not invalidate smoke measurements, but can undercount allocation usage.

2. **Offline: timing acceptance still lacks the full qualification conjunction.**  
   `smoke_accept.py:215–217` requires offline agreement and reports Q1/Q2, but does not E0-score timing fields or combine timing qualification with accuracy-job qualification. Saved fields and replay arrays permit repair after collection; another GPU run is unnecessary.

3. **Offline: reproduction identity still ignores the recorded target metric.**  
   `p1acc.py:205` records `metric_id_target`, but `p1score.py:264–281` never compares it. Altering only that field cannot affect `manifest_equal()`. Compare it with the pinned scoring metric before accepting reproduction identity.

4. **Evidence completeness: doubled retries retain predicates, not every iteration array.**  
   `p1time.py:284–286` preserves sufficient evidence to veto any failed attempt, fixing the launch blocker. Saving the four iteration arrays inside each attempt would also preserve independent diagnostic evidence.

Validation was read-only: all top-level Python sources parsed; CPU qualification probes rejected missing rho, decode mismatch and malformed rho shape; validity tests and 69 Wilcoxon comparisons passed. No GPU execution, file changes, job submissions or lab-log writes were performed.