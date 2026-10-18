**NEEDS-FIX.** Fix the job-side evidence loss and incomplete pilot coverage before launching. The remaining scorer and resource-estimator defects can be repaired after collection without repeating otherwise valid GPU runs.

Audited HEAD: `32e8c5816c0bca786a619c2480e7d01946a29e0c`. No files changed, jobs submitted, or lab-log writes. Probes used CPU Python with `-B`. Locations below are relative to `experiments/jcp-p1-scaling/`.

**Round-2 dispositions**

| Blocker | Disposition | Evidence |
|---|---|---|
| 1. Smoke acceptance `TypeError` | **RESOLVED** | `smoke_accept.py:23–25` renames conflicting information keys; kernel acceptance also no longer expands `ce`. Replayed `req(..., passed=True)` successfully. |
| 2. Qualification false passes | **PARTIALLY** | Population, principal shapes, finite states, mandatory distance comparisons, q++ validity and late disagreements are checked at `p1score.py:107–214`. The original malformed fixture now fails. Independent rho remains optional at `:166`; decoded-field disagreement is calculated at `:133` but excluded from qualification. |
| 3. Required smoke predicates | **PARTIALLY** | GAL order now requires `True` (`smoke_accept.py:151`); controls require populations; fused identities and the new subject veto are checked (`:182–190`). Doubled-work retries still discard earlier attempts (`p1time.py:271–290`). |
| 4. Submission/accounting safety | **PARTIALLY** | Lane submission locking and ambiguous-reservation retention are implemented (`cluster/submit.sh:13,25–38`); prior requeue elapsed is counted (`ledger.py:49–50,107`). However, `settle` releases on an empty accounting response without checking the live queue (`:75–87`). Certificate binding covers configurations/run list, but not source commit (`cluster/stage.py:45–49`). |
| 5. Timing qualification consumer/schema | **PARTIALLY** | The missing arrays and identities are now saved (`p1time.py:118–187`), and timing invokes `qualify()` (`smoke_accept.py:206–216`). That invocation supplies no independent rho tables; timing E0 scoring and the accuracy/timing qualification conjunction remain absent there. These consumer omissions are offline-fixable. |
| 6. Reproduction/scoring identity | **PARTIALLY** | Manifest exists; configuration/code hashes matched current files. Wrong rotation now fails; reproduction requires `job` (`p1score.py:245–283`). Several advertised numerical-identity fields still are not compared against the actual run. |
| 7. Array substitution guard | **PARTIALLY** | Ordinary replacement of Gq/Gs/Psi now changes the fingerprint (`p1common.py:177–196`). A balanced substitution at unsampled entries still produces exactly the same fingerprint. |
| 8. Complete pilot footprint | **PARTIALLY** | All configured meshes/settings, split evolution, burn-in and unknown-peak handling are added (`pilot.py:51–111`). Successor/check executables and several timing execution paths are still not exercised. |

**Ranked remaining findings**

1. **Job-side: the smoke’s finest order anchor silently overwrites half its per-step evidence.**

   `p1core.py:48` fixes `NSTEP=800`, while the configured `anchor2=.03125` produces **1,600 steps** at `dt=.005 × .03125`. `p1acc.py:278–283` runs that anchor for both forms.

   Writes clamp to slot 799 at `p1core.py:216–219`. Steps 799 through 1598 are overwritten. `run_valid()` slices the resulting 800-element buffer as though it contained the full trajectory (`p1common.py:303–306`) and ignores `worst_ratio`.

   Read-only reproduction using the exact clamped-write rule: insert a NaN ratio at step 1000, then finite later ratios. The retained buffer is finite and **`run_valid()` returns `True`**. This demonstrates lost validity evidence, not that the actual anchor necessarily becomes nonfinite.

   Size the evidence buffer for the longest configured schedule and assert capacity before execution. This cannot be repaired offline from the order files: `p1acc.py:311` saves coefficients, not the missing step evidence.

2. **Job-side: doubled-work retries destroy evidence needed by the subject-wide identity veto.**

   `p1time.py:271–284` retries drift failures, but appends only once, after the retry loop (`:285–290`). Timings, iteration arrays and identities from earlier attempts disappear.

   Consequently, an early attempt can have an identity failure and a later attempt can pass. The resulting row says `identity_ok=True`; `p1stats.identity_veto()` sees only that final result (`p1stats.py:309–310`). The newly added veto works on supplied failures, but cannot recover discarded ones.

   Persist every doubled-work attempt and aggregate identity failures across all attempts, as ordinary split triplets already do. This is a GPU-evidence issue, not merely an acceptance-script issue.

3. **Job-side, H200 pilot: its “full resident footprint” claim remains unsupported.**

   The pilot builds successor/check tables but runs only:
   
   - main-rule recording split evolution and certification;
   - main-rule non-recording tight fused queries;
   - one historical-rule query and Q3 Jacobian.

   See `pilot.py:75–86,99–105`. Production timing also executes q+/q++ fused queries, recording fused queries, separately compiled decode, and component kernels (`p1time.py:130,147,160–170,254,293–308`). These create executable/workspace demands the pilot never measures. Pilot input residency is also different: one case per mesh plus 38 fields at 4096², versus timing’s six cases per mesh.

   The added accuracy extras may provide substantial headroom, but no demonstrated bound establishes that they cover the omitted executable peaks. **This is not proof of an OOM**; it is a remaining failure to certify the advertised workload. Exercise the actual execution-path inventory before using its peak to approve production.

4. **Offline-only: qualification still permits false passes, including on the timing consumer’s actual call path.**

   Two fresh CPU fixtures reproduced:

   ```text
   no independent rho still qualified: True
   decode mismatch qualified: True; decode_rel = 0.5
   ```

   The fixtures used a stubbed independent decoder to isolate qualification logic.

   - `p1score.py:166` silently skips independent rho when tables/cases are omitted. `smoke_accept.py:213` makes precisely that call for timing.
   - `p1score.py:133` computes reconstruction disagreement but never adds it to disagreements or the final predicate. Accuracy smoke checks it separately; timing smoke does not.
   - Shape checks at `:110–113` still omit finer-replay arrays, validity arrays and rho.

   Require the independent comparisons in the qualification contract, include decoded-field agreement in the result, and complete timing scoring/conjunction. The saved timing arrays generally suffice; these repairs do **not** require another GPU run.

5. **Offline-only: `resources.py` cannot consume the rewritten pilot output.**

   `resources.py:87` expects `level`, `call` and `seconds`. Current pilot subject rows contain `evolve_seconds`, `cert_seconds` and `tight_seconds`, with no `level` or `call` (`pilot.py:87–88`). Replaying the resource comprehension against that schema raises:

   ```text
   KeyError: 'level'
   ```

   The historical-extra row also lacks `Rp` and `scheme` (`pilot.py:106`).

   Additionally, the advertised per-run ratio would compare unlike measurements unless corrected: accuracy `seconds` includes the guarded fused query, while pilot `evolve_seconds` measures split evolution only.

   Repair the estimator against the new schema and define a comparable cost measurement. Existing pilot timings remain usable for a revised estimate.

6. **Guard/accounting weaknesses remain, although they do not establish an inevitable smoke failure.**

   - **Fingerprint collision:** for a positive `(100,100)` Gq array, adding `0.5` at flattened index 1 and subtracting `0.5` at index 3 preserves both full reductions and the sampled checksum. `fingerprint()` returns the same hash. The ordinary round-2 substitution is caught, but the docstring’s claim that any same-shaped substitution changes the fingerprint is false (`p1common.py:179–189`).
   - **Reproduction identity:** wrong rotation now returns `equal=False`. However, a fixture carrying `budget=0`, `form='GAL'`, wrong metric and wrong weighting still returns `equal=True`. `manifest_equal()` checks constants in the development manifest rather than those actual-run fields (`p1score.py:266–268`). Some fields are not recorded by the producer at all.
   - **Ambiguous settlement:** an empty successful `sacct` response is treated as proof that no job exists (`ledger.py:78–87`). An accounting delay can therefore undo the improved submission trap. Keep the reservation unless reconciliation establishes absence safely, including a live-queue check.

The original round-2 qualification fixture now returns **Q1 false, Q2 offline-disagreement failure, qualified false**; empty gamma arrays also fail. Both NaN-distance implementations now return **infinity**. Selected read-only statistical/validity tests passed, including 69 Wilcoxon comparisons.

**Launch decision:** repair findings **1–2 before the real-size smoke**, and **3 before treating the H200 pilot as a production-footprint certificate**. Findings **4–5 are offline repairs**, so they alone would not justify withholding otherwise sound smoke measurements.