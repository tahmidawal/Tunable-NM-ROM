**NEEDS-FIX. Do not launch the real-size smoke or H200 pilot through the current submission path.** Several round-1 repairs are effective, but smoke acceptance currently crashes, qualification can falsely pass, required controls are weakened, and an ambiguous submission failure can release a live job’s reservation.

All locations below are relative to `experiments/jcp-p1-scaling/`. Audit was read-only: no file changes, jobs, or lab-log writes. CPU probes used the absolute environment, `-B`, and `JAX_PLATFORMS=cpu`. Observed HEAD: `626463209`.

**Ranked blockers**

1. **Smoke acceptance crashes on the kernel-equivalence result.**

   `smoke_accept.py:192` calls `req(..., **ce)`, where `ce` already contains `passed` from `p1kern.py:142` or `p1three.py:208`. `req()` then constructs `dict(required=True, passed=bool(cond), **info)` at `smoke_accept.py:24`.

   Reproduced:
   ```text
   TypeError: dict() got multiple values for keyword argument 'passed'
   ```

   Every normal kernel result reaches this failure. Remove the duplicate keyword and test the checker against the actual output schema before spending GPU time.

2. **`p1score.qualify()` does not enforce the complete Q1/Q2 contract.**

   - **Population/completeness:** `p1score.py:66–108,113–137` trusts supplied case lists and array shapes. It never verifies the expected configuration population, six dev cases, six output states, or 50 certified steps per case.
   - **Validity:** `:85–86` independently checks only W and reason 3, then trusts saved validity flags. It ignores saved states, initial coefficients, decoded-field finiteness and recorded-ratio finiteness.
   - **Offline agreement:** missing rows silently skip checks at `:88–90,100–102,121–123`. `dist_qp_qpp` is never compared with its in-job value. **ρ is read directly from `R['rho']`, not independently recomputed** at `:124`.
   - **Late disagreement:** Q1’s `offline_agrees` and `pass` are finalized at `:104–110`; Q2 discrepancies are appended later. Neither Q2’s label nor `numerically_qualified` incorporates those later discrepancies.
   - **Invalid finer replay:** `valid_qpp`, which `p1acc.py:188` saves, is never checked.
   - **NaN distance:** `npcheck.py:112` and `p1common.py:271` use Python `max()` over distances. A NaN after a finite first value can be silently ignored.

   A small CPU fixture returned **Q1 pass, Q2 pass, numerically qualified=True** with only one dev case, no comparison rows, NaN internal states and `valid_qpp=False`. Replacing gamma arrays with shape `(1,0)` still yielded Q1 pass. A trajectory with NaN at output 3 produced distance **0.0**.

   Require explicit expected populations/shapes, finite inputs and positive finite normalization; verify all required comparisons; independently recompute ρ; include q++ validity and every agreement result in the final labels.

3. **Required smoke controls do not consistently enforce their registered predicates.**

   - **CN-R order cannot fail acceptance:** `smoke_accept.py:123` tests `o['GAL']['passed'] is not None`. The producer always returns a Boolean (`p1acc.py:308`), so **False passes this required check**.
   - **Doubled-work drift is ignored:** `p1time.py:256–277` records `drift_ok`, but implements no retries or combined validity decision. `smoke_accept.py:162–163` ignores drift and accepts a pooled median ratio. Out-of-band triplets can pass.
   - **A4 subject-wide blocking remains absent:** `smoke_accept.py:151–158` exercises all-invalid synthetic arrays through `h_cost_n`; it does not exercise a subject-wide identity veto. No `analyze.py` exists. One identity failure among otherwise valid triplets can therefore evade the required subject-wide block in downstream statistics.
   - **Fused identity:** the producer checks it (`p1time.py:247–249`), but smoke acceptance only checks split `invocations`, never `fused`.
   - **Required-set completeness:** empty `equivalence`, empty certification rows, empty vendor/sync dictionaries, or an empty doubled-work list can skip or vacuously pass required controls (`smoke_accept.py:120,144,159,166,186`). An empty CPU-results dictionary passes `:202`.
   - **Budget-zero population:** `:112` says “300” in its label but tests whatever pooled size is supplied.
   - **Under-resolution:** the independent own-state ρ calculation is present, but agreement uses truncating `zip()` without shape/population assertions (`:102`). The rollout failure trusts an in-job Boolean; saved `W_succ` is not independently decoded and compared.
   - **Zero-update:** the distance check is present, but acceptance does not require the “verified tight trajectory” to be valid before using it (`:83–90`).

4. **Submission failure handling can still leave a live job unreserved.**

   `cluster/submit.sh:25` releases the pending reservation on **any** error. If `sbatch` succeeds remotely but SSH loses its response or times out at `:28`, the trap releases the reservation although the job exists. This recreates the round-1 budget violation.

   Retain the reservation on ambiguous submission outcomes and reconcile by a unique submission identity before release.

   Additional enforcement gaps:

   - Queue-cap checks happen before the atomic reservation and outside its lock (`:17–24`); simultaneous distinct submissions can exceed lane/account concurrency caps.
   - `ledger.py:45–46` counts elapsed time only after terminal state. Consumed earlier requeue attempts are omitted while the current attempt remains active.
   - `ledger.py:87–88` charges raised limits retrospectively; it provides no pre-increase reservation check as A2.6 requires.
   - `cluster/stage.py:42–46` checks only an unbound scalar projection. It does not bind the certificate to the staged commit, configurations and complete run list.

5. **Timing-job qualification still has no executable end-to-end consumer.**

   `p1time.py:118–179` saves useful arrays and certification rows, but its `S_*.npz` schema lacks fields required by `p1score.qualify()`: `f257`, `dev_case_ids`, `q2_case_ids`, and validity arrays. Smoke merely checks stored `gcert_ok`/`valid` flags at `smoke_accept.py:164–166`.

   Consequently, the smoke does not demonstrate A2.3/A3.2 timing-job Q1/Q2 labels, independent agreements, E0 scoring of timing trajectories, or the conjunction with the accuracy-job label. The producer also omits available F and w0 from its validity call at `p1time.py:140`.

6. **Reproduction and scoring identity remain incomplete.**

   `p1score.py:165–177` calls its check numerical-manifest equality, but omits rotation hash, ordered mode identities, rule-point/weight hash, scheme, dt, form, tolerances, budget, initialization, weighting, output times and metric. A CPU probe with `rotation_sha256='WRONG'` returned **equal=True**. At `:189`, omitting `job` bypasses manifest equality entirely.

   `score()` checks case/time strings and the restricted S-use rule, but does not assert the registered metric or expected store version; comparing the returned version with the currently loaded store is not pinning it to the rollout manifest. It does not attach the required reference hash/time identity to the returned record. `score_config()` manufactures time tag and dt rather than reading recorded rollout identity (`:160`).

   The explicit swap controls catch their three constructed metadata mutations. They do not establish the complete DESIGN §3 identity contract.

   I found neither `manifest.py` nor the required pre-job `MANIFEST-dev.json` among the lane files searched. The combined-archive summary is not that development manifest.

7. **The array guard is only a partial substitution guard.**

   `p1common.py:177–180` fingerprints eigenvalues, **only the first column of A**, and array element counts. It does not fingerprint Gq, Gs, Psi or the remaining A columns.

   A CPU probe replacing Gq, Gs and Psi with arbitrary same-shaped values produced the **same fingerprint**. The new negative control changes lambda (`p1acc.py:419`), so it demonstrates only the portion now protected. Direct split calls still bypass `run_query()` and its recording-size assertion.

8. **The pilot is improved but still does not establish the complete timing footprint.**

   `pilot.py:63–96` now retains its settings, runs tight BE, and executes certification/Q3. However:

   - It creates settings only for the pilot mesh, while production timing retains settings across all meshes, including additional nominal and R512 subjects.
   - It exercises fused queries, not the separately compiled split evolution executable timed by P1c.
   - `configs/pilot.json` omits some retained successor/check settings, including q++ for R128/R256.
   - Its measured blocks lack the mandated burn-in.
   - Missing memory statistics become `peak=-1` in `p1common.py:63–69`, which `pilot.py:100` treats as below the threshold.

   These do not prove an OOM. They mean the pilot cannot certify the advertised full workload’s memory or reliable warm cost.

**Round-1 blocker dispositions**

| Round-1 blocker | Disposition | Evidence |
|---|---|---|
| 1. Clean-process 3D import | **RESOLVED** | `p1three.py:37–39`; fresh CPU interpreter import succeeded. |
| 2. A/B timing scope | **RESOLVED** | `p1time.py:188–205,218–228,259–269`; argument/schedule preparation precedes measured dispatches. |
| 3. Required smoke controls | **PARTIALLY** | Correct doubled-work subjects, separate injected subject and dev6 bridge now exist (`make_configs.py:65–71,111–115`); reproduction/zero-update/swaps added, but acceptance defects above remain. |
| 4. Full validity | **PARTIALLY** | Shared and 3D checks improved (`p1common.py:274–290`, `p1three.py:245–246`); timing omits fields/w0 and offline qualification does not revalidate saved data. |
| 5. Enforceable reservations | **PARTIALLY** | Locking/reserve-before-submit and time-limit matching added; ambiguous-release, concurrency and requeue/increase gaps remain. |
| 6. Qualification/control decisions | **PARTIALLY** | `p1score.py`/`smoke_accept.py` exist; defective predicates and missing timing/subject-wide analysis remain. |
| 7. Missing-mesh verdicts | **RESOLVED** | `p1stats.py:105–108,174,205–207`; original two-mesh counterexamples now withhold the registered verdict. |
| 8. Discarded retries | **RESOLVED** | `p1time.py:233–241`, `p1kern.py:115`, `p1three.py:184` retain all attempted measurement rows. |

**Every round-1 WRONG per-file finding**

| File / finding | Disposition |
|---|---|
| `p1core.py`: insufficient recording buffer | **PARTIALLY** — wrapper assertion at `p1common.py:257–258`; core still clamps writes at `p1core.py:220`, and certifier still trusts buffer length at `:323`. Direct split calls bypass the assertion. |
| `p1common.py`: detached-array guard | **PARTIALLY** — `:236–247` checks supplied small-array fingerprints; incomplete coverage at `:177–180`, as demonstrated above. |
| `p1common.py`: incomplete validity | **RESOLVED in the helper** — `:274–290`; fields, w0, states and used step ratios checked when supplied. Timing caller omission remains. |
| `p1acc.py`: order-anchor validity | **RESOLVED** — `:284–308` records tighter/q++ validity and includes every validity flag in interpretation/pass. Smoke ignores the resulting failed Boolean. |
| `p1acc.py`: disabled/incomplete reproduction | **PARTIALLY** — moved offline and invoked by `smoke_accept.py:55–57`; incomplete equality at `p1score.py:165–189`, and no production analysis integration. |
| `p1time.py`: contaminated intervals | **RESOLVED** — `:188–205,218–228`. |
| `p1time.py`: doubled-work drift procedure | **PARTIALLY** — drift recorded at `:273–274`; no retry/validity enforcement. |
| `p1time.py`: per-invocation iteration evidence | **PARTIALLY** — four doubled-work iteration arrays retained at `:275`; ordinary and fused invocations still retain comparison Booleans rather than their per-step arrays (`:233–249`). |
| `p1comp.py`: discarded diagnostic repetitions | **RESOLVED** — `:118–127` retains raw arrays. |
| `p1kern.py`: discarded retries | **RESOLVED** — `:115`. |
| `p1kern.py`: one-fixture/no-predicate equivalence | **RESOLVED** — `:137–143`, every fixture at largest rank, explicit threshold. |
| `p1three.py`: import failure | **RESOLVED** — `:37–39`, clean-process probe passed. |
| `p1three.py`: rho nonfinite semantics | **RESOLVED** — `rho_a24()` used at `:252`. |
| `p1three.py`: one-fixture equivalence | **RESOLVED** — `:203–209`. |
| `pilot.py`: missing tight BE | **RESOLVED** — `:71–72`. |
| `pilot.py`: retained settings/certifier/Q3 footprint | **PARTIALLY** — additions at `:69–96`; incomplete full timing footprint described above. |
| `npcheck.py`: rho nonfinite handling | **RESOLVED** — `:69–77`; independent agreement enforcement remains incomplete in the consumer. |
| `make_fixtures.py`: archived setup unchecked | **RESOLVED** — `:26–33` reads and checks archived setup. |
| `make_configs.py` / configs: smoke omissions | **PARTIALLY** — correct subjects/dev6 bridge now configured; downstream predicates remain defective. |
| `cluster/stage.py`: certificate/time-limit binding | **PARTIALLY** — `:42–46,113–117`; reservation included in checksums, certificate not bound to workload. |
| `cluster/submit.sh`: non-atomic accounting | **PARTIALLY** — `:24–33`; ambiguous failure still releases reservation. |
| `cluster/submit.sh`: bare Python | **RESOLVED** — absolute interpreter at `:8`. |
| `cluster/collect.py`: unsafe partial/running cleanup | **PARTIALLY** — `:26–31` rejects marked partial archives, verifies checksums and checks queue. It accepts **any** COMPLETE marker, not completion of every staged step, and does not positively establish terminal accounting state. |
| `ledger.py`: locking/attempt/accounting gaps | **PARTIALLY** — locking and atomic writes at `:28–41`, duplicate live-attempt refusal at `:57`, requeue sum at `:85`; remaining gaps above. |
| `p1stats.py`: missing meshes | **RESOLVED** — `:105–108,205–207`; A4 subject-wide blocking remains an absent caller responsibility. |
| `test_p1.py`: missing regression coverage | **PARTIALLY** — missing-mesh/full-validity tests at `:130–153`; table-substitution, scorer and subject-wide identity failures still lack adequate regression coverage. |

The two related round-1 restatements were addressed: burn precedes warm calls in `p1comp.py:25–27`, and the historical source JSON is hashed in `ctrl_fixture_ur.py:51`.

**Smoke-control mapping**

| Registered control | Round-2 assessment |
|---|---|
| Synthetic timing fault / accuracy trend / H-acc-r precedence | Implemented; selected read-only test functions passed. |
| Synthetic H-cost-r / duplicated-column Q3 | Implemented in `test_p1.py`; not rerun in this audit. Acceptance must require their named records, not accept an empty summary. |
| Doubled work | Correct fixtures; drift/retry/acceptance semantics incomplete. |
| Component sum | Correctly descriptive under A2.2. |
| Dispatch sync and kernel drift | Producer predicates present; checker trusts stored decisions and does not require the complete component/panel inventory. |
| Component equivalence, A3.3 | Every fixture tested; acceptance crashes on the result schema. Also require finite per-fixture values before aggregating with Python `max()`. |
| Split/fused equivalence | Implemented; empty list can pass. |
| Timed=scored / injected identity, A4.1 | Split producer enforcement present; fused acceptance and subject-wide analysis incomplete. |
| Reproduction | Smoke invocation present; numerical-manifest equality incomplete. |
| Wrong-mesh | Original mismatch rejected; same-shaped table substitution still incompletely guarded. |
| Under-resolved rule, A1.1 | Correct BE fixture and independent own-state ρ; population/agreement completeness and offline rollout check incomplete. |
| Budget-zero, A1.6 | Correct pooled calculation and independent gamma comparison; does not assert six cases × 50 steps. |
| Zero-update | Distance rejection implemented; verified-anchor validity not enforced. |
| Case/time/kind swaps | Constructed metadata swaps rejected; full rollout/reference identity contract incomplete. |
| CN-R order | Producer computes the amended checks; acceptance tests existence instead of success. |
| Sign-flip, recording overhead, hardware sensitivity | Appropriately reported as probes. |

The selected CPU tests passed missing-mesh handling, shared validity, synthetic timing/accuracy controls, rank-transition precedence, and 69 Wilcoxon comparisons against SciPy. Those successes do not cover the false-pass and runtime counterexamples reproduced above.