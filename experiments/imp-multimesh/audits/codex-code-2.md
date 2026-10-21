**Verdict: FIX-FIRST for S1 and J2.** S1 has a definite missing-file failure. J2’s successful path provides the requested measurements, but its process handling and submission safeguards remain incomplete. Production qualification and cost logic also retain blockers.

Read-only audit completed: no files changed, no jobs submitted, no GPU used, and no writes to `LAB-LOG.md` or main. The intentionally deferred `load_inputs` adapter and production timing/compile configs are excluded from the blocking list.

**Verified evidence**

- Both staged manifests pass: **143 files for S1; 63 for J2**.
- J2’s historical code hashes match: **26 bank files and 33 extraction/head files**.
- S1’s resource-certificate config hash matches.
- Tufts venv executable and reference-store directory exist. Both remote attempt directories were absent; the account queue was empty when checked.
- Tiny CPU arithmetic tests pass: blocked/dense differences **6.14e-15 for G**, **9.33e-16 for b**; the Jacobian mutation changes G by **2.87e-2**.
- Python parsing and JSON parsing pass.
- Small adversarial checks reproduced two incorrect decisions described below.

**Part A — disposition of every round-1 blocking finding**

Line numbers below refer to the current worktree files.

| Round-1 finding | Round-2 disposition and evidence |
|---|---|
| Full-grid basis-prefix copies | **NEEDS-RESTATEMENT.** Still present in [mmcore.py:313](experiments/imp-multimesh/mmcore.py#L313): caching limits simultaneous prefixes but does not eliminate allocations. A5.3 replaces the old allocation certificate with empirical feasibility, so copies alone are no longer grounds to reject S1. Production memory remains unproven. |
| Obsolete full-factor rank criterion | **CORRECT.** `orth_ok` now tests orthonormality only; per-prefix singular-value conditioning is separate. `mmcore.py:134–151`. |
| Missing val32 tighter anchors | **CORRECT.** Each cohort supplies its first two cases; stationarity is pooled. `mmcore.py:383–414`. |
| Basis failures abort all ranks / incorrect terminal states | **CORRECT for recorded certificates.** Certified ranks continue; target configurations start `UNRESOLVED`, certificate failures become `NOT-CONSTRUCTIBLE`, measured configurations receive the registered states. `mmbuild.py:117`, `201–257`, `299`. **Remaining gap:** missing source-certificate evidence is silently accepted at `205–207`, `231`. |
| Missing κ=4/κ=8 rollout agreement | **PARTLY FIXED — WRONG overall.** Distances are calculated, but the result is attached only to the lower-κ configuration. `mmbuild.py:308–320`. Both members need the shared sensitivity result, otherwise κ=8 lacks its admissibility input. |
| PM trajectory timers lack burn-in | **CORRECT.** Burn-in precedes each trajectory; first-call compilation remains labelled. `mmbuild.py:333–345`. |
| Failed smoke controls do not fail the job | **CORRECT on the intended completed path.** `mmaccept` aggregates mandatory controls and exits nonzero. `mmaccept.py:17–34`, `64–69`. **Actual S1 staging prevents that path from completing**, below. |
| Missing control-4 fallback | **CORRECT.** A6.2’s half-candidate fallback exists, and acceptance requires the unmutated fixture to pass. `mmcore.py:346–359`; `mmsmoke.py:163–174`. |
| Memory certificate / missing ours-table pilot | **NEEDS-RESTATEMENT.** A5.3 withdraws the numerical bound; ours’ table/query pilot now exists. `mmsmoke.py:201–212`. **Still WRONG:** no host peak is recorded, and production matrix dimensions are not asserted. |
| Missing kept-state-mask comparison | **CORRECT.** Dense mask is reconstructed from the actual block norms and compared explicitly. `mmsmoke.py:148–155`. |
| Missing controls 8/9 | **WRONG in the staged job.** Implemented in `mmaccept.py:36–63`, but their required historical analysis file is absent from S1. |
| Timed outputs discarded / A2 unchecked | **CORRECT.** Every timed call supplies retained identity evidence, including A2 and both competitor repetitions. `mmtime.py:36–53`, `117–131`, `189–201`. |
| Missing J5 evidence / identity excluded from timing validity | **CORRECT for the implemented coefficient/state checks.** J5 is mandatory; basis and timed-call identity enter acceptance. `mmtime.py:100`, `238–243`. |
| Missing outliers / retry overwrites observations | **CORRECT.** Shared statistics include outliers; attempts are retained separately. `mmtime.py:63–66`, `218`, `244`. |
| Incomplete Q1/Q2 predicates | **WRONG.** Certifier agreement, successor validity and check-run counts were added, but finer-rule agreement still does not affect Q2 acceptance. `oursq.py:76–85`. |
| Nonfinite / unverified evidence handling | **PARTLY FIXED — WRONG overall.** `tri` handles missing/nonfinite scalar evidence, but invalid successor/check runs can become `FAIL` rather than inherited `UNVERIFIED`; finer-rule convergence is still ignored. `oursq.py:35–42`, `82–87`. |
| Exact coverage, row/NPZ identity, mesh and descriptor identity | **PARTLY FIXED — WRONG overall.** Full-case coverage and report mesh are checked. NPZ mesh/configuration metadata, recorded `W_sha`, and full descriptor identity are not validated. `oursq.py:32–60`, `91`. |
| Missing X statuses create false FIRES | **CORRECT for missing mesh keys.** They default to `UNRESOLVED`. `cost.py:46–48`. **Still incomplete:** an entirely missing X configuration is invisible; an empty `ours` dictionary returns `FIRES`. |
| Missing competitor configurations create false NO-MATCH | **WRONG at the decision API boundary.** `complete_universe` exists but neither `match` nor `stop_rule` requires or invokes it. `cost.py:51–60`, `107–137`. |
| Accounting variants affect only ours | **WRONG in the currently implemented engine.** `var` still reaches only `ours_cost`; competitor accounting has no variant-dependent selection union. `cost.py:63–79`, `109`, `132`. This needs implementation before production decisions, independently of the deferred file adapter. |
| Missing production adapter, aliases and input gates | **DEFERRED by your instruction.** Not a smoke blocker and not counted as a round-2 implementation failure. It must enforce the complete registered universe and canonical dependencies when implemented. |
| Control-8 missing preconditions / changing expectations / cumulative vacuity | **CORRECT through `selftest_strict`.** Required controls and the second-X precondition are enforced; vacuity is per workload. `cost.py:286–329`. |
| Stage ignores command failures / shell operators evade timeout | **CORRECT.** Commands execute as scripts under timeout, dependent steps skip, and final exit reflects failure. `cluster/stage.py:80–89`, `123–126`. |
| Stage timeouts exceed allocation | **CORRECT for nominal budgets.** Timeout sums are checked with a reserve. `cluster/stage.py:75–78`. |
| Deferred-import staging checks absent | **CORRECT for Python imports.** S1 explicitly checks `p1common` and `p1acc`. **Insufficient for runtime data dependencies**, as the missing analysis JSON demonstrates. |
| Probe manifest/command failures ignored; incompatible outputs | **PARTLY FIXED.** Manifest, missing marker, extraction and head failures set failure; `output/` and checksums are present. **WRONG remains:** bank process is killed without `wait`; dependent head fitting runs after extraction failure. `cluster/stage_probe.py:118–139`. |
| Probe budgets and complete invocation wall times | **CORRECT.** Nominal budgets fit 125 minutes; shell timers surround complete extraction/head invocations. `cluster/stage_probe.py:90–91`, `130–136`. |
| Missing acceptance/resource submission gates | **PARTLY FIXED — WRONG overall.** Gates exist, but arbitrary non-`prod` kind values bypass acceptance, and a certificate with `config:null` has no binding to the submitted script. `cluster/submit.sh:18–30`. |
| Submission/reservation atomicity | **WRONG under ambiguous remote failure.** A lost SSH response after successful `sbatch` causes reservation release because no JID was received. `cluster/submit.sh:45–55`. The local lock fixes the ordinary concurrent-submit race, not this case. |
| Collection lacks terminal check, archive, checksum verification and cleanup | **CORRECT on the normal path.** All are implemented. `cluster/collect.sh:10–32`. **Remaining safety defect:** the supplied JID is not verified to belong to the supplied attempt before moving/deleting its directory. |
| Ledger double-counts running time / negative projections / concurrent writes | **CORRECT.** Terminal elapsed plus active full reservations, finite positive validation, and locking are implemented. `cluster/ledger.py:26–45`. |
| Duplicate bindings / atomic persistence | **PARTLY FIXED.** Rebinding an already-bound attempt fails, but duplicate job IDs across attempts are unchecked; writes use direct `write_text`, not atomic replacement. `cluster/ledger.py:32–34`, `55–58`. |
| Smoke config lacks CPU-exit assertion and real replication comparator | **PARTLY FIXED.** CPU exit 42 is asserted; comparator is implemented. Missing staged reference data still blocks replication acceptance. `configs/smoke.json:5`; `mmaccept.py:36–44`. |

The six-setting ours configs, source recipes, replication fixture and target inventories remain consistent with their stated roles. Production aliases remain the deferred adapter’s responsibility.

**Additional confirmed code defects**

1. **Q2 can falsely qualify an unconverged quadrature rule.**  
   At [oursq.py:81](experiments/imp-multimesh/oursq.py#L81), `max_dist_qp_qpp` is collected but never used. A small mocked 38-case invocation with all finer-rule distances set to **100** returned:

   ```text
   Q1 PASS Q2 PASS QUALIFIED
   ```

   That invocation also supplied deliberately wrong `W_sha` values, which were never checked. Require all six finite finer-rule distances to satisfy the registered threshold, preserve `UNVERIFIED` for invalid/unconverged finer evidence, and validate row-to-array identity.

2. **Incomplete searches still prove NO-MATCH unless a caller remembers an optional helper.**  
   A fixture containing only one failed configuration per CP arm returned `NO-MATCH`. Calling `complete_universe` first changed it to `UNRESOLVED`. Require an explicit, validated universe at the decision boundary; smoke fixtures can declare their deliberately reduced universe.

3. **A6 compile accounting is not implemented.**  
   `mmtime` retains seconds even when its listener suspects a cache hit (`mmtime.py:113–114`, `185–186`); `cost.py` has only a multiplicative `compile_scale`. There is no implemented missing-entry treatment, absolute upper scenario assignment, or compile-dependent withholding. `mmcompile.py` provides useful fresh-process measurements, but these do not complete A6 accounting.

4. **The smoke memory record is incomplete.**  
   `mmsmoke.py:104–114` records device counters only. The actual configuration generates **10 real states**, tiled to **120**, yielding the intended **46,080 × 32,768 matrix, 12.0796 GB**, before any dropped-state filtering. It does not generate the design’s stated 12 real states. Assert the resulting dimensions and record host peaks; reconcile the real-state count explicitly.

**Part B — staged jobs**

**S1 — FIX-FIRST**

The batch shell is substantially repaired:

- H200, GPU partition, absolute venv, f64 and highest precision are configured.
- GPU preflight exits 42; the deliberate CPU control requires exactly 42.
- `JCPREF_STORE` is set correctly and `JCPREF_META` is cleared.
- Command budgets total **5,220 seconds** against **5,700 seconds** allocation.
- Outputs go under `output/`; command failures reach the final exit code.
- S1’s staged numerical code matches the current code, including the A6 fallback.

However, **acceptance inevitably fails** at [mmaccept.py:36](experiments/imp-multimesh/mmaccept.py#L36). It opens:

```text
experiments/jcp-p1-scaling/checks/analysis.json
```

That file is absent from `runs/s1`. `stage.py` excludes `checks/`, and `smoke.json` supplies no `extra_files` override. Import checks cannot discover this dependency.

| Required control | Actual staged coverage |
|---|---|
| 1: CPU preflight | Exercised and exit 42 asserted. |
| 2: prolongation/restriction mutations | Implemented, with QR reconstruction and code-transform checks. |
| 3: mapping mutation | Mapped coordinates and values checked; unmapped **value** failure checked. The required unmapped-coordinate failure is not explicitly asserted or recorded. |
| 4: Q1 mutation and fallback | Implemented; unmutated fixture must pass. |
| 5: blocked/dense construction | Candidate sets, masks, norms and mutation checked. |
| 6: case identity | Descriptor assertion mutation and both e0 diagnostics implemented. |
| 7: timing injection | Clean/injected competitor blocks implemented. No sentinel measurement/check is included in this control. |
| 8: decision fixtures | **Not reached:** missing historical analysis file is opened first. Witness-dependent tests also require an accuracy-admissible measured fixture; merely increasing ours’ cost cannot create one. |
| 9: replication | Replication run is scheduled, but **comparison cannot execute** without the missing file. |

The 4096 pilot covers resampling, contributions, NNLS, a competitor query, PM timing, and ours’ bank/query. It does **not yet supply the required host-memory evidence**. Successful execution alone would therefore not complete A5.3 feasibility evidence.

Also, `mmaccept` writes `ACCEPT.json` only at its successful end. A missing prerequisite currently produces a traceback and no structured failed acceptance record. Write explicit failure evidence even when earlier steps fail.

**J2 — FIX-FIRST; successful measurement path is otherwise suitable**

The staged historical code and checkpoint resolve correctly. The script uses the correct venv and highest precision; the imported historical modules enable f64. GPU preflight exits 42. J2 does not use E0 references, so `JCPREF_STORE` is not needed.

Its nominal step budgets are **3,600 + 1,800 + 1,200 = 6,600 seconds**, fitting the **7,500-second** allocation.

The retained bank log supplies:

- step 1;
- steps 5,000 and 15,000 for `rho_sub`;
- steps 20,000 and 25,000 for `rho_full`;
- the transition window;
- `done: recon` for `t_done`;
- the 576-trajectory generation timer.

The same decomposition applied to rounded historical A100 log timestamps gives **10,698.97 seconds**, **0.134%** from the registered **10,684.65 seconds**, within the required 5%.

Extraction and head-fit wall timers surround the complete commands. Outputs and internal logs use `output/`, and `OUTPUTS.sha256` is generated.

Remaining pre-submission fixes:

- **Reap the bank process before starting extraction.** [run.sbatch:33](experiments/imp-multimesh/runs/j2/run.sbatch#L33) sends TERM, sleeps and sends KILL, but never waits or records termination status.
- **Validate the required bank markers**, not only the final `done: recon` substring, before accepting the probe.
- **Skip head fitting after extraction failure.** It currently runs regardless and then deletes the extraction NPZ regardless of head success (`run.sbatch:40–47`).
- **Bind the resource certificate to the actual staged job.** J2’s certificate has `config:null` and still describes a **2,700-second** bank budget while the staged script uses **3,600**.
- Fix the shared submission ambiguity described above before launching either job.

No current S1/J2 output path targets another lane. The remaining namespace-corruption risk is in collection: unchecked attempt/JID pairing can authorize deletion of the wrong attempt. Validate the attempt name and bind its JID to the ledger/job identity before collection.

**Summary**

| Area | Verdict |
|---|---|
| Core resampling and blocked ECSW arithmetic | CORRECT in small CPU checks |
| Rank handling and cohort anchors | CORRECT, with missing-certificate evidence still needing rejection |
| Smoke controls and acceptance | FIX-FIRST |
| Ours Q1/Q2 qualification and identity | WRONG |
| Timed-call identity and retry retention | CORRECT for implemented checks |
| Cost universe, accounting variants and A6 handling | FIX-FIRST before production decisions |
| Submission/collection safeguards | FIX-FIRST |
| **S1 staged job** | **FIX-FIRST — missing replication reference; incomplete memory/control evidence** |
| **J2 staged job** | **FIX-FIRST — process handling and resource/submission binding** |

**Blocking fixes**

1. Stage and hash the historical analysis JSON; make failed acceptance produce a structured record.
2. Enforce finer-rule convergence and row/NPZ/descriptor identity in `oursq`.
3. Require complete declared decision universes; implement variant accounting and A6 compile handling before production verdicts.
4. Record host-memory peaks, assert pilot dimensions, and complete the mapping/sentinel control evidence.
5. Attach rollout agreement to both κ configurations; reject missing source-certificate evidence.
6. Reap and validate J2’s bank process, gate head fitting on extraction success, and preserve failed-step evidence.
7. Bind resource certificates and submission kinds to staged jobs; retain reservations until ambiguous submissions are reconciled.
8. Verify attempt/JID identity before collection and make ledger persistence atomic.