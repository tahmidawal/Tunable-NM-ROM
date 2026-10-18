**NO-GO for the registered real-size cluster smoke.** The numerical head stepper looks sound; the blockers are smoke acceptance, evidence preservation, memory certification, and job lifecycle handling.

I made no file changes, submitted no jobs, and used no scratch files. Checks ran CPU-only with bytecode and compilation caching disabled. The checkout advanced from `d9a4ea73a` to `03978a2af` during the audit; the requested implementation files did not change between those commits. I also inspected the newly added smoke checker and tests.

**Ranked required fixes**

1. **Make smoke acceptance enforce the required controls.**

   [smoke_accept.py:98](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:98) reports G-cert results, but lines 113–116 omit them from `all_controls_pass`. A smoke with failed G-cert can therefore pass.

   Similarly, lines 137–154 calculate ordinary timing validity but omit it from acceptance; doubled-work acceptance also ignores `drift_ok`. A timing smoke can pass despite unusable ordinary measurements.

   Coverage remains incomplete:

   - [hacc.py:256](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hacc.py:256) reproduces only linear **BE against combined**, whereas §7 requires **BE and CN-R against qualify**.
   - Lines 272–289 test the analytic head Jacobian only for the arm receiving the control block—currently the incumbent, not every K.
   - [test_h.py:103](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/test_h.py:103) checks the coefficient-map Jacobian with a relaxed `1e-8` threshold; it does not exercise the production step-Jacobian degeneracy classifier at `1e-10`.
   - The new checker implements zero-update and independent budget-zero checks, so those are **not missing**.

2. **Repair training completion and collection.**

   [htrain.py:247](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/htrain.py:247) finishes without writing `COMPLETE`; [collect.py:30](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/cluster/collect.py:30) requires that marker in every output directory before cleanup.

   **Failure:** a successful `smk1` containing training cannot pass normal cleanup. Collection also includes every full training pickle, contrary to A5’s compact-only archival intent.

3. **Preserve enough raw evidence to reconstruct anchor qualification.**

   [hacc.py:130](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hacc.py:130) saves tight internal states and budget counts, but not tighter internal states or tighter budget counts. Lines 175–179 retain the computed certificate and verification boolean.

   **Failure:** offline audit cannot independently reconstruct the tighter certificate or determine whether its “verified” flag concealed a budget exit. [htime.py:117](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/htime.py:117) retains still less raw anchor evidence.

   Main head coefficients **are saved**. Distances for tight/tighter/successor/check trajectories remain recoverable from saved latent outputs **provided the exact head archive and bank/rotation are retained**. That does not recover missing internal trajectories.

4. **Make the 4096² pilot a defensible memory certificate.**

   [hpilot.py:51](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hpilot.py:51) loads the same incumbent for all thirteen head subjects. `M` changes, but K remains 16, and the head cache shares weights and cold-start tables. Lines 65–69 run the first head, which is K=16/M=16, and one linear arm.

   **Failure:** `within_bar=true` is not evidence that the actual independent heads, K=32 execution, and timing-job working set fit. The main-rule tables are conservatively sized, but no demonstrated bound covers the omitted allocations. Use representative actual dimensions and independent head storage, or explicitly bound the difference.

5. **Correct reservation and concurrency rules.**

   - [ledger.py:97](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/ledger.py:97) exempts smoke attempts from mandatory-inventory validation. With an empty inventory, a smoke reserves against zero pending core work.
   - Lines 82–83 exclude only the exact attempt name. With pending `train`, reserving `trainr1` counts both the retry and the original pending training reservation. My read-only check returned pending totals **4 versus 5 hours** for these equivalent submissions.
   - [submit.sh:14](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/cluster/submit.sh:14) locks only this lane. Two lanes observing seven account jobs can both submit and reach nine.
   - Lines 24–25 permit two simultaneous smoke jobs; §8 permits only one. The two-hour smoke limit is also not enforced.

6. **Finish the registered smoke configuration coverage and production path.**

   [make_configs.py:60](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/make_configs.py:60) selects the incumbent and L8 for the rule probe, omitting the registered newly trained head. No supplied config implements the all-38 head cost probe. Line 76 has no `production` dispatch despite advertising that command.

   `resources.py` remains the qualify implementation: it imports absent `analyze`, expects `smkA`/`smkB`, and dispatches `qacc`/`qtime`/`q3run` at line 129. It cannot generate this lane’s required production certificates.

7. **Close configuration-integrity holes.**

   - [hrule.py:136](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hrule.py:136): omitted `min_pass` defaults to all arms. For a three-seed head group where exactly two pass, this violates A2.5 and can choose a later rule or exhaust the ladder. Default by arm type or require the field.
   - [harm.py:86](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/harm.py:86) and line 158 bypass recording-length checks. An arm constructed for 50 steps can run a 100-step schedule and silently overwrite the final recording slot. Current fixed-`DT0` drivers avoid this, but the interface does not enforce it.
   - [hcommon.py:44](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hcommon.py:44) ignores the compact archive’s `source_ckpt_sha256`. Its bank comparison then compares bank parameters copied from the supplied bank against that same bank. A compact archive from another bank is accepted without detecting the mismatch.

**Per-file verdicts**

Line references below refer to `experiments/jcp-heads/`.

| File | Verdict | Concrete assessment |
|---|---|---|
| `hcore.py` | **CORRECT** | Lines 47–108, 134–175: BE residual/Jacobian, LU LM, clipping, damping carry, predictor, tolerances, initialization and decode match the deployed q=0 algorithm. Lines 189–206 rebuild the correct history and use fresh `jax.jacfwd`. |
| `hcommon.py` | **WRONG** | Compact provenance hole at 44–72. Also, defaults at 26/118 retain superseded IC tolerances; registered drivers obtain A1.3 values only through `harm`’s import-time mutation. |
| `harm.py` | **WRONG** | Missing schedule/buffer guards at 86–89 and 158–161. Levels at 27–28/153–155 and initial-fit veto at 188–198 otherwise implement A1.3 correctly. |
| `htrain.py` | **WRONG** | Completion-marker defect at 247–250. The numerical training recipe matches the reference mid arm: keys, objective, schedule, batch and encoder. Identity correctly uses incumbent `Z_tr`, with pick alignment at 92–95. |
| `hrule.py` | **WRONG** | Default seed threshold at 136 violates A2.5. With explicit correct `min_pass`, lines 113–143 implement validity, rollout, own-state rho, and dev6 check-rule predicates correctly. Cache keys include arm/rule/case; role rebuilding works. |
| `hacc.py` | **WRONG** | Missing tighter raw evidence and incomplete registered controls, detailed above. Anchor predicate at 58–63 and main coefficient recording at 150 are correct. |
| `htime.py` | **NEEDS-RESTATEMENT** | Timing engine is correct: identity at 46–48, cohort repeats at 208–211, split/fused retries at 224–260, sequential doubled work at 267–294. Its retained evidence does not support a fully independent anchor audit. |
| `hpilot.py` | **WRONG** | Lines 51–69 do not reproduce or establish an upper bound for the actual head execution/storage working set. |
| `cluster/stage.py` | **NEEDS-RESTATEMENT** | Committed-blob staging, namespace, manifests, GPU partition and preflight at 65–111 are correct. Smoke-limit enforcement and the production certificate workflow are incomplete. |
| `cluster/submit.sh` | **WRONG** | Account cap is a racing observation; one-smoke-at-a-time rule absent. Lane cap, prefix, reservation call and before/after queue checks are present. |
| `cluster/collect.py` | **WRONG** | Training lifecycle mismatch at 30–31; indiscriminate pickle collection at 44–47. Checksum verification itself is sound. |
| `ledger.py` | **WRONG** | Smoke reservation bypass and retry double counting at 82–105. Ordinary exact-name pending reservations and failed-job restoration are implemented. |
| `make_configs.py` | **WRONG** | Incomplete smoke coverage at 60–72; advertised production command absent at 76. |
| `configs/smk1train.json` | **CORRECT** | Lines 5–21 specify identity plus K=4/K=32 short training as intended. |
| `configs/smk1acc.json` | **WRONG** | Lines 155–178 configure only incumbent controls and only the BE combined linear fixture. |
| `configs/smk1rule.json` | **WRONG** | Group list contains Hinc and L8, not the registered new-head probe. |
| `configs/smk2time.json` | **CORRECT** | Lines 68–118 exercise dev/validation cohort repeat dispatch, doubled work and injected identity failure. This is a smoke subset, not the required all-38 cost probe. |
| `configs/smk2pilot.json` | **WRONG** | Lines 32–95 substitute the same K=16 checkpoint for every head size and seed. |

The training disjointness comparison uses compatible physical descriptors and is meaningful for descriptors represented in `mu_tr`. Add an explicit **4,608 unique-descriptor assertion** to substantiate its stated coverage of every training trajectory.

**Verification performed:** the supplied `test_h.py` passed. A small synthetic CPU fixture reproduced the vendor’s entire 50-step BE trajectory exactly, with identical iteration counts and exit reasons. Analytic/autodiff Jacobian differences were at most \(1.11\times10^{-16}\); certificate/recorded-gamma differences were below \(1.06\times10^{-16}\) for BE and CN-R. This supports the core implementation; it does not replace the registered real-size reproduction gates.

**Final decision: NO-GO.** Fix smoke acceptance and lifecycle/evidence defects before submission; repair the pilot before treating it as authorization for 4096² production.