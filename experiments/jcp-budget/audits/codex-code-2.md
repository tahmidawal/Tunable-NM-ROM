**Overall verdict: NEEDS-FIX.**

Audited HEAD: `799e2ce5be4edc8abc4caeb29212d23f942ee975`. No files changed, jobs submitted, GPU used, or scratch created. CPU tests ran with bytecode writing disabled; the core test’s file-writing statements were removed in memory.

Paths below are relative to `experiments/jcp-budget/`.

**Verification of audit-1’s nine required fixes**

| # | Status | Evidence |
|---|---|---|
| 1. PS cross-checks, escalation, eligibility propagation | **NEEDS-FIX — partial** | Cross-check evaluation and separate escalation now exist: `anaref.py:116`, `ref/refrun.py:356`. Tier propagation remains defective; T=1 uses the wrong designated population. |
| 2. Final-purpose G-R2, prefix checks, scoring identity | **NEEDS-FIX — partial** | G-R2 now compares the new final triple by L/dt/purpose (`anaref.py:63`). Prefix checking exists but does not govern eligibility (`200–254`). Reference identity checks remain incomplete. |
| 3. G-B4 propagation and population denominators | **CORRECT** | Per-corner failures precede effect construction; failed ideal-family checks remove U-star-dependent decisions: `ana2d.py:162–195,211–212,251`, `ana3d.py:88–119,145–146,183`. Complete-case minimums and eligible-case pair denominators are implemented: `ana2d.py:361`, `ana3d.py:290`. |
| 4. Check anchors, pooled 3D acceptance, nonfinite handling | **NEEDS-FIX — partial** | 2D check replays now use `anchor=True` (`b2run.py:301,305`); 3D qp/qpp are certified (`b3run.py:101–102`). Pooled acceptance is corrected, but 3D rho serialization defeats nonfinite rejection. |
| 5. Complete B2/B7/B8 analyses | **NEEDS-FIX — partial** | Signed interactions, background terms, growth, ranking and IC diagnostics were added. The 3D F background is mathematically wrong; mandatory 3D quantities remain absent. |
| 6. Fail-closed smoke acceptance and independent verification | **NEEDS-FIX — partial** | Coverage checks and failure exit status improved (`smoke_accept.py:62–135`). Fine recomputation checks only fine error, and some controls still accept inadequate evidence. |
| 7. Promotion and `--no-fine` cleanup | **CORRECT for the sequential collection path** | Promotion now fails on errors and verifies copied hashes (`cluster/collect.py:51–66`). Missing-file exceptions are restricted to fine files; cleanup verifies their retained hashes (`34–41,81–90`). |
| 8. Pilot, certificates, smoke limits, campaign cap | **CORRECT against A4, with non-blocking limitations below** | Pilot now builds M=2048 main/qinf/qcheck (`b2run.py:543–545`). Horizon/config/GPU certificate checks and ≤2-hour smoke limits exist (`resources.py:87–130`, `cluster/stage.py:42–54`). Sequential lane smokes are enforced (`cluster/submit.sh:24`). A4 explicitly retains the cross-lane submission race. |
| 9. Resources provenance, T=1 ROM, 3D B8 disposition | **CORRECT for the original requests** | `resources.py` is declared rewritten; `b1024t1` exists; A4 explicitly defers 3D B8. However, three other provenance entries are now stale. |

**Numerical implementation checks**

| Area | Verdict | Evidence |
|---|---|---|
| `budcore` nout change | **CORRECT** | Diff from `p1core.py` is confined to output-count plumbing/allocation/indexing. All eight LSPG/GAL × BE/CNR/BDF2/CN CPU tests passed bitwise equality, prefix equality and finite/distinct tail checks. |
| Split initialization and scale | **CORRECT** | `budcommon.py:182–207` preserves the deployed scale and replaces only coefficients. |
| M ingredient and rule roles | **CORRECT** | `b2run.py:150–154,211–225` selects M=2048 and builds both converged/check roles. |
| Streamed LS QR | **CORRECT** | Direct block Q-transpose application, prefix columns, row order and acceptance: `budcommon.py:96–157`. A small rank-deficient-block fixture agreed with NumPy LS within `1.84e-16`. |
| 2D B7 certificates and anchors | **CORRECT** | Fresh LSPG gamma, GAL eta/gcert, tight/tighter and qp/qpp paths: `b2run.py:255–309`. |
| Reference horizon and extraction edits | **CORRECT** | Source diffs preserve stepping arithmetic while generalizing buffers/stages/output loops: `ref/refcore.py:102–238,251–278`, `ref/verdict.py:94–181`. A small 3D shared-node test confirmed the `s*j-1` convention. |
| 3D horizon/cache plumbing | **CORRECT** | `s3common.py:100–103,206–238`, `s3ctx.py:142`, `b3run.py:163–165`. |
| B8 numerical formulas | **CORRECT by inspection** | Fine decode, energy/dissipation, band mask and alias construction: `budcommon.py:215–253`, `b2run.py:408–450`. Independent acceptance coverage remains incomplete. |
| Config generation | **CORRECT** | Regenerated all 23 configurations in memory; every result matched its checked-in JSON. Both external input pins passed. |

**Remaining blockers**

1. **WRONG — cross-check tiers still permit forbidden verdicts.**

   `anaref.py:253–254` allows non-designated cases to remain eligible when their dimension/window is suspended. Downstream:

   - `ana7.py:98–103` emits “provisional: bounded growth” rather than withholding the verdict required by A2.3.
   - `ranking_adequacy()` and `front_order()` ignore tiers (`313–387`).
   - The lane statement uses `raw_label`, bypassing tier restrictions (`472–477`).

   CPU counterexamples with eight suspended cases returned **“preserved”** for front order and **“adequate”** for ranking.

   Additionally, `anaref.py:230–239` always requires 2D cases `{0,1,5,19}`. The registered T=1 cross-check population is `{0,19}`. Consequently, even both required T=1 PASSes cannot produce a validated W3.

2. **NEEDS-FIX — reference checks are not bound to the evidence actually used for verdicts.**

   The T=1 prefix check:

   - silently disappears when the T=0.5 input is missing;
   - can pass with only an available matched level;
   - never affects case eligibility.

   Evidence: `anaref.py:200–214,253–262`.

   The analysis also accepts an `--anaref` result based only on case/horizon keys (`ana7.py:106–114`), without verifying that its eligibility was established for the same reference file hashes/final realization now being scored. An old accepted analysis can therefore accompany replacement reference files.

   Identity checks remain incomplete: `b2run.py:344–350` and `ana7.py:77–85` omit cohort-hash, lattice/metric and expected-realization checks. `ana2d.py:63` checks only time-list length. The 3D case-swap control explicitly raises inside the test itself (`ana3d.py:460–465`), rather than demonstrating rejection by the production scoring path.

3. **WRONG — infinite 3D rho becomes an excluded “trivial” sample.**

   `b3run.py:112` generates infinity for a nonzero numerator with zero denominator. `s3ctx.py:91` passes the result through `C3.C.clean`, which converts **all nonfinite floats to null** (`vendor/quad3d/vendor/burgers3d-span/common.py:62–68`).

   `ana3d.py:364–374` converts null to NaN, counts NaNs as trivial, and rejects only infinities.

   Reproduced on CPU:

   ```text
   raw rho:         [0.01, inf]
   serialized:     [0.01, null]
   rho_nonfinite:  0
   rho_trivial:    1
   rho gate:       PASS
   ```

   Persist explicit status for zero/zero versus invalid/nonfinite values, and reject invalid samples after serialization.

4. **WRONG — the 3D F interaction uses the deployed background instead of the tight-end background.**

   The F effect is correctly constructed as `F|S − S` (`ana3d.py:164–170`). But its chi and numerical-background terms use `F['0']` for every ingredient (`203–208`).

   A3.1 explicitly requires the F background to be its own tight-end row. For F, use `F['S'] − PR`, and numerical background `F['S'] − Fstar`. The current reported values answer a different estimand.

5. **NEEDS-FIX — independent fine-route verification remains substantially incomplete.**

   `ana7.py:390–425` recomputes only `e_fine` and reference walls. It does not independently verify:

   - coarse-route error and both normalizers;
   - energy, dissipation and balance;
   - extrema and near-wall density;
   - fine floor and LS IC errors;
   - injected alias errors and C-E measurements.

   DESIGN §8 requires recomputation of **every in-job fine-route number** on check cases. Current smoke acceptance can pass when those other arrays are wrong, provided fine error agrees.

   Missing local fine files are skipped, and success requires only a nonempty passing subset (`409–425`). Smoke acceptance also permits missing local G-R3 recomputation because `None is not False` (`anaref.py:258`).

6. **NEEDS-FIX — 3D control acceptance can use invalid or uncertified comparators.**

   - C-Q’s converged FS comparator is run with `certify=False` (`b3run.py:128`).
   - `tag_acceptance()` accepts missing certificates by default (`ana3d.py:79–85`). A CPU fixture confirmed an FS row containing only `valid=True` is accepted.
   - Both C-T specs share `CTRL#T`; acceptance is keyed only by tag/case, so later rows overwrite earlier validity.
   - C-T then checks only the first case’s acceptance and applies it to every distance (`ana3d.py:448–451`).

   Key acceptance by tag/spec/case, require the applicable comparator evidence, and evaluate each control case against its own accepted comparator.

7. **NEEDS-FIX — required 3D analysis outputs remain missing.**

   `ana3d.py:191–196` reports D-a and endpoint distances but not the registered comparison against the sum/RSS of main effects. Its B7 path (`345–357`) does not compute the long-horizon projection floor or the primary E0 W1 error/eligibility that the 2D path reports.

   These omissions survive A4: only **3D B8** was deferred, not 3D B2 closure or B7 floor/error reporting.

8. **NEEDS-FIX — recorded tooling provenance no longer matches the revised files.**

   Every recorded source hash matched its source. Numerical inherited-file destination hashes also matched. However, recorded `sha256_after` values are stale for:

   - `cluster/stage.py`
   - `cluster/submit.sh`
   - `cluster/collect.py`

   Evidence: their entries in `PROVENANCE.json:129–170`. Record the current edits and hashes before staging an auditable smoke.

**Cluster and resource assessment**

**CORRECT:** unique attempt directories, committed-HEAD staging, paralab output/cache paths, GPU partition/preflight, x64/highest precision, and the normal one-GPU 45-hour reservation path are implemented (`cluster/stage.py:60–125`, `cluster/submit.sh:18–39`, `ledger.py:50–75`).

**Non-blocking limitations:** no OOM is demonstrated by this audit. The 4096² bank alone is approximately 48 GiB; QR workspaces, table-concatenation peaks, retained ICs and reference buffers remain substantial. The revised pilot exercises the missing M-table path, but not the new B8 fine-floor QR and accumulated `FQ` cache (`b2run.py:372–385`). Resource certificates select H200 through a CLI assertion rather than validating the actual smoke GPU record (`resources.py:114–117`). Treat their projections as estimates pending measured peaks.

**Non-blocking:** promotion is fail-fast but not transactional; an interrupted copy can leave partial destinations that require recovery. The campaign-cap race is explicitly disclosed by A4. Neither is repeated here as an unresolved audit-1 blocker.

**Overall verdict: NEEDS-FIX. Required fixes before smoke approval:**

1. Enforce eligibility tiers throughout decisions and use the registered T=1 cross-check roster.
2. Make prefix checks mandatory and bind eligibility/scoring to exact reference identities.
3. Preserve and reject invalid 3D rho values across JSON serialization.
4. Correct the 3D F interaction background.
5. Complete independent fine-route recomputation and require expected check-case coverage.
6. Repair 3D comparator certification and per-spec/per-case control acceptance.
7. Complete the mandatory 3D B2/B7 analysis outputs.
8. Refresh the three stale tooling provenance entries.