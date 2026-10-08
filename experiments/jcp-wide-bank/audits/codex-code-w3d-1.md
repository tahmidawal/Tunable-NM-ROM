**Do not submit the current J2/smoke configs.** Both reference a missing `rules_extra.npz` with a `null` SHA. The core numerical formulas are largely correct, but J4 tensor allocation, timing acceptance, target chunking, and retained audit evidence need fixes.

This was read-only: no files modified, no generators or GPU jobs run. I checked the source against DESIGN A0–A4 and `qpanel.py`, inspected the existing bank and hashes, and ran in-memory NumPy checks of indexing, mode prefixes, field metrics, the small-table schema, and Jacobian algebra.

Paths below are under `experiments/jcp-wide-bank/`.

1. **Refined error normalization and lattice indexing — CORRECT.**

   [w3d.py:65](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:65), lines 204–214 and 338–342 reproduce `qpanel`’s convention: interior coordinates \(k/64\), x-slowest flattening, direct bank evaluation on the shared lattice, and division by the **lattice reference initial norm**. Evolved maxima exclude time zero.

   The indexing check passed exactly at both 65 and 129 nodes. There is no missing mesh-volume factor; it cancels in the relative norm. **Fix: none.**

2. **Field-metric distance — CORRECT.**

   [w3d.py:329](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:329), lines 348 and 368 use `delta @ Lt.T`, with `Lt = L.T`. For row coefficients, this is \(\Delta C L\), exactly the transpose of \(L^\top\Delta c\). The denominator is the full-mesh initial norm, as required.

   A synthetic nonorthogonal-bank check matched the decoded field norm. Validation uses the six output states; certification checks every internal state, which is stricter. **Fix: none.**

3. **Shell-completed \(M\) and prefixes of the \(M_{\max}\) tables — CORRECT.**

   [w3d.py:227](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:227), lines 299–300: `complete_M`, `modes`, and table construction share the same stable eigenvalue ordering. Therefore `A[:M,:Rp]`, `lam[:M]`, and `kx[:M]` remain aligned.

   Verified J2 counts at 65 nodes:

   - \(R'=256\): 513, 771, 1027 for \(\kappa=2,3,4\).
   - \(R'=512\): 1027, 1538, 2052.

   At 129 nodes the corresponding \(\kappa=4\) counts are 1024 and 2049; the code correctly recomputes them. Nested Cholesky prefixes are also valid. **Fix: none to the slicing.**

4. **Per-setting off-mesh blocks and tensor handling — WRONG for J4.**

   [w3d.py:317](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:317) correctly builds off-mesh blocks directly at each setting’s \((R',M)\), satisfying A3-11. The tensor slice itself is mathematically correct.

   The defect is at **lines 233–234 and 319**: `tensor=True` builds a tensor for the **entire bank**, and the arm guard checks only whether `Rp` fits that tensor. On a 1024-column bank this builds and runs tensors above 512. Setting `tensor=False` instead removes the required new-bank tensor controls at 256 and 512.

   **Fix:** separate lean full-bank tables from a tensor table capped at 512 columns and the required test count for those tensor arms. Guard tensor deployment explicitly with `Rp <= 512`.

5. **G1–G4 and the 32-column table — NEEDS-FIX in G2 coverage; G4 is real.**

   [w3d.py:248](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:248): the small table has edges `[0,32]`; `arm_data(tbs,32,Ms)` supplies the keys and shapes expected by `make_fsc`. The extracted `arm_data` path passed a synthetic schema check.

   G4 invokes **two separately defined solver implementations**, `TB.make_fsc` and `OM.make_fsc_rule`, with identical tensor data. It compares fields, complete coefficient trajectories, iterations, reasons, and rejections. It is not a self-comparison. Shared lower-level functions limit what it independently tests, consistently with A3-13.

   However, **G2 compares only 64 tests** at lines 257 and 260, although the gate table is built for shell-completed 128 tests. A defect confined to the remaining rows would escape.

   **Fix:** use `tbs['kxyz'][:Ms]` and `tbs['Tsym'][:Ms,:32,:32]` for G2. G1/G3 formulas are correct; their sampled-point scope should remain explicit.

6. **Eligibility, selection, controls, K-conv/K-target, check alias — NEEDS-FIX in recorded evidence.**

   The actual decision logic is sound:

   - [w3d.py:112](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:112): zero reason-3 exits and the registered reason-0 fraction. With 25 steps, 1% permits **zero** reason-0 exits.
   - Lines 420–427 require both converged/check eligibility on validation **and certification**, plus their distance and target gates.
   - Lines 431–448 exclude controls and invalidate selection when an eligible control would qualify.
   - Lines 385–391 correctly alias the check rule after its single phase-1 rollout. Timing excludes `check` and includes its ladder alias once.

   Two recording defects remain:

   - `crecs` is never persisted. Certification eligibility and check distances survive only as aggregate gate values; check certification trajectories are also discarded.
   - Line 447 sets `available=True` when gates pass even if `nm_ is None`, conflating “selection permitted” with “a qualifying arm exists.”

   **Fix:** persist certification records, including per-draw distances and exits; distinguish `gates_passed` from `available = gates_passed and arm is not None`, with a reason for unavailability.

7. **ρ population, continuum targets, and chunking — NEEDS-FIX.**

   [w3d.py:363](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:363) correctly constructs the population from converged certification states \(k\ge1\): 600 states for J2. Passing full `T` to `continuum_adv` is correct: the vendor internally selects `Cs.shape[1]` columns.

   **Lines 395–396 do not implement A3-11’s 64-state target chunks.** The vendor function chunks points at 65536 but processes every supplied state together. Both targets therefore receive all 600 states. Ladder evaluations use 256-state chunks; those are mathematically fine.

   Line 412 records **p90 instead of the registered p95**.

   **Fix:** call the unchanged vendor target in outer batches of 64 states, concatenate results, and record p95. Preserve the current converged-state population and target normalization.

8. **Reached-state Jacobian — CORRECT formula; NEEDS-FIX diagnostics.**

   [w3d.py:456](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:456) computes
   \[
   S\left[A+\Delta t\left(J_u+\nu\Lambda A\right)\right]
   = A+\Delta t\,S J_u,
   \]
   exactly the Jacobian returned by vendor `tables._lm_step_parts`.

   **Yes, \(dN/dc=J_u\).** Although \(N(c)=\tfrac12J_u(c)c\), \(J_u\) also depends linearly on \(c\); differentiating supplies the other half. The synthetic algebra check agreed to \(8.9\times10^{-16}\), and a directional finite-difference check passed.

   The diagnostic omission is at lines 302–305 and 462–464: the code computes singular values but saves only summaries, despite A0-10/A2-10 requiring the spectra.

   **Fix:** persist the singular-value arrays for \(A\) and each reached-state Jacobian. Do not halve `Ju`.

9. **A–B–A timing, determinism, freeze, and final-panel survival — WRONG in acceptance and coverage.**

   [w3d.py:468](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:468) correctly warms signatures, randomizes invocations, blocks whole outputs, and saves raw query timings. The deployed family is selected from the per-setting panel and saved before the final panel.

   Concrete defects:

   - **Lines 526–532 select deployment even when timing gates fail.** Final timing gates likewise do not withdraw timings. This violates K-time.
   - **Lines 494 and 593 check only six coefficient vectors.** Unlike `qpanel`, they do not check decoded fields. A decoder-only discrepancy passes. FOM outputs receive no determinism check.
   - **Lines 585–595 omit the required 0.2-second burn after calls lasting ≥0.5 seconds.** The per-setting panel implements it.
   - **Lines 515–524 run Jacobian microbenchmarks without an explicit burn and discard repetition arrays**, retaining only median/minimum.

   The suspected use-after-delete is **not a bug**: `final` holds strong references to `q`, `data`, and reference coefficients. `jax.clear_caches()` clears compilations, not live array contents; the final panel rewarms them.

   **Fix:** enforce timing withdrawal, verify decoded outputs and the FOM baseline, apply the same post-long-call protocol in both panels, and retain microbenchmark repetitions after a burn.

   Separately, **line 350 saves converged restricted fields only for audit cases**, contrary to A2-12’s requirement to retain them for every case. Save `F` when `nm == 'conv'` as well.

10. **Projection floor — NEEDS-FIX in enforcement.**

    [w3d.py:548](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:548) uses the correct sampled bank, thin SVD, relative rank cutoff, initial-reference denominator, and evolved-time maximum. `lstsq(...,rcond=1e-12)` matches the SVD cutoff, and the near-zero branch uses an absolute comparison.

    **Lines 563–564 merely record the consistency discrepancy.** Arbitrarily bad disagreement does not fail the check.

    **Fix:** require the recorded discrepancy to be finite and ≤\(10^{-10}\), and mark the floor unavailable or abort if it fails.

11. **Cross-mesh distances — CORRECT for the registered two meshes.**

    [w3d.py:620](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:620) matches bank, setting, arm, and case, then evaluates coefficient differences using the same lattice bank. The lattice initial norm is the correct denominator. Different shell-completed \(M\) values across meshes do not invalidate this field comparison.

    Certification entries are skipped, aliases have matching entries, and `coefs` is not mutated during iteration. **Fix: none for J4’s two-mesh contract.**

12. **Memory at J4 scale — NEEDS-FIX; feasibility is not established.**

    From the registered J4 rules, counting the check alias once, I calculate **669,759 resident rule points**. At \(R'=1024,M=4097\), the rule blocks require approximately **32.9 GB**; the 129-node mesh bank adds **16.8 GB**. These exclude tensor storage, construction temporaries, compiled workspaces, target evaluation, and previously retained deployed arms.

    Specific issues:

    - [w3d.py:531](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:531) intentionally retains deployed arrays across settings. Thus line 545 does **not** implement a complete per-setting release.
    - Loop-local `d`/`s_` references can also retain the last arm beyond deletion of `arms`.
    - The full-width tensor defect in item 4 adds about **34.4 GB** at rank 1024, before tensor-construction temporaries.
    - [w3d.py:106](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w3d.py:106) records sampled maxima only. It omits the lifetime peak required alongside them by A4-11 and the start/end accounting specified by A3-11. Resetting shared counters without synchronization can also misattribute a boundary sample.

    **Fix:** cap tensor construction, implement target state chunking, explicitly release temporary references, and either rebuild deployed blocks for the final panel or budget their cumulative residency. Synchronize interval sampling and record allocator lifetime peaks plus start/end usage. Run the mandated full-size J4 smoke before claiming it fits; static accounting alone does not prove OOM or feasibility.

13. **Staging, rule generation/pinning, and late crashes — WRONG as currently packaged.**

    [configs/j2_3d.json:29](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/configs/j2_3d.json:29) and [configs/smoke_3d.json:26](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/configs/smoke_3d.json:26) contain `null` for the extra-rule SHA. Neither extra-rule artifact exists in this checkout.

    [make_configs.py:56](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/make_configs.py:56) silently generates that invalid configuration when metadata is absent. Runtime pinning **fails closed**, rather than bypassing verification: `sha_file` first fails for the missing file; supplying the NPZ alone then fails `h == None`.

    **Fix:** generate and commit the rules and metadata, regenerate both configs, and make config generation fail immediately unless the artifact exists and its actual SHA matches metadata.

    Additional staging findings:

    - [cluster/stage.py:24](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/cluster/stage.py:24) **does include `b3d_common.py`**. The J2 repository-local import chain is complete.
    - The fixed file list includes only M2. J4 needs its new bank explicitly supplied through `--extra`, or preferably discovered and validated from `cfg['banks']`.
    - The stager requires committed, clean inputs; the audited checkout currently has modified source and untracked 3D configs, so it is not stage-ready independently of the missing rules.
    - `rules3d/make_rules.py` uses the correct vendor generators and seed-0 shifts and checks duplicated `gl64` bytewise. I found no generator mathematics defect.
    - Existing M2 spread keys include `"32"`, `"256"`, and `"512"`; the suspected current spread-key crash is absent. Validate all configured ranks/spread keys for every bank before expensive setup to prevent future late failures.
    - Arm deletion iterates over `list(arms)`, so it does **not** trigger dictionary-size mutation errors.

| Item | Verdict | Required action |
|---|---|---|
| 1. Refined error/indexing | **CORRECT** | None |
| 2. Field metric | **CORRECT** | None |
| 3. Shell completion/table prefixes | **CORRECT** | None |
| 4. Blocks/tensor | **WRONG** | Cap tensor construction and deployment at 512 |
| 5. G1–G4 | **NEEDS-FIX** | Extend G2 to all gate-table tests; G4 is real |
| 6. Selection/gates/alias | **NEEDS-FIX** | Persist certification evidence; distinguish availability |
| 7. ρ targets/population | **NEEDS-FIX** | Chunk targets by 64 states; record p95 |
| 8. Reached Jacobian | **NEEDS-FIX** | Formula correct; retain singular spectra |
| 9. Timing/freeze/retention | **WRONG** | Enforce gates, check fields, complete burn/data protocol |
| 10. Projection floor | **NEEDS-FIX** | Enforce SVD–lstsq consistency |
| 11. Cross-mesh distance | **CORRECT** | None for two meshes |
| 12. J4 memory | **NEEDS-FIX** | Account for retained arrays and peaks; full-size smoke |
| 13. Packaging/staging/pins | **WRONG** | Generate/pin artifacts; validate J4 inputs before staging |