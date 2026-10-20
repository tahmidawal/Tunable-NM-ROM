**Overall: NOT READY for an unconditional p1/m1 handoff.** The screening selections and FOM matches reproduce correctly. Two issues need disposition: m1’s large-mesh calibration uses fewer cases than registered, and 15 certification comparisons miss the registered G2 tolerance. “All replay labels are mixed” also needs correction.

Audit was read-only, CPU-only. No repository files, LAB-LOG, or main were written; no jobs were submitted.

**1. Job integrity — NEEDS-RESTATEMENT**

For all four archives, `OUTPUTS.sha256` verifies. The results contain cohort 923801, cases 0–63, `complete=true`, H200, and commit `6f61b238a35f4e55282e28cc60223bd406eaedd3`. Logs confirm `jax_backend=gpu x64=True precision=highest`. The 21 staged top-level Python files checked per job match that commit.

I found no substantial W1024 anomaly attributable to the 3.26 GB captured-constants warning:

- Against the smoke, 76 common coefficient arrays differ by at most \(3.19\times10^{-16}\); 172 certification records differ by at most \(1.82\times10^{-14}\).
- W1024 comparator case-0 timing: smoke **488.865 ms**, g3 **486.588/486.165 ms**. Ratios: **8.4122**, versus **8.4226/8.3625**.
- g3 T1 median is **2.002725**; worst subject drift is **1.004601**; no invocation exceeds the drift band.

However, the raw `rows[].gam_cert` versus `rows[].steps.gn` comparison finds **5 G2 violations in g2 and 10 in g3**, all on fixed-phase FS steps. A1.4 requires
\[
|\gamma_{\rm cert}-\gamma_{\rm rec}|\le10^{-9}\gamma_{\rm rec}+10^{-12}.
\]
For example, g2 `W768|lat32768|fs|BDF2|0.01|K2`, case 59, index 24: certified **2.318685e-12**, recorded **4.287280e-13**, difference **1.889957e-12**.

These are tiny, near-zero discrepancies; none changes stationarity classification. They nevertheless fail the literal gate. They need an explicit numerical-tolerance disposition, not an unqualified “all gates passed.” This does **not** establish that the captured-constants warning caused them.

**2. Eligibility — CORRECT arithmetic, subject to the G2 qualification above**

I independently recomputed all **684 configurations** from rows, certification arrays, ADT validity, distances, and scores: **436 eligible, 248 ineligible, zero unresolved**. These agree with `checks/analysis-grid.json`.

All eight distinct comparators/winners/B* satisfy E-a. Their other eligibility quantities are:

| Selection label | Certified non-stationary steps | Worst ADT distance |
|---|---:|---:|
| C_old | 0.0625% | 3.72916e-5 |
| C_W768 | 0% | 3.60277e-5 |
| C_W1024 | 0.125% | 2.97016e-5 |
| W_C_old, also W_C_W1024 | 0.46875% | 2.07839e-4 |
| W_C_W768 | 0% | 2.26943e-4 |
| B_C_old | 0.0625% | 3.72907e-5 |
| B_C_W768 | 0% | 1.75031e-4 |
| B_C_W1024 | 0.125% | 3.04392e-5 |

The largest deployed ADT tight/tighter distance is **1.81785e-7**, comfortably below **2.5e-5**.

For these eight configurations plus **20 randomly sampled others**—seed 20261017—I reconstructed all 64-case distances from `W_grid.npz`. Maximum discrepancy: **1.11e-16**. Decoding cases 0, 5, and 7 against pinned E0 references gave maximum score discrepancy **2.22e-16**. All deployed tight/tighter coefficient checks also reproduce.

**3. Controls and population — CORRECT**

Recomputed from the archived records:

| Setting | K1 non-stationary steps / 640 | K2 non-stationary steps / 640 | K3-CN cases exceeding distance bar |
|---|---:|---:|---:|
| M2_512 | 251 | 640 | 24 |
| W256 | 138 | 640 | 37 |
| W512 | 201 | 640 | 24 |
| W768 | 248 | 640 | 14 |
| W1024 | 276 | 640 | 14 |

Thus K1–K3 fire at every setting. K3-BE also fires.

Independent K4 decodes reproduce swap changes **0.675373–0.676936** and time-shift changes **0.184713–0.186310** across g1–g3. K9 reconstructs **0.00012500000000000176** and **0.0004999999999999983**, correctly classified within/beyond. Both K10 injections are rejected.

All five screening panels pass T1: medians **1.995224–2.002725**. The planned population has no missing, extra, or duplicate configurations.

Individual drift outliers exist—g1 panels **49/1160** and **44/1448**, g2 **40/1160** and **20/872**, g3 **0/872**. They do not invalidate the registered **subject-median** drift gate, which independently reproduces.

**4. H2 selections — CORRECT**

Using all common ranked cases, the registered endpoint inequalities, screening costs, and 3% tie rule:

| Comparator | Matching configurations | Winner | B* | S_screen | S_BE_screen |
|---|---:|---|---|---:|---:|
| C_old | 31 | M2_512 gl24, FS-K1 BDF2, dt=.0125 | M2_512 gl24, FS-K1 BE, dt=.01 | 1.995307 | 1.717769 |
| C_W768 | 89 | M2_512 gl24, QN-4 CN, dt=.016666667 | W768 lat16384, QN-4 BE, dt=.01 | 5.220656 | 2.063435 |
| C_W1024 | 37 | M2_512 gl24, FS-K1 BDF2, dt=.0125 | W1024 lat32768, FS-K1 BE, dt=.01 | 16.815526 | 1.898340 |

Each winner’s 3% tie pool contains only that winner.

The M2 selections are plausible and permitted: H2 searches across banks and schemes. The BDF2 winner’s median/worst errors are **0.00641837 / 0.04407916**, versus C_W1024’s **0.01529880 / 0.04586627**. The CN winner gives **0.00335389 / 0.04645986**, versus C_W768’s **0.01538463 / 0.05284161**.

Every panel uses `A=M2_512|gl32|fs|BE|0.01|K1`. Panel-wide baseline medians range **61.19–63.42 ms**. These ratios are valid for the registered screening procedure; p1 must still confirm the speedups directly.

**5. H1 verdicts and replay labels — NEEDS-RESTATEMENT**

H1.2–H1.4 reproduce:

- H1.2: **10 ineligible**.
- H1.3: **42 eligible, 48 ineligible**.
- H1.4: **2 screening passes, 20 screening failures, 3 rescues, 5 unavailable**.

The passes are W768/BDF2/dt=.0125 at **0.547976**, and W1024/BDF2/dt=.025 at **0.568725**, relative to FS-K2.

The classifier is correct, but **“all mixed” is false**. `analysis-grid.json.H11` contains **71 mixed and one baseline-already-stationary**:

`H11.W256["5"]`: distance-only, `n_star=6`, baseline gamma **0.000520905839**, reproduced successfully.

For all 71 interpreted interventions, both additional sweeps with \(K\le4\) and trust×10 succeed; resetting damping fails. A2.2 therefore assigns “mixed.” The W256 exception follows A3. Replay case populations also match the registered failure predicates.

**6. FOM matching — CORRECT as screening evidence**

Independent calibration, admissibility, uncertainty alignment, and matching calculations reproduce `checks/fom-match.json`.

`ps|N16|0.025` matches **all eight ROM score entries**:

- Median warm time: **4.059232 ms**
- Median E0 error: **0.0004162209**
- Worst E0 error: **0.0255335664**

All four PS candidates with \(N\ge24,\ dt=.025\) are correctly inadmissible: **case 43 is non-finite/non-converged**. Eleven of fifteen PS candidates remain admissible.

Ten saved fields from `W_fom.npz`, spanning PS, CC, and UB, reproduce their E0 errors within **8.4e-17**. All 64 reference eligibility/eta values agree with the pinned reference store.

UB and UC genuinely have no joint match. Their best median/worst errors in the ladder are respectively **0.0226483 / 0.0781418** and **0.0194848 / 0.0718901**. CC matches reproduce, including dt=.01 for W_C_W768 and dt=.0125 for the other entries.

PS’s **boundary** label is correct: it is the cheapest admissible PS member. **Upper bound** correctly refers to FOM/ROM speedup relative to an unrestricted FOM search.

The supported statement is:

> On validation screening, PS N16 clearly outperforms the selected ROMs at ROM mesh L=64 in prepared-input latency and median/worst E0 error. Timings come from separate H200 jobs; paired m1 confirmation remains pending.

This is cross-resolution evidence, not a same-resolution spectral-64³ comparison. E0 retains its conditional Richardson qualification.

**7. p1/m1 configurations — WRONG in one concrete respect**

The selection JSONs, `rom-scores.json`, p1, and m1 regenerate **byte-for-byte** in `/tmp`.

p1 contains the required selections: **52 configurations**, **31 confirmation pairs** at 8 cases × 3 repetitions, all five distinct finalists’ converged-rule sensitivities and reduced-rule ADT checks, and the specified profiling/mesh-free/batching populations.

m1 contains four ROMs, eight matched pairs, all four meshes, and the A6 mesh-path match panels through `s3modes.py`.

But [`configs/m1.json:145`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/configs/m1.json:145) sets **`mesh.calib_large=[0,1]`**. [`select_s3.py:138`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/select_s3.py:138) generates it, and `s3modes.py:448` uses it for L=256/384. A1.9/A2.5 require **four calibration cases, 0–3**; A7 supplies no exception. Fix the generator/configuration or explicitly amend that requirement before accepting m1 as conforming.

**Required disposition:** correct the m1 calibration population, address the literal G2 failures without claiming they passed, and replace “all mixed” with the actual counts. The screening winner and FOM-match arithmetic itself needs no correction.