**CONFIRMED**

All held-out table numbers match independent recomputation from `result.json`. Errors below are worst/median across 64 case-wise evolved errors; ROM timings use all A1 ∪ A2 invocations.

| Mesh | Arm | Worst % | Median % | GPU ms | Table-1 speedup |
|---|---|---:|---:|---:|---:|
| 256² | Accurate: R384, graphs | 0.874860 | 0.026419 | 45.0905 | 0.378690× |
| 256² | Fast: R128, graphs | 2.546901 | 0.267509 | 18.1658 | 0.939970× |
| 256² | Parent R384, exact first step | 0.874860 | 0.026384 | 125.2144 | 0.136369× |
| 256² | Parent R128, exact first step | 2.546901 | 0.267548 | 33.8631 | 0.504245× |
| 512² | Accurate: R384, default | 1.128390 | 0.028552 | 46.2785 | 0.614577× |
| 512² | Fast: R128, default | 2.781243 | 0.280902 | 18.2368 | 1.559580× |
| 512² | Parent R384, exact first step | 1.128389 | 0.028484 | 235.7123 | 0.120663× |
| 512² | Parent R512, q=0 head | 7.838666 | 0.700637 | 32.9070 | 0.864304× |
| 1024² | Accurate: R384, default | 1.249860 | 0.029219 | 53.1034 | 1.557606× |
| 1024² | Fast: R128, graphs | 2.959556 | 0.288717 | 22.0707 | 3.747682× |
| 1024² | Parent R384 | 1.249860 | 0.029219 | 89.1085 | 0.928240× |
| 1024² | Parent R128 | 2.959556 | 0.288717 | 33.2141 | 2.490331× |

All engineered picks omit the exact first step; parent arms use default compilation.

The fastest eligible FOM is `lean_nt3e-3_l3e-3_dt005` at every mesh, with these modes:

| Mesh | Mode | Worst error % | GPU ms | Nonlinear convergence |
|---|---|---:|---:|---|
| 256² | graphs | 0.109068 | 17.0753 | 64/64 |
| 512² | graphs | 0.118959 | 28.4417 | 64/64 |
| 1024² | default | 0.129034 | 82.7141 | 64/64 |

Each passes the accurate pick’s worst-error threshold. Both speedups per mesh use this same FOM numerator, as prescribed by A0.

- **Frozen selection/provenance:** exactly the four named selections/twins, without extra ROM arms. Selection commit `079ffc07` precedes config commit `ff1c15f1`, which precedes held-out summary commit `fde3868b`. A0 also predates them. Every archive records `ff1c15f1`; committed config bytes match the provenance SHA and current config, and parsed configs exactly match `result.json`.
- **Repetitions/SHA:** each mesh retains 2,944 invocations: four ROM repetitions per arm/case and one FOM repetition per setting/mode/case. Every invocation records full-SHA checking and identity with its quick output. All 15 FOM mode pairs have identical output hashes across all 64 cases.
- **Timing gates:** worst symmetric drift factors are **1.005970 / 1.001457 / 1.010224**; maximum neighbour ratios are **1.000817 / 1.000622 / 1.006594**. All satisfy 1.10.
- **1024² parity:** all 64 cases pass for both engineered/parent pairs, including recorded integer diagnostics. Maximum recorded field/state differences are **2.713e−15** and **3.427e−15**. Independent saved restricted-field/internal-state comparisons also pass.
- **Saved fields:** restricted-error checks pass their stated thresholds. Full-grid case-0 recomputation differs by at most **2.42e−16**; saved full-field hashes match. All committed summary source hashes match the archived results.

**DISCREPANCIES**

- Raw `result.json` archives are **git-ignored**, so there is no raw-result commit to verify. The chronology is established through committed summaries at `fde3868b`, whose SHA references match the raw results.
- Held-out **FOM per-step integer parity was disabled** (`fom_mode_parity=false`, empty records), despite A0 requiring it. Output SHA parity is confirmed; per-step Newton-vector parity is not.
- The FOM neighbour test is **degenerate**: one repetition per case makes each case-normalized observation exactly 1. All 30 FOM ratios therefore equal 1 automatically.
- No numerical discrepancies in the held-out table.

**WORDING ISSUES**

- “All gates pass” should distinguish executed checks from skipped/vacuous checks. Held-out certificate records are empty, NumPy rho checks cover zero states, and FOM integer parity is disabled.
- “hold64 … never used for any choice” is too broad: A0 explicitly uses hold64 to choose the FOM comparator. Say “never used to select ROM settings or compile modes.”
- Full-grid error recomputation covers case 0; other cases at 512²/1024² have restricted-grid consistency checks, not independent full-grid reconstruction.

No files were changed.