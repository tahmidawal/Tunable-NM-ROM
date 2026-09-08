

## 2026-09-07

### Heat frozen-head transfer04 closed: accurate transfer, efficient FOM still faster

Continued in the user-approved separate heat worktree/branch and existing namespace, with no new branch or merge. The bounded transfer study used scientific source `88ae5905d3f1ec424edbc4d7b8027e19eebe296b` and job `3354958` on pax049 / NVIDIA A100 80GB PCIe; driver elapsed time was 647.216831s. Native source, complete checked archive, audits and generated findings are committed through `7fcd0237fb5aaa3949e6e2a27236caa9a0598514`. The exact raw results SHA256 is `9efffdee15de45d5c375703877eb52654dc6569ea22ada919af0b56224886a76`.

Both expanded-coverage heads and their training-code libraries stayed frozen. The declared restricted polynomial-boundary Gaussian family, diffusivity and output times were unchanged. Original development seed 790711 contributed 4 cases and fresh development seed 790716 contributed 8; neither cohort entered training or initializer lookup. Final confirmation remains unopened. Requested meshes were [64, 128, 256, 512, 1024], shared observations 64, fixed Crank–Nicolson step 0.025 and validated gradient tolerance 1e-05. Mesh-dependent full-input QR projections and weak operators were rebuilt, and all complete/modular query parity checks passed.

Every FOM returns the exact supplied requested-grid initial field; host coarse restriction, direct-time sine-transform propagation, physically aligned interpolation and complete contiguous host output construction are charged. ROM instead returns its actual fitted initial field and charges full input transfer/projection, multistart fitting, evolution and all outputs. Coarse solver choices and the same-grid solver are deduplicated only when their actual solver/output configuration is identical. Paired GPU burn-in and synchronization precede each block; every cost and error uses the same actual invocation.

The native independent NumPy/SciPy audit checked 1152 invocations and 528 field files, maximum metric disagreement 5.26245713672e-14. Every FOM evolved/interpolated field matches independent SciPy/NumPy calculations to relative discrepancy 6.44690924177e-16; complete spectral-reference trajectories match to 1.1044037433e-15. The observed nested-reference discrepancy is 1.62810884586e-12; it is empirical evidence only and the strict bound is null. Selected nonstationary initial fits/steps: 0/0. Root independently checked the preserved production fields, cohorts, frozen weights/libraries, repetitions and t0 policy; maximum metric difference 0, source audit JSON SHA256 `36cd9886199a1c514fdcb829c333052c5609ebd1bc759ca97ddc7639648ec11a`.

Across both heads and all union-cohort meshes, the worst full current-relative physical error was 4.560008975% and worst shared-grid error 4.560001183%. Both frozen heads satisfy the empirically adjusted development target throughout this mesh ladder. The efficient FOM remains faster. The following union-cohort selections use the full-grid 5% target, medians of per-case timing medians, and paired ratios as medians of per-case FOM/ROM ratios of those medians. All repetitions and reference allowances enter eligibility.

| Output intervals | Selected ROM | Selected FOM | ROM ms | FOM ms | Paired FOM/ROM |
|---:|---|---|---:|---:|---:|
| 64 | expanded_seed790715 | fom_dst_64 | 12.565909 | 0.509903 | 0.040695447 |
| 128 | expanded_seed790714 | fom_dst_64 | 12.260174 | 0.673200 | 0.056623713 |
| 256 | expanded_seed790714 | fom_dst_64 | 13.207777 | 1.164132 | 0.087957124 |
| 512 | expanded_seed790714 | fom_dst_16 | 14.986074 | 4.515824 | 0.291338607 |
| 1024 | expanded_seed790715 | fom_dst_16 | 27.732991 | 21.410562 | 0.766730009 |

Full and shared physical norms are reported separately, with current normalization, initial normalization, absolute error and decay/energy/advancement records at every output. Same-grid semidiscrete discrepancies remain separate from discrete-versus-spectral spatial errors. ROM-versus-discrete discrepancy contains representation, weak-solve and time error; this transfer study does not independently separate those contributions. No earlier scientific result is retracted. The planned GPU memory figure is a live-array estimate, not a measured allocator peak; detailed Slurm host-memory accounting is preserved.

All 553 member checksums passed, and the 8974452498 archive bytes are tracked in 96 checked chunks with a tested restoration helper. Large extracted NPZ fields are omitted only as duplicate git blobs. Exact remote `/cluster/tufts/paralab/tawal01/mr_heat2d_20260907/transfer04` was deleted and absence verified. Generated native report, audit/summary JSON and PNG/PDF accuracy/cost/component figures live at `worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/transfer04/analysis/`.

This establishes frozen transfer for the declared restricted single-bump development family, not coverage of archived multi-bump heat or an efficient-FOM speed advantage. Broader scientific cohorts, tighter accuracy and further runtime work remain open for a later separately bounded continuation. No additional GPU study was launched this round, and all worktrees remain separate.
