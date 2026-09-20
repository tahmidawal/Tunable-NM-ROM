### NS3D frozen final comparison accepted

Job `4027788` (source `7681447fe74efb6c2755bb19304174d6867533e7`, NVIDIA A100 80GB PCIe, 81920 MiB) completed the frozen 32-case cohort, seed `202609203`. All 2880 paired invocations, 960 saved trajectories, 1600 independent history checks and 480 POD cold projections pass. Every empirical reference refinement passes. No recorded selected solve misses its stopping rule.

| Method | Worst evolved physical error (%) | Median GPU time (ms) | Cases failing physical target |
|---|---:|---:|---:|
| nmrom_pca64_free_q0 | 20.340976 | 1258.291151 | 31/32 |
| nmrom_pca64_free_q32 | 20.210269 | 1684.398519 | 31/32 |
| nmrom_pca64_free_q64 | 20.106909 | 2081.260450 | 31/32 |
| nmrom_pca64_free_q128 | 19.641035 | 2925.733331 | 31/32 |
| nmrom_pca64_free_q256 | 18.917805 | 4641.160268 | 31/32 |
| pod_weak_64 | 61.565062 | 534.661151 | 32/32 |
| pod_weak_320 | 30.747056 | 2496.188823 | 32/32 |
| pod_galerkin_64 | 61.565112 | 14.881493 | 32/32 |
| pod_galerkin_320 | 30.908760 | 30.560231 | 32/32 |
| free_bank_galerkin | 10.158266 | 104.961131 | 16/32 |
| fno3d_increment_projected | 0.848502 | 9.341563 | 0/32 |
| unet3d_increment_projected | 2.726019 | 2.145155 | 0/32 |
| deeponet3d_increment_projected | 36.034442 | 4.468062 | 32/32 |
| transolver3d_increment_projected | 3.849842 | 3.975968 | 0/32 |
| fom_dt0.01 | 2.589178 | 3.170075 | 0/32 |

The fixed-M correction ladder improves worst error by 1.075229 times for 3.688463 times the GPU cost. It is monotone but misses the physical target. NM is more accurate than matched latent-dimension POD and slower. DeepONet remains weak. This is a dense full-grid implementation and one warm training lineage, not independent seed replication.

The original strict local parameter gate is retained as failed. Two NumPy CPU implementations round scalar exp differently; all original data and final query results are retained unchanged.

Complete archive `10071abd0cf3845b4329a4d2c10367c741c2c124`: 157 files, 289 parts, 19355535360 bytes, full roundtrip and actual-Git verification passed. Exact remote final07 was removed. Root-authorized duplicate final fields were released only after acceptance; 14962020352 allocated bytes freed, with every path/hash/restoration command retained in `runs/final07/materialization_release.json`. All frozen inputs, checkpoints, POD/operator assets, latent histories and small provenance remain. No scientific values were changed. Final tables are `runs/final07/paper_summary.json` and `.csv`; original failed strict-local audit and its exact-runtime disposition remain retained.
