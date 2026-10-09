- **PASS — Sbatch:** H200 ×1, 240G host RAM, 10 hours, correct cluster venv, f64/highest precision, GPU preflight, and isolated paralab output are configured.
- **PASS — Provenance:** All 12 staged payload files match HEAD `68120f403`, including both banks and both rule archives; every manifest checksum verifies.
- **PASS — Scope:** Both meshes, all six bank/rank settings, κ=4, validation 923801 ×64, certification 923811–923813 ×8, and tensors only at R′≤512 match J4, with no held-out cohort.
- **PASS — Rules and thresholds:** Lattice/Gauss ladders, gl56 convergence, gl48 check aliased once, gl80/gl64 targets, controls, both taus, and gate bars match registration.
- **PASS — Amendments:** A6’s dual G1 checks, A7’s separately pooled validation/certification eligibility, and A8’s `adaptive_first=6` for every production arm are implemented.
- **PASS — Smoke calibration:** Identical driver code completed s4e with passing convergence, target, and timing gates; its setting took 300.05 seconds, whole job 396.8 seconds.
- **PASS — Memory estimate:** At 129 nodes, each bank’s tensor table adds 8.39 GB of bank rows plus 4.30 GB for the 2049×512×512 tensor, raising the smoke-based estimate to approximately 82.25 GB before build temporaries and retained timing data, with substantial headroom below the approximately 127 GB allocator allowance.
- **FAIL — Budget:** Although 10 hours exceeds twice the stated 2.8-hour projection, A4 explicitly requires `max(2×projection, A3 request)`—at least **12 hours** for J4—and BUDGET.json omits explicit tensor-build and final-panel costs.

SUBMIT: NO