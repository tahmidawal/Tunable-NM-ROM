- PASS — Sbatch requests one H100 on `gpu`, six hours, 160 GB host RAM, isolated paralab paths, f64/highest precision, manifest verification, and mandatory GPU preflight.
- PASS — All 13 manifest entries verify; all 10 provenance-listed files match HEAD `b84d52c8f`, and the complete local import chain is staged.
- PASS — J2 uses `model_M2`, 65 nodes, R′=512/256 × κ=4/3/2, shell-completed M, and the tensor comparator.
- PASS — All registered lattice/Gauss candidates and controls match, with gl48 converged, gl40 correctly aliased, and gl80/gl64 targets.
- PASS — Taus, convergence/target/rho bars, solver pins, validation seed 923801 ×64, certification seeds 923811–923813 ×8, and timing 16×2 match; no held-out cohort runs.
- PASS — Accepted smoke uses identical driver code and passes numerical/control gates; its 1.140 timing drift fails only the small panel, with A5 withholding cost claims until production timing passes.
- PASS — Memory fits 80 GB: smoke sampled 21.62 GB and allocator lifetime peak 23.85 GB, leaving ample room for larger cohorts and retained selected-rule blocks.
- PASS — The smoke-calibrated projection is approximately 1.1 hours, so six hours exceeds the required 2× margin.

SUBMIT: YES