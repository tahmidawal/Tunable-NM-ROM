**FIX-FIRST — one blocker: the resource certificate underprices square adaptive solves.**

[resources.py:179](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/resources.py:179) selects the largest-work adaptive smoke unit, κ=8, and scales it by \(mMr\). It ignores the substantially slower square-system behavior:

- κ=1 smoke: **370.35 s / 8 form–rung–case units = 46.29 s/unit**, including anchors and diagnostics.
- Certificate allowance for the entire production κ=1 block: **635.72 s** across 384 units.
- At lat131072, the model allows approximately **1.56 s/unit**; even smoke production solves alone took **7.79–31.79 s**, before anchors.

The **3.612 h projection / 5.5 h limit** reproduces mathematically, but its calibration is unsupported for κ=1. This does not prove a timeout; it requires square-specific pricing, accounting for the different tighter-case fractions. Regenerate the certificate, reservation, ledger allowance, and staged bundle afterward.

Other checks passed:

- Design/config/manifest agree on M values, ladders, forms, 64-case population, tighter cases 0–7, square tolerances, and exclusion of fs at κ=1.
- Adaptive keys, `Wstore`’s `|prod` suffix, and `dist_next` match the analysis consumer.
- M=4097 path was exercised; smoke peak **52.646 GB**, below 120 GB.
- All **173 manifest entries** verified; provenance hashes and available committed blobs match commit `51123802d`.
- Sbatch uses H200, GPU preflight, f64/highest, and isolated paralab paths.
- Ledger commitments total **48.3568/55 GPU-h**, including pending k3=5.5. Live queue: **2/3 lane**, **7/8 account**; submission wrapper rechecks caps.

The κ=1 smoke’s **46–47/50 non-certified steps** and large tight distances are scientific outcomes, not submission blockers.

No files changed; no jobs submitted.