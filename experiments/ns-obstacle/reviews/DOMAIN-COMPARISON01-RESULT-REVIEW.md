# Complete retained-state domain comparison review

Independent Codex reviewer `numerical_resume` verifies all 899 retained files, four states, all original integration points, strict barycentric membership and physical-coordinate reconstruction. No points are dropped or moved and no pressure shift is fitted. Raw velocity differences, both force-rule comparisons and exact common-pole pressure drops independently reproduce.

The generated diagnostic is `generated/domain-comparison01/summary.json`; its original-source hashes and generator are retained. Upstream sensitivity fails the unchanged original velocity/drag/lift bars. Downstream and alternative-outlet comparisons pass their listed sensitivity bars. Unshifted pressure-level differences remain diagnostic because moving/changing the natural outlet changes the pressure datum.

This confirms the original inlet failure. FD1's explicitly finite-channel scope remains unchanged; numerical population qualification and training are still not admitted.
