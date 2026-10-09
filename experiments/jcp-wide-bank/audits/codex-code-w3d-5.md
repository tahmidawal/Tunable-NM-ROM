Audited `b1ce86457`, read-only. **The substantive code blockers are fixed; no new blocking bug found.**

- **Reason 2:** pooled eligibility counts both 0 and 2. In-memory boundary tests passed.
- **A7:** corrected definition and candid chronology now acknowledge seeing per-arm errors. Minor wording inconsistency remains: the heading says “before any 3D selection was read,” while the body acknowledges reading “deployed None.”
- **Legacy flag:** `w3d.py` explicitly labels the per-rollout diagnostic; console output reports pooled eligibility.
- **Replay:** adds `arm_eligible_pooled` and derives final-panel coverage from input subjects. In-memory S2c replay passed with both present and absent panels, preserving `as_run`.

Minor diagnostic leftovers: replay retains historical `arms[*].all_eligible` without relabelling it, and sets `deployed.final_panel=False` even when the new coverage object reports coverage. These do not affect selection.

J2’s completed artifact remains absent locally. This approves running the corrected code; accepting amended J2 results still requires validating that artifact. No files modified.

OK-TO-RUN: YES