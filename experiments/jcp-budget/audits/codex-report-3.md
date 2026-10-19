Audited HEAD `4d6e98e4a`. No repository changes, SSH, or submissions. Scratch usage: 22.3 MB.

**Reproducibility confirmed:** `report.md`, `runs/EXPORT.json`, and all five PNGs regenerate byte-identically. All **20 GFM tables** parse correctly, with matching source and parsed column counts.

| Round-2 blocking item | Verdict | Evidence |
|---|---|---|
| 1. Registered B7 reporting | **NEEDS-RESTATEMENT** | Both normalisations now appear for 3D and T=1; monotonicity counts and reported Kendall τ values pass independent recomputation. However, median/worst floor trajectories cover only 1024² through T=0.5. The generator’s [floor block](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/make_report.py:132) never renders the other panels. Eligible arrays are available: 3D has 16 cases × 11 times; T=1 has 9 cases × 21 times. [DESIGN §5.3](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/DESIGN.md:244) requires floor trajectories through the horizon. |
| 2. Glossary corrections | **CORRECT** | Completeness and population gates agree with `ana2d.summarise`, `ana3d.summarise3`, and `ana7` growth/sampling rules. Stability correctly uses initial energy, while monotonicity counts compare consecutive outputs. Ingredients now distinguish 2D Gauss 320²/640² from 3D gl56/gl64 and exclude I/M in 3D. |
| 3. B8 descriptive populations | **CORRECT** | Tables explicitly label descriptive populations and explain their inclusion of non-rankable references. Verified 38 stored cases at 256²/1024², including `val32|10` and `val32|31`; 4096² has 16. The glossary now preserves this exception. |

No additional blocking defect identified.

**NOT-READY**

1. Complete the registered median/worst floor trajectories for the remaining B7 panels, including 3D through T=0.5 and 2D through T=1. Show eligible population sizes and retain W3’s provisional status; regenerate the outputs.