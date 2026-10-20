**SUBMIT: c2budA, c2budB, c2budC.** Audited HEAD `840c96c5649f8e651efb622bb4fd85c8356838b2`.

Both round-4 blockers are resolved:

- **G-score controls:** every required field in both dimensions rejects zero, missing, NaN, ±∞, Boolean, nonnumeric, negative and threshold-equal (`1e-6`) values.
- **Smoke binding:** altered smoke contents with unchanged path and job ID fail both offline and full smoke gates.

The round-3/4 mutation set also rejects as expected, including coordinated 3D coverage reductions. Structural checks were tested independently of file binding. Untouched smokes pass; reproduction counts remain **3,888 / 462** (2D / 3D).

**266 checks passed with no unexpected outcomes.** All three committed inventories verify against HEAD config/source blobs and the requested A100-80G, 48-hour, 128G allocation. Injecting a zero control makes all three certificate verifications reject.

| Job | Projected hours, including ×1.5 | Peak GB | Decision |
|---|---:|---:|---|
| c2budA | 46.181540 | 40.820703 | **SUBMIT** |
| c2budB | 46.181540 | 40.820703 | **SUBMIT** |
| c2budC | 4.587112 | 40.820703 | **SUBMIT** |

No files were created, modified or appended; no jobs were submitted. The pre-existing untracked audit log remains; staging’s clean-tree check will require it to be handled.