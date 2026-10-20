**Overall: FIX-FIRST — ledger blocks submission.** Audited HEAD/staged commit `f6dd77c023891b10d34bae7096660a5180d96cad`.

| Item | Verdict | Finding |
|---|---|---|
| Diff since `f8242f3d0` | PASS for this job | No changes to the `dacc` execution path or configuration. Analysis/qualification changes affect later interpretation; the 3D driver change does not affect this run. Resource changes implement A7.1. |
| `run.sbatch` / reservation | PASS | One H200, `gpu` partition, 02:45:00, dedicated paralab directory, GPU preflight, f64/highest precision. Reservation agrees. |
| COMMIT / manifest / provenance | PASS | All **167 manifest entries** verify, no uncovered files; **163 staged source/input files** match HEAD. Provenance hashes, sizes and commits agree. |
| Configuration identity | PASS | Byte-identical to the previously audited config; certificate hash matches. Five R512/M cells × eight Gauss rungs = **40 settings**. |
| Certificate arithmetic | PASS | Independently reproduced **1.7504056963 h**. Multiplying by 1.5 and rounding upward to quarter-hours gives **2.75 h**. |
| Model plausibility | PASS, estimate only | Intercept **2.443708 s/case**; R512/M512 predictions span **2.473–3.489 s/case**, consistent in scale with the reported ~2.6 s observation. This does not establish a completion bound. |
| Certificate description | NEEDS CORRECTION | JSON `note` and module overview still describe the superseded floored-linear model. |
| Ledger | **FIX-FIRST** | Pending `f2b` is **1.5 h**, not 2.75 h. Two other pending allowances are also below their regenerated certificates. |

Calling the ledger’s **read-only validation function** reproduces three submission blockers:

- `f2b`: pending **1.5**, certificate **2.75 h**.
- `f2k`: pending **7.25**, certificate **21.5 h**.
- `b2f`: pending **3.75**, certificate **12.5 h**.

Simply raising all three makes spent + reserved + pending **56.915 h**, exceeding the **55 h** cap. The remaining campaign budget therefore needs reconciliation before submission.

Also, retry reservation handling removes only the exact attempt key: reserving `f2br1` does **not** consume pending `f2b`. Correct the accounting so the retry replaces that allowance without double-counting.

No files changed, jobs submitted, large copies made, or LAB-LOG writes performed.