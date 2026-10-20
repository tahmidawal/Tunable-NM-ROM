**Overall: SUBMIT** for staged smk3 at commit `b29e8be342c93865802a644022fa92f25cb017a5`. No remaining submission blocker found.

| Item | Verdict | Evidence |
|---|---|---|
| 1. Batch script | **CORRECT** | `jd_smk3`; `gpu` partition; one H200; two-hour limit; correct activated venv; x64 and highest precision; GPU preflight exits 42 on CPU; manifest verification; output, cache and temporary paths under the smk3 paralab namespace. Shell syntax passes. |
| 2. Staged provenance and dependencies | **CORRECT** | All **160 provenance entries** match the recorded commit byte-for-byte; manifest passes. Driver imports resolve entirely within staging. W1024 and M2 bank pins, both inherited rule archives, `rules_dims.npz`, and `fixture3d.npz` match. |
| 3. Smoke evidence and acceptance | **CORRECT** | Required configurations and writer/reader paths agree. The exact adaptive inventory and mandatory G-cert checks fix round 1’s acceptance gap. Details below. |
| 4. Reservation and caps | **CORRECT** | Two-hour reservation matches sbatch. Including smk2’s **0.1747 GPU-h**, smk3 and **24.5 GPU-h** pending core work gives **26.6747 / 55 GPU-h**. Lane submission is serialized; campaign submission uses the shared lock and aborts on failed `squeue`. |

The smk3 evidence paths cover:

- Independent M=1027 slicing, advection, shape and mode-hash checks.
- Both default-M reproductions on cases 0–3; matching speed3d archive keys exist.
- W1024 Gauss and lattice tails with ADT anchors.
- κ=1 and κ=8 adaptive GAL/LSPG runs, certification, and square diagnostics.
- Heavy three-table round member on 64 cases, plus the under-resolved control.
- 3D fixgate; timing panel with T1 and both W1024 heavy subjects; uncertified-panel rejection.
- Block and job memory peaks with the 120 GB acceptance limit.

`dist_next` and successor validity are now recorded in **`result.json` block results** for both ladder types. Adaptive NPZ filenames and reproduction archives match what `smoke_accept.py smk3` reads after collection.

**Round-1 correction:** the missing-Smol8 objection was incorrect: A1.2 had already withdrawn that gate; A5.1 makes this explicit. A5 resolves the smoke-allowance interpretation, and production release is recorded. Live accounting confirms smk2 completed in **629 seconds**; the queue snapshot contained three account jobs and no `jd_` jobs.

This approves submission readiness; numerical acceptance, memory fit and completion within two hours remain for the smoke to establish. No files changed, jobs submitted, GPU computations run, or scratch created.