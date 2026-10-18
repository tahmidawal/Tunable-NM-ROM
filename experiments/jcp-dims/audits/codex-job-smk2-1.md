**Overall: SUBMIT — for this 2D smoke only.** No blocking defect found in staged commit `4a1d0817308abf9f206cf05c4bb1397f63f4cfc0`.

| Item | Verdict | Evidence |
|---|---|---|
| **1. Batch script** | **CORRECT** | `jd_smk2`, `gpu` partition, one H200, 02:00:00 limit; cluster venv activated; x64 and `highest` precision exported. GPU preflight exits 42 on CPU fallback. Logs, outputs, cache and temporary files use the specified paralab namespace. Manifest verification precedes both drivers. |
| **2. Staging and dependencies** | **CORRECT** | All **155 provenance entries** match their committed Git blobs, byte counts and SHA-256 values. All **159 manifest entries** verify; no unmanifested files or missing expected lane files. All four `EXTRA` files, vendored modules, rotation, `fixture2d.npz` and checkpoint are present. CPU-only imports of both staged drivers resolved every loaded project module inside the staged tree. |
| **3. Smoke evidence coverage** | **CORRECT** | The configs exercise the requested paths, detailed below. They generate evidence for subsequent acceptance; successful process exit alone is insufficient. |
| **4. Reservation and lane limits** | **CORRECT** | Reservation matches the script’s two-hour limit, GPU, drivers, configs and commit. Ledger currently has zero spent/reserved hours and **24.5 GPU-h** pending core work; adding this smoke gives **26.5/55 GPU-h**. Submission serializes lane checks and reservations, refuses duplicate attempts, checks lane count `<3` and account count `<8`, verifies remote hashes, and retains ambiguous submissions’ reservations. |
| **Campaign-cap guarantee** | **NEEDS-RESTATEMENT** | Account concurrency is checked through `squeue`; the lock is lane-local. This is **not atomic enforcement across lanes**. Likewise, the ledger enforces 55 GPU-h, not the coordinator’s separate ≤2 GPU-h pre-production smoke allowance. Neither limitation blocks this currently empty lane’s first two-hour smoke. |

The evidence coverage includes:

- **Reproduction:** R384/M1536/gauss128 and R128/M512/fib4181, both LSPG-CNR and BE on **dev6**. I inspected qualify’s four corresponding `smkA` archives: their populations, dimensions and rules match. The coefficient comparison against the ≤`1e-10` bar remains an offline acceptance step.
- **Square and heavy cells:** square R128/M128 and R384/M384 paths; R512/M4096 gauss256→320→384, including actual gauss384 production and GAL.
- **Timing:** persistent reference A; both `q+` and `q++` certification phases, tight/tighter replays, square certification, doubled-work control, and the correctly targeted injected identity fault.
- **Families and LMF:** two draws × two rungs for Fibonacci, Sobol, Halton and MC; Smolyak coverage; paired LMF/default settings with rank, conditioning and Q3 evidence.
- **Controls and memory:** dev6 under-resolution and budget-zero controls, wrong-table checks, both fixture-gate blocks, and recorded device peaks with the 120 GB rejection threshold. Timing also checks memory release after each block.

The read-only cluster check found **two account jobs, zero `jd_` jobs**, and the `smk2` namespace absent. Submission must repeat its queue checks.

GPU runtime, memory bounds, reproduction and control outcomes remain unproven until collection and acceptance review. A later smoke must fit the **remaining actual** pre-production allowance.

No files changed, jobs submitted, LAB-LOG writes, or scratch created.