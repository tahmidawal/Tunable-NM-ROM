**FIX-FIRST — settle J6’s stale ledger reservation before submission.** The basis fix and J6b staging pass this narrow audit.

- **Blocking:** [GPU-LEDGER.json](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/runs/GPU-LEDGER.json) still marks J6 `PENDING`, reserving 3.5833 GPU-h. Adding J6b gives **2.8369 + 3.5833 + 2.5833 = 9.0035 GPU-h**, exceeding the 8-hour cap. `submit.sh` does not settle automatically.
- **Basis/trust/identity:** Correct. Prolongation operates column-wise; sign-normalized two-pass QR preserves restriction prefixes. A small CPU QR check matched basis, triangular factor, and transformed codes exactly. Existing basis and trajectory identity gates remain enforced.
- **Memory:** The wider basis plus prefix copy is eliminated. Basis384, sentinel, 38 inputs, and the observed **32 GiB** autotuning allocation total approximately **108.1 GB**, against a **138.7 GB** allocator pool. This provides plausible headroom, though rank-384 autotuning remains unmeasured.
- **Staging/time:** All **159 manifest hashes pass**; staged code and certificate hash match. The 4096 panel is unchanged, all 12 probes run independently with `always_run`, and nominal command limits total 150 minutes within the 155-minute allocation.
- **Merge:** Correctly retains J6’s valid 256/1024 panels and replaces its incomplete 4096 panel only with a valid successor.

No files changed, jobs submitted, or large arrays loaded.