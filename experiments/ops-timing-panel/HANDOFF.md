# ops-timing-panel — HANDOFF

**State (2026-09-22): DONE.** `opt101` = job **4179247**, COMPLETED 00:24:58 on pax049
(NVIDIA A100 80GB PCIe, `GPU-d881b2b0-b08f-83e6-0f3d-e646626625cc`), `jax_backend=gpu`, f64,
highest matmul precision. Audit: **0 failed gates**. Remote directory and the namespace are
deleted. Nothing is running.

- Worktree `worktrees/2026-09-22-ops-timing-panel`, branch `exp/2026-09-22-ops-timing-panel`,
  forked from `exp/2026-09-17-no-second` @ `ea812685`.
- Namespace `/cluster/tufts/paralab/tawal01/opstime_20260922/` — created, used, removed.
- Jobs used: **1 of 3**. Running: 0. `squeue` was empty before the submit and showed exactly
  one job after it.

## The result in one line

The U-Net, Transolver and FNO checkpoints now have **admissible** cost numbers: `unet-medium`
5.842 ms, `unet-refine` 5.859 ms, `tsol-refine` 11.058 ms, `fno-large` 7.330 ms GPU-query,
measured in the same allocation as the NM-ROM (fast 38.637 ms, accurate 465.396 ms), POD-LSPG
and the eight Newton–BiCGStab full-order settings. Three operator arms beat the FOM chosen by
the paper's rule (1.18–1.48×); every NM-ROM and POD arm is slower than it at 256².

## Where everything is

| what | path |
|---|---|
| design, incl. the Codex audit disposition (§11) | `DESIGN.md` |
| Codex design audit | `checks/codex-design-audit.md` |
| independent NumPy audit of the job | `checks/opt101-audit.json` |
| report (source-generated, glossary at the end) | `reports/2026-09-22-ops-timing-panel.md` |
| machine-readable summary | `reports/summary.json` |
| the table on its own | `reports/table-256.md` |
| generator (reads only the audit) | `reports/generate_ops_panel.py` |
| job logs, sbatch, provenance, manifests, per-arm timing JSONs | `logs/opt101/` |
| copied-file provenance | `COPIED-FROM.json` |
| checkpoint manifest with recorded hashes | `operators.json` |

`runs/opt101/` holds the 1.2 GB collected archive and is **git-ignored**: field arrays are not
committed. Everything needed to re-derive the report from the audit is committed.

## Open / next

- Nothing is running and the lane's question is answered. Jobs 2 and 3 are unused.
- Open, deliberately not run here: the same panel at 512² and 1024², where the operators' flat
  cost and the FOM's growing cost would move the crossover. `b-panel` has the ROM/FOM side at
  both meshes already; only an operator phase would need adding.
- Open: the operator arms' errors (4.46–7.42 % worst evolved) sit above the 256² grid's own
  discretisation error of 4.03 %, so on these six cases they are not resolving the physics
  better than the mesh does. Worth saying in any paper table that quotes them.
