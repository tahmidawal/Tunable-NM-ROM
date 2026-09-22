# ops-timing-panel — HANDOFF

**State (2026-09-22): DONE.** Two jobs, both COMPLETED, both audited with **0 failed gates**,
both remote directories and the namespace deleted. Nothing is running.

| attempt | job | what | GPU | elapsed | audit |
|---|---|---|---|---|---|
| `opt101` | 4179247 | 21 JAX subjects + **9** operator arms (FNO, U-Net ×4, Transolver ×4) | A100 80GB PCIe, pax049 | 24:58 | `checks/opt101-audit.json`, 0 failed |
| `opt201` | 4181372 | the same panel + **4 DeepONet arms** = 13 operator arms | A100 80GB PCIe, pax049 | 25:15 | `checks/opt201-audit.json`, 0 failed |

**`opt201` is the reportable job**: it carries the whole 2D operator comparison in one
allocation. `opt101` is kept as the earlier, smaller panel and as the `--previous` reference
that marks which arms are new. The two are never merged into one table.

- Worktree `worktrees/2026-09-22-ops-timing-panel`, branch `exp/2026-09-22-ops-timing-panel`,
  forked from `exp/2026-09-17-no-second` @ `ea812685`.
- Namespace `/cluster/tufts/paralab/tawal01/opstime_20260922/` — created, used, removed twice.
- Jobs used: **2 of 3**. Running: 0. `squeue` checked before and after both submits; exactly
  one job per directory each time.

## The result in one line

Every operator checkpoint in the 2D Burgers comparison now has an admissible cost:
`don-refine` 3.712 ms, `don-small` 3.721, `unet-medium` 5.854, `unet-refine` 5.860,
`don-medium` 5.638, `fno-large` 7.376, `unet-small` 9.431, `unet-large` 10.078,
`don-large` 10.574, `tsol-small` 11.056, `tsol-refine` 11.071, `tsol-medium` 12.813,
`tsol-large` 15.249 ms GPU-query — beside the NM-ROM (fast 38.175 ms, accurate 465.736 ms),
POD-LSPG and the eight Newton–BiCGStab full-order settings, all in one allocation. The
DeepONet arms are the cheapest and by far the least accurate (32.5–36.3 % worst evolved vs
4.46–7.42 % for the others).

## Where everything is

| what | path |
|---|---|
| design + the A1 amendment for the DeepONet arms | `DESIGN.md` |
| Codex design audit (6 findings, all accepted) | `checks/codex-design-audit.md` |
| audits | `checks/opt101-audit.json`, `checks/opt201-audit.json` |
| module-swap parity gate + its log | `checks/parity_families.py`, `checks/parity_families.log` |
| report (source-generated, glossary at the end) | `reports/2026-09-22-ops-timing-panel.md` |
| summary / table | `reports/summary.json`, `reports/table-256.md` |
| generator (reads only the audits) | `reports/generate_ops_panel.py` |
| logs, sbatch, provenance, manifests, per-arm timing JSONs | `logs/opt101/`, `logs/opt201/` |
| copied-file provenance, checkpoint manifest | `COPIED-FROM.json`, `operators.json` |

`runs/` holds the collected field archives (1.2 GB + 1.5 GB) and is **git-ignored**.

## Open / next

- Nothing running; 1 of 3 lane jobs unused.
- Open, deliberately not run: the same panel at 512² and 1024². The operators' cost is flat in
  mesh and the FOM's is not, so the crossover moves; `b-panel` already has the ROM/FOM side at
  both meshes and only an operator phase would need adding.
- Open: every operator arm's error sits above the 256² grid's own 4.0265 % discretisation
  error. Any paper table quoting operator speed must quote that too.
- Open, and **not** a question this lane may answer: whether DeepONet's weakness here is
  architectural or data-limited. All four arms early-stopped with the training loss still
  falling, on 128 training cases. Separating the two readings needs a data-scaling run in the
  DeepONet lane, not a timing job here.
