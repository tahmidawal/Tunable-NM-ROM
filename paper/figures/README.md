# Planned figures and their generators

**Rule, without exception:** no number reaches a figure, a table, or the manuscript by
being typed. Each generator reads the run JSONs named below, writes the figure **and** a
paired `.json` holding every plotted value, and the LaTeX carries `\gen{...}` until that
generator exists and has run. A figure whose generator cannot be re-run from the recorded
JSONs is not a figure this paper may use.

Generators live beside this file, in `paper/figures/`. Each must:

1. take `--evidence <path>` for every JSON it reads and `--out <stem>` for what it writes;
2. refuse to run if a recorded checkpoint SHA, config SHA or job id is missing;
3. emit `<stem>.pdf`, `<stem>.png` and `<stem>.json`;
4. write the exit-reason distribution into the `.json` for any panel showing a solve;
5. carry an inline label for provisional data — single-seed, development-cohort, or
   accuracy-target-failing — so the label cannot be lost between the JSON and the caption.

Status labels follow `paper/OUTLINE.md` §0.

---

## F1 — Architecture and data flow
- **File:** `architecture.mmd` (mermaid; no numbers, so no generator).
- **Supports:** C1, C2.
- **Encoding:** `classDef` colour by *when a quantity is fixed* — trained (purple),
  offline per-mesh assembly (blue), solved inside the timed query (orange), deployment-time
  control (green), supplied or comparator (grey). This is the distinction the previous
  submission blurred, so the diagram is built around it rather than around boxes-and-arrows
  of the network.
- **Status:** **exists.**

## F2 — Operator parity: preassembled versus full-grid weak evaluation
- **Generator:** `gen_fig_operator_parity.py`
- **Supports:** C1, C3.
- **Panels:** (a) relative residual parity by mesh; (b) relative Jacobian parity by mesh;
  (c) positivity audit — fraction of decoded interior nodes below zero, and the
  tensor-versus-sign-upwind departure it produces. Panel (c) is not optional: the
  exactness statement is conditional on the sign restriction and the two must appear
  together.
- **Reads:** the in-job assertion records written by `run_pilot.assemble`
  (`exact_weak_operator_relative_error`, `exact_weak_jacobian_relative_error`) and the
  Burgers tensor gate records (T0 parity, TB chunking reproducibility) from the
  `b2dtensor` run JSONs; E1 outputs once run.
- **Status:** parity data **exists**; the matched three-arm comparison is **not run**.

## F3 — Exact preassembly against a fitted quadrature rule, one frozen checkpoint
- **Generator:** `gen_fig_preassembly_vs_eq.py`
- **Supports:** C3.
- **Panels:** (a) physical error by arm and mesh; (b) complete device-query cost with its
  input / setup / solve / output components stacked; (c) **offline** cost — operator
  assembly and NNLS fitting — on the same axis scale as (b), because the online-cost
  comparison alone is already known to be close and is not a result; (d) memory, $MR^2$
  against the stored rule's cached blocks.
- **Reads:** E2 outputs.
- **Status:** **not run.**

## F4 — Head ablation at matched latent dimension
- **Generator:** `gen_fig_head_ablation.py`
- **Supports:** C4.
- **Panels:** three stacked, deliberately not combined, so the error sources are not read
  as additive: (a) bank projection error; (b) best-found nonlinear fitting error, labelled
  *best found*, never *optimal*; (c) deployed rollout or solve error. A fourth panel gives
  cost at matched error and error at matched cost.
- **Arms:** linear head · quadratic head · neural head · unrestricted bank coefficients ·
  correctly solved linear POD-LSPG, all at matched $k$.
- **Reads:** E3 outputs.
- **Status:** **not started.**

## F5 — Frozen-checkpoint mesh ladder, 64 to 1024
- **Generator:** `gen_fig_mesh_ladder.py`
- **Supports:** C5.
- **Panels:** (a) cached reduced evaluation cost against mesh size — the quantity the
  mesh-independence claim is about; (b) complete device-query cost with components
  separated, so the growing input and output terms are visible next to the flat one;
  (c) iteration counts, which are hardware-free; (d) physical error against mesh, at fixed
  $(k,R,M)$; (e) offline re-assembly cost at each mesh. Direct **DST** and
  **Newton–BiCGStab** on the same axes wherever they apply.
- **Reads:** E4 outputs. Provisionally
  `reports/2026-09-11-iterative-fom-multiresolution.json` — which must be plotted with its
  own labels intact: Poisson and waves miss their accuracy target there, and the Burgers
  exits are stalls with unmeasured stationarity.
- **Status:** **provisional** data exists; the frozen-checkpoint ladder with DST and
  Newton in one job per mesh is **not started**.

## F6 — Deployment-time controls at a fixed checkpoint
- **Generator:** `gen_fig_fixed_checkpoint_controls.py`
- **Supports:** C6.
- **Encoding:** physical error against complete query cost; one marker per operating point;
  **marker style encodes the exit reason** (stationary / residual / cap / stall), so an
  early-stopped point cannot be read as a converged one; the frozen decoder's
  representation floor drawn as a horizontal line, because no solver setting can pass it.
  Connecting line only where the relation is actually monotone; gaps left visible where it
  is not.
- **Reads:** the E5 driver's `index.json` — `summaries/quadrature/*`, `summaries/effort/*`,
  `summaries/validation/*`, plus `rule_selection`, `eq_rules[]` and `config_sha256`.
- **Status:** **prepared, not run.** The driver is currently untracked and must be
  committed before it runs.

## F7 — Common-data comparison against a neural operator
- **Generator:** `gen_fig_common_data_operator.py`
- **Supports:** the paper's framing, and reviewer complaints F11 / M1.
- **Panels:** (a) accuracy by model capacity, ROM and neural operator, on the identical
  dataset and split with the index hashes printed in the caption; (b) an explicitly
  **empty** latency panel carrying the sentence that no paired same-job latency measurement
  exists. Showing the absence is the point: omitting the panel would let a reader infer a
  cost comparison that was never made.
- **Reads:** `worktrees/2026-09-14-no-poisson/.../runs/matched01/matched-summary.json`;
  `worktrees/2026-09-14-no-audit/.../runs/fno_poisson01/field-audit.json`; the Burgers
  arm's outputs once run.
- **Status:** Poisson arm **collected (negative)**; Burgers arm **prepared, not run.**

---

## Tables

## T1 — Per-cell configuration and provenance
- **Generator:** `gen_tab_configuration.py`
- **Columns:** PDE · $N$ · $n$ · $k$ · $R$ · $M$ · $m$ · $\Delta t$ · checkpoint SHA256 ·
  config SHA256 · commit · job id · GPU model · backend · precision.
- **Reads:** the `metadata` / `info` block of every run JSON.
- **Status:** **exists** (the fields are already recorded in the collected JSONs).

## T2 — Exit-reason distribution per panel
- **Generator:** `gen_tab_exit_reasons.py`
- **Columns:** panel · stationary · residual tolerance · iteration cap · small step ·
  damping limit · non-finite. Reason codes differ between the three cells and are reported
  per cell rather than merged; `methods.tex` records the mapping.
- **Reads:** the solver records of every run JSON.
- **Status:** **exists.**

---

## Glossary for this file

- **Generator** — a script that turns recorded run JSONs into a figure or table, so that
  neither can drift from its data.
- **Evidence JSON / run JSON** — the file a solver run wrote, holding per-case errors,
  timings, solver exit records and provenance hashes.
- **Provenance hash (SHA256)** — a checksum identifying exactly which checkpoint, config or
  source file produced a result.
- **Exit reason** — why a solve stopped: a tolerance was met (stationary, or residual), or
  it stalled (iteration cap, vanishing step, damping limit, non-finite value).
- **Representation floor** — the smallest error the frozen decoder can achieve on a field
  regardless of how well the latent problem is solved.
- **Development cohort / sealed cohort** — cases already opened, therefore usable for
  choosing settings; cases opened once at the end with every choice frozen.
- **Provisional** — a number that is real but rests on a single seed, on an opened
  development cohort, or on a panel that failed its own accuracy target; it must carry that
  label everywhere it appears.
