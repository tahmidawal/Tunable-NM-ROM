# Lane brief: jcp-qualify (priority 2: Q1–Q3 full; B1, B4, B5)

Read `briefs/COMMON-BRIEF.md` first; it binds you. Lane specifics below.

## Where
- New worktree `worktrees/2026-10-18-jcp-qualify`, branch `exp/2026-10-18-jcp-qualify`, created by the
  coordinator from the fork point `21c175a1b` (sparse; `sync_github.sh` configured). Lane dir
  `experiments/jcp-qualify`. Namespace `/cluster/tufts/paralab/tawal01/jcp-qualify/`.
- Concurrent-job cap **3**. GPU-hour cap **35** including contingency (priority-2 share), with
  reservation of queued/running projections (reuse P1's `ledger.py` / `cluster/submit.sh` pattern).

## Build on (read report.md + DESIGN.md)
- `worktrees/2026-10-17-jcp-p1-scaling` (P1: frozen mode lists, Q1/Q2 certifier, split query, timing
  harness, statistics, manifest; 22 configurations already Q1+Q2 qualified at 256²/1024²/4096² for
  R128 fib4181, R256/R384 gauss128 × BE/CN-R at Δt₀; R512 gauss192 at 1024²). **Reuse its code by copying
  with provenance; do not redo what it certified** — cite its certificates.
- `worktrees/2026-10-13-jcp-combined` (2D grid of schemes × Δt factors, finalist table, budget code).
- `worktrees/2026-10-14-jcp-speed3d` (3D g-grid, Q1 certificates and screening front; EXPORT.json).
- `worktrees/2026-10-08-jcp-time2` (temporal order results; LSPG-CN is NOT second order).

## Items owned (plan V2 Group Q + B1, B4, B5; V3-1, V3-5, V4-2, V4-5)
- **Q1–Q3 full (2D):** for every 2D configuration that will enter the B5 front or the C1/C2 comparison:
  r ∈ {128, 256, 384, 512} × schemes {LSPG-BE, LSPG-CN-R, GAL-CN (diagnostic)} × Δt ∈ {Δt₀, 2Δt₀, and one
  larger step per scheme chosen pre-data from combined's grid} × meshes {256², 1024², 4096²}, frozen
  mode lists and frozen rules (plus registered fallbacks where a frozen rule fails Q2). Q1 = V4-2 contract;
  Q2 = AM-2 of `campaign/P1-RESTATEMENT-AMENDMENTS.md`; Q3 conditioning. Unqualified configs reported,
  never ranked. Three labels kept separate.
- **Q2 for the 3D front:** speed3d's reduced-rule arms that will be on the 3D front (its winners and
  front members) are not assumed quadrature-equivalent; give each its own Q2 check (successor rule,
  own reached states, 65³, validation cohort), memory pilot first.
- **B1:** error vs r × N for every qualified 2D configuration (E0 ST primary; H-acc-N / H-acc-r rules of
  AM-4/AM-7 applied to the new configurations, e.g. CN-R at 2Δt₀).
- **B4:** observed temporal order per scheme against a same-form converged anchor, with solver and
  quadrature contributions verified smaller (AM-7 rule). Never award order by method name.
- **B5:** qualified Pareto front of error (median and worst, E0 ST) vs online cost (paired timing in one
  job per mesh, A–B–A with the P1 reference A) parameterised by r, at each mesh, 2D; plus the 3D front
  assembled from speed3d's audited outputs (mesh-free contract, labelled). Failed/unqualified configs
  shown in a separate panel. The front need not be monotone in r.
- **Deliverable for jcp-fom2d:** a frozen `runs/FRONT-2D.json` (each front member's full configuration
  manifest, Q labels, errors per case, and timing) — publish it as soon as it is final and audited,
  commit it, and note the commit in your heartbeat so the coordinator can pass it on. jcp-fom2d runs its
  paired ROM–FOM panels from this file.
- **Defer order** if the cap binds: 4096² larger-step arms → GAL-CN diagnostic → 3D Q2 for non-winner
  front members → 256² larger-step arms. Never dropped: front at 1024² and 4096² for Δt₀/2Δt₀ CN-R and BE.
