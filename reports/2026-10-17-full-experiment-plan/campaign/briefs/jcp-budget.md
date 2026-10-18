# Lane brief: jcp-budget (priority 3: B2, B7, B8)

Read `briefs/COMMON-BRIEF.md` first; it binds you (note §11, §12).

## Where
- Worktree `worktrees/2026-10-18-jcp-budget`, branch `exp/2026-10-18-jcp-budget` (coordinator-created
  from `21c175a1b`). Lane dir `experiments/jcp-budget`. Namespace `/cluster/tufts/paralab/tawal01/jcp-budget/`.
  Cap **3** concurrent jobs, **45 GPU-h** incl. contingency (plan V3-6 adds 20–40 for B7/B8), with reservation.

## Build on
`jcp-combined` (its 2D/3D budget method: δ_quad, δ_solve, δ_form, δ_time, e_P, e_dyn; `budget2d.py`,
`budget3d.py`), `jcp-references` (E0 generator `refcore.py`, `pscore.py`, order verification, Richardson
ε_ref, store layout — **write new references only into your own namespace, never into the E0 store, never
touch sealed/ or custody/**), `jcp-p1-scaling`, `jcp-qualify` (FRONT-2D; qualified configurations).

## Items owned (plan V2 B2, B7, B8; V3-6)
- **B2 error budget** for the qualified front configurations (2D per mesh; 3D qualified subset):
  one-ingredient ablations (bank floor, quadrature, time, test space, solver, IC projection, scoring) **plus
  interactions** (pairwise at least for the two largest components). Not an additive decomposition of
  norms; the per-time identity $e_\star^2=e_P^2+e_{\rm dyn}^2$ is a gate, as in combined.
- **B7 longer horizons:** T beyond 0.25 (pre-declare, e.g. T = 0.5 and 1.0) for qualified 2D configurations
  at 1024² (and 4096² if affordable) and the qualified 3D subset at 65³: error growth and stability.
  Needs **new order-verified references** to the longer horizon generated with the E0 code (three nested
  levels, order check per case, Richardson ε_ref; a pseudo-spectral cross-check on a subset) on the
  dev/validation cohorts only, in your namespace. Q1 at the long horizon.
- **B8 physical diagnostics:** boundary error, energy and dissipation balance, overshoots/undershoots,
  IC-projection error, and **output-sampling adequacy**: score on a finer observation route than the
  257²/63³ scoring lattice (full-mesh decode vs a finer reference sampling) to test whether the lattice
  misses fine features.
- Decision rules and must-fail controls (e.g. a manufactured bank-floor ablation; an injected sampling
  alias that the finer route must detect) frozen in DESIGN.md.
- **Defer order:** B7 at 4096² → B8 3D → B2 interactions beyond pairwise → B7 T = 1.0.
  Never dropped: B2 2D at 1024², B7 T = 0.5 at 1024², B8 2D sampling adequacy.
