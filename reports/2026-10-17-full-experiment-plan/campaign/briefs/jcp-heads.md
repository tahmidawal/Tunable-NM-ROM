# Lane brief: jcp-heads (priority 3: C5 — linear span vs nonlinear head; K sweep)

Read `briefs/COMMON-BRIEF.md` first; it binds you (note §11, §12).

## Where
- Worktree `worktrees/2026-10-18-jcp-heads`, branch `exp/2026-10-18-jcp-heads` (coordinator-created from
  `21c175a1b`). Lane dir `experiments/jcp-heads`. Namespace `/cluster/tufts/paralab/tawal01/jcp-heads/`.
  Cap **2** concurrent jobs, **35 GPU-h** incl. contingency, with reservation.

## Why
The advisor asked whether cost depends on "k". The campaign default reads k = bank width r (priorities 1–2).
This lane covers the other reading: k = **K, the nonlinear head's latent size**. Plan V2 C5: the headline
results use the linear span; the claim for the head is either measured under pre-registration or the paper
drops "nonlinear manifold". Governing plan E3c (`reports/2026-10-06-jcp-offmesh-paper-plan.md:87`) lists
the head-repair hypotheses.

## Items owned
- **K sweep, K ∈ {4, 8, 16, 32}**, 2D (the incumbent head is K = 16 on the R512 bank): train **new matched
  heads** (a K sweep cannot be made by slicing the trained K = 16 input — audit 2 C5), same bank, same data,
  same training budget and recipe, seeds recorded (retrains are not bitwise reproducible). Then the
  head-based ROM (latent K + linear correction rank as deployed) at 1024², and cost vs K at
  256²/1024²/4096² (H-cost-N analogue for K), error vs K (E0 ST, Q1/Q2 per configuration).
- **Linear span vs head at matched online dimension:** for each K, compare against the linear-span ROM at
  r = K and at the r giving equal online cost; report which dominates (three-way, TOST for "matches").
- One-ingredient-at-a-time head diagnostics only if cheap (E3c hypotheses); otherwise list them as open.
- **Defer order:** 4096² cost panel → K = 32 → matched-cost comparison at 256². Never dropped: K ∈ {4, 8, 16}
  trained, error vs K and cost vs K at 1024², linear-vs-head at r = K.
