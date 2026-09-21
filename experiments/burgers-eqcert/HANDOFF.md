# HANDOFF — burgers-eqcert (kept current; read this first after an interruption)

**Lane:** `experiments/burgers-eqcert/` in `worktrees/2026-09-21-burgers-eqcert`
(`exp/2026-09-21-burgers-eqcert`, fork `exp/2026-09-20-hires-burgers` @ `0ab60014`). Cluster namespace
`/cluster/tufts/paralab/tawal01/bcert_20260921/` (NOT created yet). Contract:
`reports/2026-09-20-speed-accuracy-campaign-protocol.md` (on main). Budget: ≤ 4 GPU jobs total, ≤ 2 running,
account-wide ≤ 6 running.

## State (2026-09-21 11:05 EDT)

- **PAUSED by user instruction (via coordinator) before any work product.** No GPU job submitted, no
  cluster directory created, nothing running, lab log untouched. Jobs used: **0 / 4**.
- Done so far: read the protocol, CLAUDE.md, hires-burgers HANDOFF/DESIGN (incl. A1)/SPEED-LOG, `hires.py`,
  `hops.py`, `audit_hires.py`, `cluster/{stage,collect}.py`, and the paper's Table 1 / Table D.2 sources
  (`paper/main.tex`, `paper/tables/TH_headline.tex`, `paper/tables/T09_eq_ladder.tex`, `paper/gen_headline.py`).
- DESIGN.md NOT written yet; no Codex audit yet.

## Goal (from the brief)

Accurate q=256 Burgers 2D rung with a CERTIFIED rule (primary bar ρ_max ≤ 0.116 on held-out reached states,
never NNLS fit residual) at 256², 512², 1024², timed in the same job as Newton–BiCGStab, FOM chosen by the
paper rule (fastest tested FOM, converged every step, worst evolved error ≤ the accurate setting's); with
chol/clip/lamcarry/pred2, q=0 fast and q=128 alongside; dev6 and (if budget) hold64; `bad0` control kept.

## Findings from reading (no new numbers)

- Paper Table 1 today: 256² 0.51 % at 0.043×, 512² 0.55 % at 0.068× (both the b-eqtop stored rule, marginal
  1/5 re-draws, Table D.2), 1024² 0.59 % dense at 0.0037× (b-panel job, A100).
- In hires-burgers the lattice rule `lat64` (63×63, m=3969) at q=256/M=1088 had held-out ρ_max 0.1000
  (2048²) and 0.0908 (4096²), always attained at held-out state index 357 = 7×51 = the **initial-fit state
  w0 of trajectory 15**. ρ rises as the mesh coarsens, and the local 256² exploration in
  `experiments/hires-burgers/DESIGN.md` §4 saw ρ_max 0.126 on w0 → **lat64 may FAIL the bar at 256²**; that
  would be reported as a failure, not softened.
- `lat128` failed badly at 2048²/4096² (ρ_max ≈ 0.62, same w0 state): likely aliasing of the bank's
  training-mesh (256²) content near sine index 2s = 256. Not a promising fallback without checking.

## Planned next steps when resumed (not yet started)

1. Write `experiments/burgers-eqcert/DESIGN.md`: reuse `hires.py`/`hops.py`/`hfast.py`/`audit_hires.py`
   (copy with recorded source commit 0ab60014) with configs at L = 256, 512, 1024; population mesh = target
   mesh (dense query affordable ≤ 1024²); candidate rules at q=256/M=1088 pre-declared: `lat64`
   (deterministic, certify per mesh), `lat128`, lattice-support NNLS refit with 5 fit-state draws (random
   construction → confirmed only if 5/5 pass), b-eqtop `scaled` rule for reference, `bad0` control; optional
   second disjoint held-out population (trajectories 16–23) as a replication of the certificate. FOM grid =
   hires-burgers grid incl. `lean_nt3e-3_l3e-3`, coarse FOM at L/2, L/4, refined reference 4L (deadline-guarded).
   Selection: cheapest certified non-control arm ≤ 1 % on dev6; hold64 reported, never used to choose.
2. Cheap local check (GB10, sub-minute): ρ of lat32/64/128 on the w0 states of trajectories 8–15 at
   L = 256/512/1024, to know before submission whether lat64 fails at 256² (exploration, not a result).
3. Codex read-only audit of DESIGN.md; record it.
4. Jobs (≤ 4): one per mesh (dev6 timed 5 reps, then hold64 1 rep in the same allocation), 1 spare.
   Cluster stage/collect scripts need NAMESPACE → `bcert_20260921` and LANE → `experiments/burgers-eqcert`.
