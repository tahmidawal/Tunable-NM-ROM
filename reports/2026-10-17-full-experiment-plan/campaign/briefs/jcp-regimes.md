# Lane brief: jcp-regimes (priority 4: D1 parameters and IC families)

Read `briefs/COMMON-BRIEF.md` first; it binds you (§11–§16).

## Where
Worktree `worktrees/2026-10-19-jcp-regimes`, branch `exp/2026-10-19-jcp-regimes` (coordinator-created from
`21c175a1b`). Lane dir `experiments/jcp-regimes`. Namespace `/cluster/tufts/paralab/tawal01/jcp-regimes/`.
Cap **3** concurrent jobs, **60 GPU-h** incl. contingency, with reservation.

## Build on
jcp-references (E0 generator, order verification, Richardson ε_ref, the training/dev/val/test/sealed/OOD
cohort generators and their parameter ranges), jcp-qualify (qualified configurations), jcp-p1-scaling,
jcp-speed3d. **Reserved OOD cohorts and sealed cohorts stay closed** — do not open, regenerate or sample
them; new regimes use **newly drawn development cases** with fresh, recorded seeds disjoint from every
existing cohort seed (check against the E0 cohort registry).

## Items owned (plan V2 D1)
- Pre-declared ν ranges and IC families that separate three effects: (a) **parameter extrapolation**
  (ν outside the training range, both sides), (b) **IC-family change** (a family the bank was not trained
  on, at in-range ν), (c) **stiffness** (smaller ν → steeper fronts) — one factor at a time.
- For each regime: new order-verified references (E0 code, three nested levels, per-case order check,
  Richardson ε_ref, written into your namespace, never into the E0 store), then the frozen qualified ROM
  configurations (2D at 1024², 3D qualified subset at 65³), Q1/Q2 per configuration, E0-style errors, and
  the FOM ladder's best family at matched accuracy where affordable (2D only).
- Report where the ROM degrades (error vs distance from the training range) and whether the qualified
  rules still qualify. "Fails outside the training range" is a valid, expected outcome; say it plainly.
- Decision rules frozen in DESIGN.md before data.
- **Defer order:** FOM comparison → 3D regimes → stiffness (c) beyond one step. Never dropped: 2D (a), (b),
  (c) one step each with references and Q1/Q2.
