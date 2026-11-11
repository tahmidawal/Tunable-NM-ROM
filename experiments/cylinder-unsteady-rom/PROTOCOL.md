# Rebuttal push protocol: neural bank on the unsteady cylinder (fixed 2026-11-11, before any new outcome)

Scope: continue the G5 ROM lane. Reference, export, quadrature, ROM formulation (rom_core.py), time
step dt0 = 1/3200 and the six targets are unchanged from the finished campaign (report 2026-11-06).

## Cases
- Train Re 90/100/110 (POD teachers only), development Re 95, held-out Re 105.
- Re 105 is the only held-out trajectory. It was opened once in the earlier campaign for the then
  frozen arms (POD r8/16/24, bank r16, frozen bank r4). No arm, FE configuration or setting in this
  push is selected, tuned or inspected on Re 105.

## Metric and gates (unchanged, rom_r4.TARGETS)
e_u RMS <= 1e-2, e_u max <= 3e-2, e_p RMS <= 2e-2, St rel <= 1e-2, CD_max rel <= 2e-2,
CL_max abs <= 2e-2, phase-sensitive over the 1.7 s recorded window, rollout from the projected
first state. Divergence (0.1*||div u||/sqrt|Omega|) is reported, not gated.

## Step 1: diagnosis (development data only)
For the existing r16 bank and POD at the same rank: per-mode regression errors (value, gradient,
divergence), sigma^2-weighted subspace miss against the 24 POD modes, development projection floors,
and the share of regression error and of rollout error by region
(cylinder: within 0.05 of the surface; walls: within 0.03 of a channel wall; wake: x > 0.25
otherwise; upstream: the rest).

## Step 2: training arms (each its own job directory; all recorded, none dropped)
Common: coordinate network with the same boundary envelope; fixed seed 3112026; teacher = POD modes.
- A1 r16-long: original architecture/features (3x128), Adam 20k + L-BFGS to its deadline.
- A2 r16-cap: 4x256, extra streamwise Fourier features (16, 32), long training.
- A3 r24-cap: as A2 at rank 24.
- A4 r24-vp: A3 warm start, then joint sigma^2-weighted subspace (variable-projection) fine-tune.
- A5 r24-stream: divergence-free construction (stream function, u = curl psi), rank 24.
- A6 r24-div: A3 warm start + divergence penalty.
- A7 r24-wake: A3 warm start + loss weight toward cylinder/near wake (x in [0.12, 1.3], factor 4).
- A8+ r32: needs a new export with 32 modes; best r24 recipe(s) repeated at rank 32.
Further arms may be added only as a recorded amendment with a reason. The saved bank is always put
in PCA order by the fixed linear map to the best approximation of POD modes 1..r.

## Selection (development Re 95 only)
Among arms meeting all six targets on Re 95, freeze the one with the smallest e_u RMS; if several
are within 5 % of each other prefer the smaller rank, then the smaller divergence. If no arm meets
all six, freeze the arm with the most targets met, then the smallest e_u RMS, and report the miss.

## Step 3: held-out evaluation (once)
One job: frozen bank arm at full rank and its PCA prefixes, POD at the same ranks, on Re 105:
all six targets, divergence, paired timings (warm-up excluded, repeated, with spread, one allocation).

## Step 4: accuracy-matched FE comparator
FE ladder with the reference solver (same mesh and scheme): order P4/P3, P3/P2, P2/P1 and step
1/1600 ... 1/200 (frozen lag kept at 1/80 s), from the reference state at the start of the Re 95
window (L2-projected to the lower order), scored on the same six targets against the reference.
Comparator rule: the cheapest (wall time per simulated second, 16 threads, timing mode) ladder
entry meeting all six on Re 95 is the accuracy-matched FE. It is then run once on Re 105 (targets
checked, timed in the same allocation as the ROM). Not covered, to be stated: coarser meshes (no
admitted coarser mesh exists) and other time integrators.

## Amendments
(none yet)
