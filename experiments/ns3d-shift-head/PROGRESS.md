# Progress — ns3d-shift-head

Branch `exp/2026-09-23-ns3d-shift-head` (local commits only; no push per coordinator).
Cluster namespace `/cluster/tufts/paralab/tawal01/nshead_20260923/`.

## 2026-09-23 ~03:00 EDT
- Design written (`DESIGN.md`), amended before any job for the user's direction change:
  corrections dropped; primary arms = (a) head only in the frame (z, delta),
  (b) importance-ordered bank span R' in {64,48,32,16,8} (c, delta).
- FOM finding: the parent's CNAB2 is Fourier pseudo-spectral — no linear solve at all
  (viscous step = pointwise division in Fourier space, pressure = Leray projector).
  Added `fd_fom.py`: 2nd-order FD, CN viscous step by CG, exact discrete projection by CG.
- Local gates pass (`test_head_solver.py`, parent `test_fast_solver.py`).
- Job `a1_h32` (4196941, A100) submitted: 32^3 first check.

## 2026-09-23 ~03:30 EDT — 32^3 check landed (job 4196941, A100-40GB), `results/a1_h32.md`
- Pipeline clean: bank floor reproduces the parent (0.125 %), rotation gate 1e-15,
  frame-zero control fails (39.8 %), neighbour timing gate passes, NumPy audit passes
  (perturbed-copy control rejected).
- Head k=8 inside the frame: 0.151 % evolved worst at dt 0.02 (floor 0.126 %), vs the
  full span R'=64 at 0.602 %; at equal size the span R'=8 is 5.55 %. Head costs
  ~6.4 ms vs 3.5-5.8 ms for the spans at this mesh.
- R' ladder monotone in both error and cost at 32^3; only the head beats CNAB2 at 32^3
  (1.09x). No FD-CG setting (32^3/64^3 FD) is as accurate as the accurate arms.
- Submitted 64^3 (4197294) and 96^3 (4197368, attempt a2 — a1_h96 dir was created by a
  failed sbatch gres request and deleted unused).

## 2026-09-23 ~04:30 EDT — 64^3 v1 (job 4197294): accuracy in, timing gate FAILED
- Accuracy mirrors 32^3: head k=8 0.152 % (dt 0.02), span R'=64 0.616 %, R'=8 5.55 %;
  R' ladder monotone in error and cost; vs CNAB2 2.5-4.5x (spans), 2.87x (head).
- Neighbour gate failed: ladder arms 17-23 % slower right after the 128^3 FD-CG arm
  (fine inside the randomised fast block). FD-CG speedups from v1 are not usable.
- Timing protocol v2 (DESIGN amendment 2); 96^3 v1 (4197368) cancelled before timing.
  Rerunning all meshes under v2: a2_h64 (4198090), a3_h96 (4198101), then a2_h32.
- FD-CG FOM is ~75x slower than CNAB2 at 64^3 at comparable accuracy (1.5 s at 128^3
  for 0.29 %); reported as the rule-compliant comparator, with that caveat stated.

## 2026-09-23 ~05:20 EDT — 64^3 v2 landed (job 4198090), `results/a2_h64.md` — supersedes a1_h64 timing
- All gates pass (fast block, after-heavy+cool-down; after-heavy ratios 1.02-1.06).
- Parent reproduction: bank floor identical, parent linear arm within 1.2e-12 on 6 settings.
- Ladder (dt 0.02, 3 sweeps): R'=64/48/32/16/8 = 0.616/1.005/1.262/2.959/5.549 % at
  7.84/5.80/5.46/4.64/4.18 ms -> 2.46x..4.61x vs CNAB2; head k=8 0.152 % at 8.19 ms, 2.89x vs CNAB2.
- FD-CG never as accurate as the head at 64^3 (best 0.26 % at 128^3, 1.6 s).

## 2026-09-23 ~06:30 EDT — development complete (v2: a2_h32 4198840, a2_h64 4198090, a3_h96 4198101)
- `results/ladder.md` headline. 96^3: head k=8 0.153 % at 15.96 ms (6.67x CNAB2);
  span R'=64..8 = 0.616..5.549 % at 15.39..8.16 ms (6.92x..10.97x CNAB2); monotone in
  error and cost at every mesh. All gates pass; parent reproduction exact.
- FD-CG never reaches the head's accuracy (best 0.277 % at 192^3, 7.7 s at 96^3).
- Settings frozen (`frozen/`, DESIGN amendment 3); held-out job next.
