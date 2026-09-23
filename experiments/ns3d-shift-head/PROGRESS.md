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
