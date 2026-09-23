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
