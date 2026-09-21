# Audit of diag01 before the first GPU job

The Codex CLI filesystem sandbox failed (`bwrap: loopback: Failed RTM_NEWADDR`).
A piped read-only review therefore saw an older copy of `diag_floor.py`. That
copy is not the submitted script. The dispositions below are a self-audit of
the file that the local smoke actually ran.

Kept from that review, and now in the smoked script:

- Saved fields and stored errors must be finite. A NaN gap is a failure in
  `verify_diag.py`, and `diag_floor.py` rejects nonfinite trajectories.
- Oracle-shift reconstructions and one affine-PCA reconstruction are saved and
  recomputed by `verify_diag.py`.
- Full Galerkin trajectories are timed at ranks 64, 256, 1024, and the largest
  rank that exists, after a discarded warmup, with the repetition arrays kept.
- The script requires a float64 GPU backend before it generates data.
- A requested rank above the positive spectrum is replaced by the largest
  available rank, so the extra time-step arm uses that rank.

Already true before the review, and confirmed by the smoke: training-only POD,
development-only evaluation, the final seed is refused, the POD prefix is
QR-orthonormalised before Galerkin, frame 0 must match that projection,
projection error may not rise by more than $10^{-4}$ as rank grows, and the
augmented snapshot matrix is built in memory.
