# Handoff — ns3d-grok

Updated 2026-09-21 before the coefficient-space development job.

## State

Seed 202609203 stays closed. Seed 202609211 is reserved for one later sealed
draw and has not been generated. Its parameter rows do not overlap training,
development, or 202609203. diag06 is development only: the diag04 tracker
rewritten with a quadratic tensor, coefficient centroid, and Fourier
shift-reproject. A local $N=8$ smoke matched the grid tracker to about
$10^{-15}$.

## What is running

Nothing yet. The next command submits `ns3dgrok_diag06`.

## Next step

Submit diag06. On exit, NumPy-verify, delete the remote directory, and seal
only the method `summary.json` selects. If the selection is null, do not open
202609211.
