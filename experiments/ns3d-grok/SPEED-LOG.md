# Speed log — ns3d-grok

No optimisation is accepted yet. Each entry is a hypothesis, the smallest test,
the measured effect, and whether it was kept.

## 2026-09-21 — inherited split, before any new timing

Hypothesis. The published query is slow because every weak step evaluates a
full-grid Fourier residual inside Levenberg–Marquardt, while CNAB2 itself is a
handful of $32^3$ FFTs. Accuracy and cost are different failures: the final
panel's POD weak and POD Galerkin errors match, and the Galerkin form is the
cheap one (about 15 ms versus about 535 ms at rank 64, and about 105 ms for the
free bank, against about 3.2 ms for CNAB2 at $\Delta t=0.01$). Those costs are
cross-job orientations from final07, not a ratio this lane may quote.

Change. None. `diag01` remeasures FOM, Galerkin, and one dense weak query in
one allocation, and records a per-call breakdown of a Galerkin step (decode,
nonlinearity, projection, linear solve).

Measured effect. Pending `diag01`. A local smoke (four $N=8$ trajectories,
5.6 s) only checked that the timer runs: one weak query was slower than one
Galerkin rollout on that toy mesh. Those milliseconds are not a result.

Kept or reverted. Not applied.
