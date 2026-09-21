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

Kept or reverted. Not applied. The measurement is `results/diag01.md`.

## 2026-09-21 — center the bank on the initial field

Hypothesis. diag01 shows that a fixed linear subspace needs rank 3072 before
CNAB2 stays under 5%, while a per-time oracle shift reaches that bar at rank
64. The oracle center is not available online. The centroid of $u_0$ is
known, and the true center moves only a few hundredths of the period, so
freezing it may keep the rollout under 5% in a rank-64 centered POD.

Change. `diag02.py`. No change to the production NM-ROM.

Measured effect. Pending `diag02`.

Kept or reverted. Not applied. The measurement is `results/diag02.md`.
Freezing the centroid of $u_0$ is already past 5% at the first saved time, so
it is not kept.

## 2026-09-21 — re-center from the ROM field every step

Hypothesis. A shift of a few hundredths of the period takes the field out of
the centered subspace, but one time step moves the center by much less. Taking
the energy centroid of the decoded field after every startup step, and
re-projecting in that frame, should stay near the per-time oracle floor. The
same loop driven by the true centroid is the control. Re-centering only at the
saved output times should still miss, because diag02 already does at that lag.

Change. `diag03.py`. The substep is the startup step of the production
Galerkin runner, not multistep CNAB2.

Measured effect. Pending `diag03`. The measurement is `results/diag03.md`.
Every-step ROM-centroid tracking at ranks 64 and 128, and at both
$\Delta t=0.004$ and $0.01$, has 0/16 development cases over 5%. Re-centering
only at the saved times stays over 5%. Kept as the mechanism. The Python-loop
times in that job are not a query cost.

## 2026-09-21 — fuse the tracker and time it against CNAB2

Hypothesis. The same startup-step-plus-recenter scan, compiled as one
trajectory, matches the Python tracker and is the query whose median belongs
next to CNAB2 from the same allocation.

Change. `diag04.py`. Smoke agrees with the Python tracker to $10^{-15}$.

Measured effect. `results/diag04.md`. The fused scan reproduces the diag03
ROM errors. On this development job the fastest setting with zero cases over
5% is rank 64 at $\Delta t=0.01$. Its median is 14.853 ms; CNAB2 at
$\Delta t=0.01$ on the same job is 2.845 ms. Those times are separate calls
from the accuracy loop, so the sealed job repeats accuracy and time from the
same calls. Kept.

## 2026-09-21 — one sealed evaluation of that frozen setting

Hypothesis. Rank 64 and $\Delta t=0.01$ is the fastest development setting
with no case over 5%. The sealed cohort either stays under 5% or shows the
held-out miss. The FOM comparator is the fastest tested CNAB2 step whose
sealed error is no worse than this tracker, chosen after the run from the
saved table, not before.

Change. `diag05.py` with `--final`. Rank and dt are constants.

Measured effect. `results/diag05.md`, generated from the sealed summary.
Held-out evolved worst is under 5% with 0/32 cases over the bar. The script
picks CNAB2 at $\Delta t=0.01$ as the fastest eligible comparator. The paired
speedup is below one: the tracker is slower. Kept as the accuracy result for
this classical tracker. It is not a change to the coordinate-network NM-ROM.
