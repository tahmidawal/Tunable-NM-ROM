# spectral-fom — progress (kept current)

Lane `exp/2026-09-23-spectral-fom`, forked from `exp/2026-09-23-poisson-bank-knob` @ 06546331.
Local commits only; never push. Cluster namespace: `/cluster/tufts/paralab/tawal01/specfom_20260923/<attempt>/`.
Design and pre-registration: `DESIGN.md`.

## State (2026-09-23, updated after spA/spB)

| problem | spectral solver | ROM settings source | state |
|---|---|---|---|
| Poisson 2D 256²/1024²/2048² | DST-I fft / mm | poisson-bank-knob @ 06546331 | DONE: spA 4206052 (A100 80GB), gates pass, audit PASS |
| Poisson 2D 4096² | DST-I fft / mm | same | DONE: spB 4206053 (A100 80GB), gates pass, audit PASS |
| Poisson 3D cube 32³/64³ | 3D DST-I fft / mm | poisson-bank-knob-3d @ d2c775a9, final cohort | DONE: in spA, gates pass, audit PASS |
| Poisson 3D cube 128³/256³ | same | lane not selected yet (c256 running there) | waiting on the lane |
| Burgers 2D 256²–2048² | DST Picard / IMEX ladder | burgers-bank-knob @ b393fa55 | spC 4206571 running (256², 512² done) |
| Burgers 2D 4096² | same | lane selection pending (bk4096b now running) | waiting |
| Heat 2D 1024²/2048²/4096² | modal CN / modal exp | heat-bank-knob h2d summary @ e45cae7e (code c2fbe50b) | spD 4207215 submitted |
| Heat 3D | same | heat-bank-knob h3d / h3d256 running | waiting |
| L-shape | none (no fast transform on the L-shaped domain) | — | no arm |
| NS 3D | the lane's CNAB2 is pseudo-spectral | ns3d-shift-head @ 708c70fe | recorded: lane-ref/ns3d.json |

**Finding so far.** The DST solver is faster than both of our settings on the square and the cube at every
mesh. Spectral ms / our ms:

| | accurate | fast |
|---|---|---|
| square | 0.10–0.35 | 0.33–0.57 |
| cube | 0.40–0.51 | 0.52–0.59 |

Report: `reports/2026-09-24-spectral-fom-vs-nmrom.md` (generated).

**Pre-registered sub-gate failed.** The "CG converges to spectral" gate requires the paper's CG to certify
its own true residual at rtol $10^{-8}$ and $10^{-10}$. That fails at 1024², 2048² and 4096²: unpreconditioned
f64 CG hits a true-residual floor (4.5e-9 at 4096²). The distance from CG to the DST field still falls
monotonically, to 1.8e-12 / 1.3e-12 / 8.2e-13. The gate is reported as failed and was not redefined. For heat
the CN–CG validation rtol was set to 1e-11 before any heat job (DESIGN H).

## Local smokes (GB10, not results)
- smoke1: Poisson 2D, 128², passes validation; audit PASS.
- smoke2: cube, 32³, 4 cases. ROM errors reproduce the lane's c32final per-case values to about 1e-15. CG
  converges to the DST field (1e-10 gives 5.8e-12). Audit PASS.
- smoke3: Burgers, 128² (256² selection), 2 cases.
  - Validation: paper FOM at 1e-10 vs Picard at 1e-10 agree to 1.6e-10 of ‖u0‖; the 1.01ν control is off by 2.2e-3.
  - Audit PASS.
