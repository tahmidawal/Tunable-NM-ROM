# spectral-fom — progress (kept current)

Lane `exp/2026-09-23-spectral-fom`, forked from `exp/2026-09-23-poisson-bank-knob` @ 06546331.
Local commits only; never push. Cluster namespace: `/cluster/tufts/paralab/tawal01/specfom_20260923/<attempt>/`.
Design and pre-registration: `DESIGN.md`.

## State (2026-09-23)

| problem | spectral solver | ROM settings source | state |
|---|---|---|---|
| Poisson 2D 256²/1024²/2048² | DST-I fft / mm | poisson-bank-knob @ 06546331 | job spA 4206052 (with cube) running |
| Poisson 2D 4096² | DST-I fft / mm | same | job spB 4206053 pending |
| Poisson 3D cube 32³/64³ | 3D DST-I fft / mm | poisson-bank-knob-3d @ d2c775a9, final cohort | in spA |
| Poisson 3D cube 128³/256³ | same | lane not selected yet (c256 pending there) | waiting on the lane |
| Burgers 2D 256²–2048² | DST Picard / IMEX ladder | burgers-bank-knob @ b393fa55 | driver + local smoke OK; submit when a slot frees |
| Burgers 2D 4096² | same | lane selection pending (bk4096b pending on H200) | waiting |
| Heat 2D/3D | modal CN / modal exp | heat-bank-knob (h2d/h3d running) | waiting on the lane's selection |
| L-shape | none (no fast transform on the L-shaped domain) | — | no arm |
| NS 3D | lane's CNAB2 is pseudo-spectral | ns3d-shift-head @ 708c70fe | record only |

## Local smokes (GB10, not results)
- smoke1: Poisson 2D, 128², passes validation; audit PASS.
- smoke2: cube, 32³, 4 cases. ROM errors reproduce the lane's c32final per-case values to about 1e-15. CG
  converges to the DST field (1e-10 gives 5.8e-12). Audit PASS.
- smoke3: Burgers, 128² (256² selection), 2 cases.
  - Validation: paper FOM at 1e-10 vs Picard at 1e-10 agree to 1.6e-10 of ‖u0‖; the 1.01ν control is off by 2.2e-3.
  - Audit PASS.
