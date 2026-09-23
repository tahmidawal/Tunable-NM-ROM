# spectral-fom — progress (kept current)

Lane `exp/2026-09-23-spectral-fom`, forked from `exp/2026-09-23-poisson-bank-knob` @ 06546331.
Local commits only; never push. Cluster namespace: `/cluster/tufts/paralab/tawal01/specfom_20260923/<attempt>/`.
Design and pre-registration: `DESIGN.md`.

## State (2026-09-23 ~11:20 EDT)

| problem | state |
|---|---|
| Poisson 2D 256²–4096² | DONE (spA, spB) |
| cube 32³/64³ (final) and 128³/256³ (dev) | DONE (spA, spM) |
| Burgers 256²–1024² | DONE (spF, L1 ladder) |
| Burgers 2048² | DONE (spJ, L1 ladder + half-DST candidate) |
| Burgers 4096² | DONE (spL) |
| Burgers records | spC, spE, spG |
| Heat 2D 1024²–4096² | DONE (spI, L2); record spD |
| Heat 3D 32³/64³ | DONE (spN) |
| Heat 3D 128³/256³ | spP 4218304 pending (amendment D) |
| NS 3D | recorded (lane-ref/ns3d.json) |
| L-shape | no spectral arm |

**INCIDENT, about 10:30–11:10 EDT.** The heat 3D field saving of spN/spO filled the shared paralab share to 100 %.
- Fields were deleted and spO was cancelled; the share is back to about 280 GB free.
- Other lanes' jobs running in that window may have hit disk-full errors.
- See DESIGN "Incident and amendment D". It must be reported in the lab log and the final message.

**Codex review.** A read-only codex review of the lane code is running in the background. It has not concluded.

## Local smokes (GB10, not results)
- smoke1: Poisson 2D, 128², passes validation; audit PASS.
- smoke2: cube, 32³, 4 cases. ROM errors reproduce the lane's c32final per-case values to about 1e-15. CG
  converges to the DST field (1e-10 gives 5.8e-12). Audit PASS.
- smoke3: Burgers, 128² (256² selection), 2 cases.
  - Validation: paper FOM at 1e-10 vs Picard at 1e-10 agree to 1.6e-10 of ‖u0‖; the 1.01ν control is off by 2.2e-3.
  - Audit PASS.
