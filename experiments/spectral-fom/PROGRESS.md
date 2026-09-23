# spectral-fom — progress (kept current)

Lane `exp/2026-09-23-spectral-fom`, forked from `exp/2026-09-23-poisson-bank-knob` @ 06546331.
Local commits only; never push. Cluster namespace: `/cluster/tufts/paralab/tawal01/specfom_20260923/<attempt>/`.
Design and pre-registration: `DESIGN.md`.

## STATE: LANE ARCHIVED (2026-09-23, user decision; not used in the paper)

All planned problems and meshes are measured, collected and audited. Nothing is running, and the cluster namespace
`specfom_20260923` has been removed.
- The report is generated: `reports/2026-09-24-spectral-fom-vs-nmrom.md`, from `reports/summary.json`.
- The lab-entry numbers are in `reports/lab-entry.generated.md`.
- `check_timing.py`: CHECK PASS.
- Codex review 1: no blockers (see `checks/`).
- The disk-full incident (spN/spO, about 10:30–11:10 EDT) is recorded in DESIGN under "Incident and amendment D".

## Local smokes (GB10, not results)
- smoke1: Poisson 2D, 128², passes validation; audit PASS.
- smoke2: cube, 32³, 4 cases. ROM errors reproduce the lane's c32final per-case values to about 1e-15. CG
  converges to the DST field (1e-10 gives 5.8e-12). Audit PASS.
- smoke3: Burgers, 128² (256² selection), 2 cases.
  - Validation: paper FOM at 1e-10 vs Picard at 1e-10 agree to 1.6e-10 of ‖u0‖; the 1.01ν control is off by 2.2e-3.
  - Audit PASS.
