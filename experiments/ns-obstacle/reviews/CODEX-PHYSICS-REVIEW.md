# Independent Codex physics/reference design review

Reviewer: separate read-only `physics_reviewer`, 2026-10-27, reviewed central original draft
and discovery sources independently of other reviews; no code edits/jobs.

Verdict: release dependency, geometry and analytic-control implementation; no production
references or training. Core PDE, DFG geometry, pressure convention and force normal are sound.

Findings (original draft line numbers):
- Lines125–128,168: actual alternative outlet required, not length/backflow inspection alone.
  Symmetric traction-free on longer downstream domain is defensible as a distinct BVP;
  compare common local velocities, forces and local pressure differences.
- Lines154–170,290–295: Poiseuille alone has zero convection. Restore nontrivial manufactured
  divergence-free mixed control, including all components in 3D, consistent forcing/traction
  and independent nonlinear Jacobian/spatial convergence checks.
- Lines130–133: dt0.04 does not divide output0.1. Use aligned .05/.025/.0125 or specify and
  verify interpolation/variable stepping before release.
- Lines95–100: finite DFG numerical correctness differs from length-independence. Faithful
  DFG reproduction may pass while expansion/wake interpretation gate fails. Pressure levels
  change under domain extension; compare local pressure differences, not raw offset fields.
- Freeze central executable configs to retire differing 3D discovery proposal. Define finite-
  divergence corrections to balances; physical symmetric identity is continuum-only while
  Taylor–Hood is weakly divergence-free. Gradient balance is an independent control.

Primary sources checked: official Featflow DFG page and Barber–Emerson report
https://epubs.stfc.ac.uk/manifestation/255/dltr-2001001.pdf . Its Table4 normalized drag row
is correctly transcribed with radius-to-diameter Reynolds conversion. Table1 extent is
±10 radii = ±5 sphere diameters; Eq26 prescribes zero axial velocity gradients, not an
established identical pseudotraction condition. Table8 contains nonzero computed normal
stress that should vanish in no-slip limit. Rounded source drag is auxiliary, not a tight
uncertainty certificate. Project Re10 sphere case and no transition-threshold claim are sound.
