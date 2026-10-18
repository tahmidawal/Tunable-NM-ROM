# 1D wave manifold check (CPU, numpy/scipy) — archived from the 2026-09-02 Fable analysis

`wave_manifold_check2.py K RS wrinkle_eps wrinkle_freq` — 1D Dirichlet wave, N=128, CN FOM at
80 substeps, quadratic-manifold decoder on a POD bank (`u = G h(z)`), optional high-frequency
"wrinkle" orthogonal to the data. Arms: A = 08-16 LSPG-Newmark, B = Galerkin at the new time,
C = variational Verlet, D = variational midpoint, P = POD-K Galerkin. `wave_manifold_check.py`
is the earlier draft.

Reproductions on 2026-09-03 with `/home/tahmid/Dev/.venv/bin/python` (raw stdout, ~6 s each):

- `repro-2026-09-03-K6-RS20-smooth.txt` — smooth manifold: all arms sit on the floor,
  A/B energy 0.9995, C/D 1.0000.
- `repro-2026-09-03-K6-RS20-wrinkle0.15-freq3.txt` — wrinkled manifold: A/B lose half the
  energy, C/D keep it to 1e-3, **and none of them recovers the accuracy** (all ~7x the floor).

These are not project results; they are the mechanism check that motivates the 2D cell.
