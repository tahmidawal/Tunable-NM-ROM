# Handoff — ns3d-grok

Updated 2026-09-21 after the sealed evaluation. Nothing is running.

## State

The sealed cohort (seed 202609203, 32 cases) was opened once by job **4142139**.
The frozen setting was centered POD rank 64, ROM-centroid shift every startup
step, $\Delta t=0.01$. Generated table: `results/diag05.md`. Local NumPy verify
passed. Remote directory deleted. This is a classical centered POD plus an
online shift, not the coordinate-network NM-ROM.

Job 4142080 died at import before any trajectory, because `diag03.py` was not
staged. It did not read the seed. 4142139 is the only opening.

## What is running

Nothing. Do not submit another sealed job and do not change rank or dt from
the sealed errors.

## Next step

If work continues, the remaining block is cost: every startup step still
evaluates the nonlinearity on the $32^3$ grid, so the tracker is slower than
CNAB2. A shift that stays in coefficient space would be the next hypothesis.
Do not retune on seed 202609203.
