# Handoff — ns3d-grok

Updated 2026-09-21 after the sealed truncation. Nothing is running.

## State

Seed 202609211 was opened once by job **4148215**. The frozen setting is
centered POD rank 64, startup step, dt=0.01, Fourier tail 1e-6. Generated
table: `results/diag07.md`. Local NumPy verify passed. Remote directory
deleted. The coefficient form matches the grid tracker and is slower than
CNAB2. It is not the coordinate-network NM-ROM.

Seed 202609203 stays closed. Do not open 202609211 again.

## What is running

Nothing.

## Next step

The remaining time is the shift-reproject over the retained Fourier
coefficients of the POD modes. A parity-passing truncation did not make that
cheaper than CNAB2. A shift that is cheap in a vortex-adapted basis would be
the next hypothesis.
