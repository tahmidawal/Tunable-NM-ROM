# Separate native adapter control

This is an unexecuted source candidate. It consumes the one actual coarse DFG
producer control from job 6613426 and never treats it as admitted training data.
The reviewed production loader is unchanged.

`native_adapter_control.py` uses a distinct `ControlView` with only the reusable
geometry, mapped evaluation and metric methods. It exposes no production
`open`, `build` or `export` entry point. It checks the binary mesh and ordered
layout, recomputes the actual A8 normal certificate, replays the saved solution,
checks affine gradients and scalar/vector component maps, compares sampled and
assembled dimensionless metrics, and runs the fixed bounded scalar eigen routines.

There is no fabricated nine-state family, bank/POD fitting, reference accuracy
claim, changed scientific threshold or memory-cap increase. A successful result
would qualify these native implementation primitives only. Actual pressure POD,
supremizers, mixed reduced stability and end-to-end solves remain dependent on
genuine G2 data and their runtime checks.

The exact selected input files and source hashes are in `SOURCE-CANDIDATE.json`;
the proposed request and resource projection are alongside it. The runtime request
uses a relative `control/` directory containing those exact bytes. The source has
not been staged or submitted. An independently reviewed launcher must retain
the standard job-owned library prefix, GPU/f64/highest preflight, GPU partition,
single-job directory, CPU/thread policy and allocation deadline.

The script retains full field replay, normal certificate, sparse metric matrices,
eigenvectors and JSON diagnostics, including partial output on a check failure.
All arrays are diagnostic evidence, not a training export.

## Glossary

DFG identifies the fixed cylinder-in-channel case. G2 is physical reference/data
qualification. A component map identifies how scalar coefficients occupy vector
finite-element coefficients. H1 includes both values and gradients; mass/L2
includes values only. The eigen routines construct candidate test functions;
their residual checks do not prove complete spectral enumeration.
