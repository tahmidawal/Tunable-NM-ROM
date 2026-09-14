# Consolidated solver entrypoints

This directory binds the selected accuracy-campaign implementations to their frozen
models and reproducible checks. Project state remains in the canonical
[lab log](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md).
The branch is `exp/2026-09-13-nmrom-consolidated`, created on September 14 using the
previously approved name and corrected heat base `5974d3e0`.

## Selected methods

| PDE | Retained method | Native query and assembly | Model files |
| --- | --- | --- | --- |
| Poisson | `r128_q32`: frozen head with analytically eliminated linear corrections | `correction_core.prepare_correction` / `correction_query`, `core.assemble` | `experiments/multiresolution-poisson/runs/correction_accuracy10/checkpoints/r128_joint.pkl` and `basis.npz` in that run directory |
| Heat | `nmrom_initial_tail` | `cp_algebra_paths.build(...)["nmrom_cholesky"]`, `run_pilot.assemble` | `experiments/mr-heat2d/runs/accuracy10/archive/outputs/checkpoints/initial_tail.pkl` |
| Burgers | Original head with explicit stationarity stopping | `accuracy_paths.make_rom`, `engines.build_rom` / `build_gauss_cold` | `experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl` |
| Reflective waves | `trained_nested40`, guarded Cholesky, selected time step | `acceleration_replay.query`, `pilot.rebuild` and the selected head transformation | `experiments/multiresolution-wave/runs/accel10/cluster/out/pilot/head_trained_nested40.npz` and `initializer_trained_nested40.npz` |

The executable examples in [replay.py](replay.py) supply the accepted solver settings
from the retained raw JSON. They include full input projection, nonlinear solving
or evolution, and dense output. Each PDE runs in a separate Python process because
the native experiment modules reuse names such as `pilot` and `core`.

The heat and Burgers training drivers contain multiple experimental arms, including
rejected ones. Use the selected files above when building the next experiment.
The selected wave uses only the fresh post-reset reflective model. Its native
rebuild inputs are retained under
`experiments/multiresolution-wave/runs/accel12/cluster/in/dirichlet/`.

## Verify the baseline

From the worktree root, verify every imported content hash, recorded Git blob,
Python syntax, and exact regeneration of the campaign and aggregate reports:

```bash
OPENBLAS_NUM_THREADS=1 /home/tahmid/Dev/.venv/bin/python consolidated/verify.py --output consolidated/checks/content-latest.json
```

Replay one saved, already-opened development case on the local GPU. Run the commands
sequentially; each process has a timeout and its own memory-limited scope.

```bash
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun timeout 55s /home/tahmid/Dev/.venv/bin/python consolidated/replay.py poisson --output consolidated/checks/poisson-latest.json
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun timeout 55s /home/tahmid/Dev/.venv/bin/python consolidated/replay.py heat --output consolidated/checks/heat-latest.json
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun timeout 55s /home/tahmid/Dev/.venv/bin/python consolidated/replay.py burgers --output consolidated/checks/burgers-latest.json
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun timeout 55s /home/tahmid/Dev/.venv/bin/python consolidated/replay.py wave --output consolidated/checks/wave-latest.json
```

Use a fresh output filename on subsequent GPU replays. The JSON records backend,
precision, device, manifest hash, field discrepancies and solver checks.
[Recorded checks](checks/) retain the completed baseline verification; the replay
logs are beside this guide. The initial wrapper failures are retained in
`heat-smoke.log` and `poisson-smoke.log` and explained in [verification notes](verification-notes.md).

These are bounded integration checks on the smallest campaign mesh. Poisson, heat
and waves rebuild their mesh operators from frozen model files. Burgers rebuilds
the spatial bank and uses the archived fitted quadrature and initial-fit operators
to keep the smoke bounded. A different mesh, test-mode count or training endpoint
requires rebuilding its operators and refitting EQ from decoder outputs.

The Poisson supplied source is regenerated with the original function and saved
parameters because the run did not retain its full source array. Both the archived
and regenerated hashes are recorded; source bitwise equality across the cluster
and local machine is not established. The saved output is checked numerically.
Heat and waves use archived initial fields directly. Burgers regenerates its
initial field using the recorded physical parameters and original function.

These checks do not constitute new accuracy or timing measurements, a complete
cohort rerun, or final paper validation. The retained Poisson and wave accuracy
misses remain documented in the campaign report.

## Provenance and reports

[manifest.json](manifest.json) records the exact source heads, paths, Git blobs and
SHA256 hashes. Source code is copied without numerical changes from the accepted
PDE branches. The corrected heat implementation and its archive are inherited from
the base. Shared decoder and solver dependencies agree with all source heads.
[import_sources.py](import_sources.py) reproduces this scoped import from the
recorded local source trees and refuses changed source heads.

The selected implementations and model files are available at native paths under
`experiments/`. The original result archives and unsuccessful alternatives remain
preserved in their source branches; full archives are not duplicated by this import.

Canonical reports remain on main. This branch contains exact snapshots of the
[campaign report](../reports/2026-09-11-accuracy-improvements-and-wave-speed.md),
[aggregate errors](../reports/2026-09-11-accuracy-aggregates.md), and
[handoff](../reports/2026-09-11-accuracy-campaign-handoff.md), including their original
dated branch-status wording. `evidence/` mirrors every JSON and additional model/code
dependency read by their generators. It contains ordinary evidence directories,
not Git worktrees. `verify.py` redirects the unchanged generators into a temporary
mirror and compares all generated outputs byte for byte. The historical direct
generator commands and worktree links in those report snapshots refer to the
canonical repository layout; use `verify.py` from this consolidated tree.

## Plain-language glossary

- **PDE:** one of the equation families studied here.
- **Checkpoint / frozen:** saved model parameters / parameters held unchanged during a query.
- **Native query / assembly:** the original solver function / preparation of mesh-dependent matrices.
- **EQ:** empirical quadrature, a learned weighted set of spatial samples used here for Burgers advection.
- **Stationarity:** a small optimization gradient under the recorded stopping rule; it does not guarantee physical accuracy.
- **Replay / parity:** executing a saved input again / agreement of the output within a declared numerical tolerance.
- **Manifest / Git blob / SHA256:** a provenance inventory / Git's identifier for file contents / a content checksum.
- **Development / final validation:** cases available for method selection / later evaluation reserved for the paper.
