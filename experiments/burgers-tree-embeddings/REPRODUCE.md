# Reproduce the new tree-embedding study

This is a separate study, forked from93f399faad67f94ea2b56fe25f13003143e64694. The old burgers-quadtree campaign remains paused. Its vendored utilities were read-only inputs, with hashes and modifications in `manifests/PROVENANCE.json`.

The authoritative scientific contract is the copied `TREE-EMBEDDING-PLAN.md` and `tree-embedding-protocol.json`. The canonical lab log remains `/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md`. `OWNER.json` and the exclusive claim under QuadtreeNMROM identify the new owner. Do not overwrite that claim to reproduce a job.

## Execution order and evidence

1. `data.py` generates exact prefix-stable training/development parameters. Final-test access is guarded and was never enabled. `references.py` generates all five independent space/time ladder levels. `crosscheck.py` checks all alignment cases, both visible fixtures and the largest-allowance development cases against independently discretized central finite differences with CN/Rannacher time integration.
2. `train_global.py` trains the ordered global bank, then separate nonlinear heads. The training-only POD teacher/control and every checkpoint are retained.
3. `train_local.py` freezes the selected16-dimensional parent, initializes independent local coefficients from its missing physical field, and optimizes the assembled field across prefixes/configurations. `train_children.py` freezes the same local checkpoint and fits the nonlinear child head. R1/R2 commands and the single-factor changes are in `configs/REPAIR-R1.md` and `configs/REPAIR-R2.md`.
4. `diagnose.py` supplies direct-prefix controls, optimistic linear spans and longer fitting checks. `evaluate.py` refits at doubled scoring resolution, checks zero-child nesting and independent Jacobian directions, records paired cached-feature component costs, and produces actual4096-by4096 reconstruction witnesses. These are representation diagnostics, not PDE solves.
5. `audit_saved.py` independently rescores retained arrays with NumPy and tests the current-time denominator. `report.py` generates the initial plots and tables from JSON/NPZ evidence. All raw files are listed in per-job retention manifests.

Use the absolute environments from the parent operational rules. Real numerical jobs run on Tufts partition `gpu`, through `tufts-login`, with GPU preflight, float64 and highest matmul precision. `cluster/run.py` submits committed snapshots only, enforces the registered cumulative reservation and two-job limit, and records accounting. Every job has a unique directory and its staged source hashes are checked before execution. `cluster/collect.py` verifies local and remote output hashes and never deletes originals.

The exact invocation for every allocation is in `cluster/ledger.json` and its corresponding `.sbatch` file. Inputs are deterministically regenerated reference jobs or explicitly verified caches in this study's own namespace. A future replay must use a new owned job name and current resource reservation; archived output paths are not silently reused as active caches.

## Important limits

- Global models implement the planned matched-data Tunable separable form. They are not a claim of faithfully reproducing an unchanged historical checkpoint or a complete tuned online solver.
- The first common head implementation also anchors global heads at zero using the same linear-skip plus shifted-MLP formula as children. This is an architecture choice in this recipe, not an assertion that every native Tunable head uses that constraint; unrestricted-bias global heads were not reproduced. The global direct-bank control does not have this nonlinear-head restriction.
- The first local Adam-code recipe had a substantial training-code optimization gap. Its early stop does not establish a feature-learning plateau. R1 explicitly refits codes during weight training; all original failures remain available.
- Code optimization reports finite-start fits and stationarity, not certified global minima.
- Cached-feature field/JVP timings exclude complete initialization, assembly, adaptation, dynamics and requested dense output. They cannot establish a PDE speedup.
- `operators.py` preserves the incumbent tested weak upwind residual and has a dense arithmetic fixture. Production fine-mesh dynamics remain a subsequent gated stage.

## Glossary

- Prefix: the first registered subset of learned feature channels.
- Current-time norm: the norm of the reference field at the same snapshot being scored.
- Retained node: a parent whose independent correction remains active after adding children.
- Prescribed tree: a tree chosen using the known initial-field callable only.
- Oracle: a reference-assisted diagnostic or training allocation, never an online policy.
- Gate: the prospectively registered criterion required to advance a method.
