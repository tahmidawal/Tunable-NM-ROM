Reviewed staged commit **`025b11f99cfb8831265d508b2f3023c2898936a8`**. No files modified, jobs submitted, SSH connections, or GPU computations performed.

| Item | Verdict | Finding |
|---|---|---|
| 1. Sbatch rules | **CORRECT** | Both use `gpu`, the absolute paralab venv, f64, `highest`, GPU preflight, per-job directories, checksum checks, and paralab output/cache paths. A1’s meshes execute sequentially. `submit.sh` enforces the lane cap with a locked, fail-closed queue check and checks `squeue` before/after submission. Current queue state was not inspected. |
| 2. Dependencies and commit equality | **CORRECT** | All **16 A1** and **23 A2** staged source/input files match their committed blobs and provenance hashes. Both manifests pass. The resolved trees include `engines`, `iterative_paths`, `sep_common`, all required quad2d modules, checkpoint, rotation, EQ rule, and A2’s `offmesh → common/tables → b3d_common` chain, bank, rules and state archive. |
| 3. References | **NEEDS-RESTATEMENT** | The contract is satisfiable using the files listed in `REFS.json`; however, **the local staged directory contains no `refs2d/`**. The existing submit wrapper copies those files separately. Cohort byte-hash portability also needs qualification, below. |
| 4. Memory | **CORRECT**, static assessment | The dominant allocations fit the requested resources, including construction copies—not merely the final `Psi`. This is a feasibility assessment, not a measured peak-memory certification. |
| 5. Outputs answer DESIGN | **CORRECT** as data collectors | Both jobs retain the evidence needed for the registered analyses. Scientific acceptance, slopes, bootstrap labels and historical reproduction remain report/results-audit responsibilities. This does not clear `make_report.py`’s earlier audit findings. |

Reference checks:

- All **77 source-file checksums** pass: 76 reference NPZs plus `result.json`.
- The manifest reports complete/all accepted and matches mesh 8192, both timesteps, solver tolerances and acceptance threshold.
- Every required dev6/val32 × ST/S entry exists, is accepted, meets the residual threshold, and has finite `(6,257,257)` fields matching its recorded field hash.
- [qstudy.py:217](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/q2d/qstudy.py:217) correctly resolves `refs2d` to `$TASK_ROOT/refs2d`.

**Cohort qualification:** local regeneration produces different byte hashes because one viscosity value in each cohort differs from the historical descriptors by one ULP: maximum differences **3.47e−18** and **6.94e−18**. Both historical cluster jobs, `dv256` and `dv1024`, used the byte-identical `engines.py` and produced the reference manifest’s hashes. Thus this is not evidence of a different cohort or an inevitable cluster failure; fresh cluster compatibility remains unverified. The exact-hash assertion will stop execution if it differs.

Memory accounting:

| Allocation | Size |
|---|---:|
| A1, L1024 acc: nodes `Psi` | **11.98 GiB** |
| Nodes `Gq + Gs + Psi` | **17.96 GiB** |
| Gauss640 `Gq + Gs + Psi` | **7.03 GiB** |
| Mesh bank + nodes + Gauss640 + Gauss96 | **28.15 GiB** |
| Same, including duplicate nodes blocks during concatenation | **46.11 GiB** |
| A2, n257: host `U + DU`, 300 states | **74.12 GiB** |
| A2, L4096: host `U + GS`, 396 states | **98.95 GiB** |

A1 has room for the dense tangent batches and gate temporaries on A100-80GB; host truth fields add approximately **1.78 GiB**. A2 releases the large host pairs between meshes and streams bank evaluation/state projection. Its two maxima are not simultaneous; `--mem=240G` is adequate.

Code/output findings:

- **A6-1b G1 — CORRECT.** [qstudy.py:129](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/q2d/qstudy.py:129) implements second-order differences at `1e-5` and `1e-6`, the `[30,300]` ratio, fine-error limit `1e-5`, and `≤1e-9` alternative. The relative-error helper rejects nonfinite arrays and reference magnitudes below `1e-8`; the result is recorded and asserted.
- **A1 label evidence — CORRECT.** Running `gref` first ensures both dense and nodes rows contain per-case `vs_gref_restricted_evolved`. Rows include finite flags and exit counts. `targets_valid` combines the original-population and nodes-reached target checks.
- **A2 — CORRECT.** `a2_gap.py` is unchanged since the reviewed commit. The staged archive has the registered finite state populations and unique labels. Outputs retain per-state/mesh/stencil gaps, fixed tests, Gauss-check rho, independent-family rho for fixed and own-mesh populations, and manufactured controls with leading-coefficient norms and vector discrepancies.

Remaining WRONG: none