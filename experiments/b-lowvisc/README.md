# b-lowvisc — is there a viscosity at which the nonlinear manifold is worth having?

Read `DESIGN.md` first: it is the pre-registration (the viscosity family, the meshes, the pass
criterion, the falsification clauses, the cohort discipline) and every amendment since.

| file | what it is |
|---|---|
| `DESIGN.md` | the pre-registration and its amendments; nothing here is run before it is committed |
| `lv_common.py` | the one thing this lane changes: `params_draw` with the viscosity bounds as arguments, plus the checks that the low-viscosity cohort is the incumbent cohort with nu/10 and nothing else |
| `make_configs.py` | writes `config-gate.json` and `config-smoke64.json`; every comparator number is read out of `comparators/`, never typed |
| `gate.py` | the gate job: both families, one allocation, one GPU, no trained model. Leg (a) = POD floors; leg (b) = the tuned full-order Newton grid, timed |
| `audit_gate.py` | the independent NumPy audit; imports neither JAX nor the driver, and recomputes the POD floors by the Gram-eigenvector route rather than from the modes |
| `cluster/` | `stage.py` (byte-checked staging + sbatch), `collect.py` (checksum collection), `preserve_archive.py` (bounded Git chunks) |
| `comparators/` | verbatim copies of b-panel's `config-256.json` and `reports/summary.json` (job 3780638), with `PROVENANCE.json` |
| `checks/` | the 64-interval local smoke's `result.json` and its audit, run before the first submission |
| `reports/` | the generated report, `summary.json`, and the substituted written audits |

## Running it

```bash
# regenerate the configurations from the comparators
python experiments/b-lowvisc/make_configs.py

# local smoke (GB10), 64 intervals, validates no number
PYTHONPATH=experiments/mr-burgers2d:experiments/head-ablation:experiments/b-lowvisc \
  JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun /home/tahmid/Dev/.venv/bin/python \
  experiments/b-lowvisc/gate.py --config experiments/b-lowvisc/config-smoke64.json --out <dir>
python experiments/b-lowvisc/audit_gate.py --attempt smoke --root <dir> --out <dir>-audit.json

# stage, submit, collect, audit, archive, then delete the remote attempt directory
python experiments/b-lowvisc/cluster/stage.py lvg01 a100
python experiments/b-lowvisc/cluster/collect.py lvg01
python experiments/b-lowvisc/audit_gate.py --attempt lvg01
python experiments/b-lowvisc/cluster/preserve_archive.py lvg01
```
