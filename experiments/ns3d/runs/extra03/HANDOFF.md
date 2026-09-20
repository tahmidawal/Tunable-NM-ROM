# NS3D larger-bank and four-operator continuation

This is an operational handoff for the running development job, not a results
report. The final cohort remains sealed.

The running attempt is job `3995695`, `ctol_ns3d_920_extra03`, on A100 80GB
node `pax106`; source `8ba0a11ee83aaeb36bb486add6517a51aa921a87`. It has a
two-hour limit and its own namespace:
`/cluster/tufts/paralab/tawal01/paper_ns3d_20260920/extra03`.
GPU preflight and exact data-cohort regeneration hashes passed. Only one NS job
may be queued or running at a time; the shared campaign cap remains four GPUs.
Do not modify the running remote source or cancel it without root coordination.

`output/capacity/screen.json` is the live larger-bank screen. Its fixed prospective
training-only criterion selected rank 1024 from 1024 and 1536. It first trains a
new bank with unrestricted coefficients and initial-normalized relative loss,
then independent heads with latent sizes 32 and 64 and terminal width 1024.
These are representation diagnostics; only a promising measured larger bank
justifies a subsequent rollout panel. The physical family remains amplitude 0.2,
all parameter ranges fixed, all three velocity components and dimensions active.
The new bank's frozen checkpoint and head checkpoints are separate: combine the
selected `head_K*.pkl` parameters/codes with `frozen_bank.pkl` bank/QR when building
a later panel. Never mistake the initial head inside the frozen-bank checkpoint
for a selected head.

After the screen, this job trains the shared DeepONet3D and genuine structured
physics-attention Transolver3D with explicit periodic coordinates and viscosity.
It then benchmarks all four operators, the original frozen R512 NM-ROM ladder,
POD controls and efficient FOM controls together. The original R512 bank/head
accuracy target failed and remains a negative comparison. Parameters loaded
from pickle are device resident before timing. Complete initial fitting and dense
velocity readout are charged. No cross-job runtime ratio is valid.

The prior comparison02 completed, was collected, independently audited, archived
and removed remotely. Its retained local directory is:
`experiments/ns3d/runs/comparison02/collected/`.
The full archive is split into parts under
`experiments/ns3d/artifacts/comparison02/`, with per-part hashes and a full original
output-file roundtrip check in `archive.json`. Raw collected directories are
ignored by Git only because every byte is retained in the tracked archive.
The independent comparison audit and generated summaries are
`experiments/ns3d/runs/comparison02/audit.json`. Every source hash, reconstruction,
reference refinement and saved invocation metric passed; poor scientific accuracy
remains poor. Original `over_5pct` keys inside timing summaries count seconds over
0.05 and have no scientific meaning; use the independent derived summaries.
The CUDA delay-kernel warning occurred during operator training, before complete
query compilation and timing burn-ins, and is disclosed in the retained audit.

When extra03 is quiescent, check both squeue and sacct, collect original output,
logs, OUTPUTS.sha256, MANIFEST.sha256, PROVENANCE.json and REUSE.json directly from
its exact namespace. Preserve the reuse manifest and original source artifacts;
no data/ directory is synced. Generate a partial checksum manifest only if the
scheduler killed the driver before it could produce one, and label it partial.
Never edit original collected JSON to make an audit pass.

Run independent audits locally using the absolute environment path and bounded
CPU thread counts. These audit scripts do not import production JAX solvers:

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python \
  experiments/ns3d/audit_extra.py \
  experiments/ns3d/runs/extra03/collected \
  --previous experiments/ns3d/runs/comparison02/collected \
  --out experiments/ns3d/runs/extra03/audit.json

OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python \
  experiments/ns3d/audit_histories.py \
  experiments/ns3d/runs/extra03/collected \
  --reuse experiments/ns3d/runs/comparison02/collected/output \
  --out experiments/ns3d/runs/extra03/history_audit.json
```

The second audit checks all history hashes, then independently differentiates
cold and first/middle/last weak solves for every case and arm on repetition zero.
Its sampling scope is explicit, not full independent verification of every
step. Frozen-method replay can differ at rounding level because retained states
change compilation; the original fields stay preserved. Investigate failures
instead of weakening gates after observation. New operator training curves,
selected checkpoints and full development errors are separate from the timed
first-eight-case panel. All four operator families need satisfactory monitored
training; a budget cap is never called convergence.

Keep final seed 202609203 unopened until the main coordinator freezes selections.
A second training seed and final frozen evaluation remain outstanding. Root owns
canonical LAB-LOG.md and reports; send root audited facts and source/job paths.
Do not merge or push this branch without the user's later merge decision.

If the larger learned bank still has a substantial development gap, the coordinator
identified an exact-symmetry coverage experiment: deterministic integer periodic
translations of complete training trajectories. The Fourier projection, nonlinear
operator, diffusion and time integrator commute with the same spatial roll. Verify
that property independently, preserve viscosity and all times, and record every
base-case/offset pair. Give identical augmented membership/distribution to bank,
head, POD and all four operators, and report the base and augmented counts and
training budgets. No generator parameters enter predictors, no family range is
narrowed, and neither validation nor final membership is modified. This is a next
attempt proposal; the currently running job has no augmentation.
