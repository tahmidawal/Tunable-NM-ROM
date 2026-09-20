# NS3D augmented coverage and matched confirmation handoff

This is an operational handoff for a development experiment. Its training and
operator curves are provisional until collected and independently audited; the
final cohort remains sealed.

The scientific source is `ef1d1e9f5c29a2e77444a2d190d9d0c6a70cdf7f`.
`launch.json` records the actual job and submission time. The earlier unsubmitted
stage at `821143cb` is preserved in Git; it was replaced before any job launched.
The isolated remote path is
`/cluster/tufts/paralab/tawal01/paper_ns3d_20260920/coverage04`.
This attempt requests one A100 80GB with the `a100-80G` feature, excludes pax007,
and has a three-hour safety limit. At most one NS job may be queued/running;
the campaign has at most four simultaneous single-GPU jobs.

The config fixes the base physical family, original training and development
seeds, and one deterministic membership table of complete integer-translated
training trajectories. All bank, head, POD and operator models use this same
augmented membership. These are exact symmetry augmentations of the original
base trajectories, not independent newly sampled physical cases. Every velocity
component and output time is translated together; viscosity remains unchanged.
Production and independent NumPy projection, nonlinear operator and complete
trajectory symmetry checks passed before submission. Data regenerate on the GPU
node; no local data directory is synced.

The larger spatial bank remains a learned periodic coordinate MLP. Its frozen
checkpoint contains an untrained placeholder head: do not use that head. Each
selected head has its own checkpoint and correct latent-dimension metadata.
Use `capacity/frozen_bank.pkl` for bank/QR/coefficient arrays and the selected
`capacity/<head_name>.pkl` for head parameters and training codes. Head selection
includes the explicitly retained step-zero affine PCA candidates, fixed-PCA-code
trained heads, and a jointly optimized-code head. Stored-code training error,
affine-PCA development error and fitted-head development error remain distinct.

The four new operators learn the increment from the supplied initial field.
Complete inference adds that same physical initial field back, with its cost
included. This is an explicit residual-output training variant. Older direct
operators and their negative results remain retained in the extra03 panel.
No generator parameters enter the models. A fixed step/wall budget does not
establish training convergence; preserve all curves and selected/latest weights.

The current exploratory larger-bank rollout is conditional on the prospective
representation gates in the config. If eligible it evaluates the frozen selected
head, a fixed-M correction ladder and matched-dimension POD controls using exact
full-grid Fourier extraction of the weak residual. It avoids the large quadratic
tensor but remains grid-bound. Initial fitting, evolution, output and optional
host transfer are charged. Per-step latent histories are retained. Tensor/dense
parity and independent advective-form Jacobian smoke checks passed before launch.

**The final matched comparison is mandatory even if those exploratory gates
fail.** Select the best available current NM-ROM on the augmented training cohort
using only development data, then include it with the same augmented POD and all
four augmented operators plus efficient FOM controls in a bounded final panel.
Preserve and report accuracy/stationarity failures. Do not compare an old
512-case NM-ROM only against new augmented operators, and do not omit the newest
NM-ROM merely because it loses. The earlier panel remains a separate comparison.

The prospective next q extension is recorded in `planned_extended_ladder`:
remeasure the entire longer ladder at one fixed sufficiently large M when the
smaller ladder remains representation-limited. Do not claim a q-only effect by
mixing rows from different M. Bound the next job by measured dense query cost.
The larger initial-condition fit and field readout remain charged at every rung.

The FOM development ladder includes coarse steps preserving all requested output
times. Retain unstable/failed attempts. Freeze the cheapest passing development
choice, all model checkpoints and all solver settings before opening the final
cohort. Never inherit the tight data-generation tolerance as the only comparator.
An independent training seed on the *same augmented membership* is still needed
before claiming seed robustness. Changed bank rank, augmentation or training
membership is not a same-configuration seed replicate.

When the job is quiescent, check squeue and sacct, collect its original `output`,
logs, `OUTPUTS.sha256`, source provenance and batch script directly from the exact
paralab directory. Recompute source hashes against the recorded Git commit and
verify every output checksum. Then run, using CPU BLAS thread bounds:

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python \
  experiments/ns3d/audit_coverage.py experiments/ns3d/runs/coverage04/collected \
  --out experiments/ns3d/runs/coverage04/audit.json
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python \
  experiments/ns3d/audit_dense_histories.py experiments/ns3d/runs/coverage04/collected \
  --out experiments/ns3d/runs/coverage04/history_audit.json
```

The coverage audit recomputes every augmented training projection, all saved
head development fields, fixed PCA codes, and all paired invocation errors.
The dense audit checks every history hash and independently differentiates cold
and first/middle/last solves for every case/weak arm on repetition zero. Its
bounded sampling scope is explicit. A skipped exploratory rollout is recorded as
not applicable; it is not a final-method success.

Create a durable checked archive with `retain_archive.py`, commit the parts and
manifest, then delete only the exact completed remote directory after audits
pass. The helper verifies a full tar roundtrip and releases only its immutable
archive/collected-file page cache, avoiding interference with others' GPU smokes.
Do not alter raw JSON or lower an audit tolerance after observing a failure.

Root owns the canonical lab log and main paper reports. Send audited numbers,
source/job hashes, all failures and remaining work to root. No merges or pushes
are authorized; the user's later merge decision remains necessary.
