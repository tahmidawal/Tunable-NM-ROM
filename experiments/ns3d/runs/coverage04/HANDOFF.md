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
The prospective final cohort size is fixed in `configs/final32_protocol.json`
before any draw. All final model/solver choices and hashes must freeze using
development data by the recorded selection deadline. The record also preserves
the mandatory matched negative NM-ROM comparison and the pending seed repeat.

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

## Prepared capacity extension

`configs/capacity05.json` prepares a larger learned-bank warm expansion on this
exact augmented membership. It is not submitted and must not modify this live
attempt. Its machine-readable readiness record, calculated memory estimates,
bounded runtime estimate and independent smoke results are in
`checks/capacity05_readiness.json`. These are estimates and tiny implementation
checks, not larger-grid scientific results.

`expand_bank.py` retains the old coordinate MLP features and copies terminal
weights with explicit component-by-column indexing. It checks actual old
Leray-projected physical columns and measures the new bank's singular spectrum
after projection and QR whitening. New coefficients start at zero, while old
coefficients use the previous physical projection coefficients. The optimizer
and new-column RNG restart. This is a declared warm capacity extension, not an
independent complete training-seed repeat. Final trained whitening must pass;
width or output dimension alone never certifies bank rank.

The extension preserves initialized affine PCA as a competing head. Its trained
heads run only when the larger bank passes their prospective bank-floor gate.
Its longer correction ladder remeasures every rung at one fixed larger test
count. A missed exploratory gate must still lead to an explicitly negative,
bounded final NM-ROM measurement under the final-panel requirement above.

After this attempt is quiescent, checksum-collected, source-verified and audited,
stage the committed future source in its own unsubmitted attempt:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/ns3d/cluster/stage.py capacity05 \
  --driver coverage04.py --config capacity05.json --gpu a100 --a100-memory 80G \
  --hours 2
/home/tahmid/Dev/.venv/bin/python experiments/ns3d/prepare_capacity_reuse.py \
  experiments/ns3d/runs/coverage04/collected experiments/ns3d/runs/capacity05 \
  --audit experiments/ns3d/runs/coverage04/audit.json \
  --history-audit experiments/ns3d/runs/coverage04/history_audit.json \
  --source-audit experiments/ns3d/runs/coverage04/source_audit.json
```

The helper verifies all reused output hashes and the audit gates, copies the
bank and every selected operator checkpoint into `reuse/`, records source
lineage, and rebuilds the attempt manifest. It copies no data. The driver
regenerates base data and checks exact augmented/development hashes and training
statistics before reusing operators. Transfer this complete staged directory
directly to its paralab path. Check disk and the full account queue before and
after submission; keep at most one NS job and the global campaign cap. Record
the actual job, source commit and GPU preflight in a new launch record.

The updated `audit_coverage.py` additionally verifies final projected numerical
rank, independent initial expanded-bank columns/rank, and byte-identical reused
operator weights. Collect the new `reuse/` directory with the future result so
these lineage checks remain possible. Archive the full source/output/reuse
bundle and retain any failed conditioning, accuracy or stationarity outcome.

The prior extra03 archive is committed and its exact completed remote directory
has been deleted. Its strict cross-run replay audit remains failed, unchanged.
`runs/extra03/numerical_validity.json` separately qualifies its independently
checked new paired measurements; no implementation-speed or cross-run parity
claim follows from those measurements.
