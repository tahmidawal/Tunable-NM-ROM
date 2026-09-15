# Matched neural-operator pilot

This directory supplies the first FNO capacity screen and independent dataset and
precision checks. It is experiment infrastructure; current campaign state and
scientific findings belong in the canonical root `LAB-LOG.md`.

## Scientific contract

The first Poisson screen uses the independently generated shared training and
validation cases at 256 intervals per axis. Each model receives the source field
and coordinates and produces the complete zero-Dirichlet solution field. Burgers
uses initial field, viscosity and coordinates and directly predicts five later
fields; the supplied initial state is returned exactly. Gaussian generation
descriptors, case IDs and reference solutions never enter model inputs.

The small, medium and large FNO configurations have four layers and respective
hidden-channel/Fourier-mode settings (32,16), (48,24), (64,32). Each starts from
seed 20260914. The initial screen uses 500 maximum epochs, a validation plateau
learning-rate schedule, early stopping after 80 epochs without improvement, and
a two-hour maximum per model. These are initial disclosed settings, not evidence
that the baseline has received exhaustive tuning. Validation curves and all
selected-model validation predictions are retained. Checkpoints are selected by
mean case-maximum relative error against the declared discrete dataset target.
Physical-reference evaluation is a separate subsequent audit.

The larger campaign compares accuracy and complete-query runtime against both
neural operators and efficient FOMs. Poisson must include direct DST as well as
tolerance-tuned CG. Burgers must include calibrated mesh/timestep and convergence
choices. Each eventual speed ratio must use methods measured together on the same
GPU with identical inputs, requested outputs, reference and accuracy gate. This
training worker does not produce a FOM or ROM speedup claim. Historical ROM weights
have unmatched training histories; the ROM owner prepares common-data controls.

The current Gaussian continuum-family pilot generates finer reference inputs from
the known analytic family. It does not establish arbitrary sampled-field operator
generalization. Later nested dataset sizes, resolution transfer, repeated seeds,
TFNO, expanded families and sealed final tests follow model selection.

## Burgers common-data baseline

`fno_burgers01` trains the same FNO family on the Burgers common dataset produced
and verified by the Burgers lane, so the ROM, the FNO and the efficient FOM can
later be compared on the same held-out cases.

The input is the sampled initial nodal field, the viscosity and the coordinate
channels. Generation descriptors, case identifiers and solver-audit sidecars are
offline metadata and never enter the model.

**Time dependence is a direct multi-time output, not autoregressive rollout.**
One forward pass maps the supplied state to all five evolved fields as separate
output channels, and the supplied initial state is returned exactly, so the query
produces the complete six-time trajectory with no error-accumulating stepping and
no intermediate state feedback. The cost is that the output time set is fixed by
training; this design cannot be queried at an unseen time or continued past the
last trained time. Because there is no rollout, there is no rollout error growth
to report; per-time errors are still retained so any growth across the requested
times is visible.

Accuracy uses the Burgers lane's own physical metric: the maximum over the six
requested output times of the interior field discrepancy divided by the interior
norm of the supplied initial field. `check_burgers_error_definition.py` imports
that lane's `fixed_initial_errors` by path and proves the two implementations
agree on contract-valid fields; `checks/burgers-error-definition.json` records
the proof and the imported source hash.

`configs/burgers/` fixes an equal 200-epoch budget for the small, medium and
large capacities so the capacity comparison is not confounded by epoch count;
each still carries a wall-clock cap, and truncation is recorded per run.
`worker_burgers.py` then selects the best capacity by validation mean
case-maximum error and repeats it at a lower learning rate, if the remaining
budget allows, before running the timing block.

**Training is deliberately not resumable.** Each training run is bounded inside
one allocation by its own wall budget and by a global deadline that always
reserves time for timing. An interrupted run is reported as truncated at its
recorded epoch; it is never described as resumed.

`timing.py` measures the complete query from the supplied initial field already
resident on the GPU to the complete trajectory resident on the GPU, with a
dedicated GPU burn-in before every timed block, synchronisation around every
repetition, host transfer timed as a separate block, and every repetition
retained in `timing.npz`. Its module docstring is the protocol of record for the
later interleaved ROM/FNO/FOM confirmation job. Timings from different jobs are
never divided by one another.

`collect_burgers.py` is the bounded local monitor for that job. It recomputes
every validation error independently with NumPy, re-derives the timing medians
from the retained repetition arrays, checks that the supplied initial state is
returned bitwise, verifies checkpoint hashes, preserves the archive as Git
parts, and only then removes the exact remote job directory. It never touches
the shared `pilot-data01` cache, which belongs to the Burgers lane.

## Precision and environment

The runtime uses float64 and complex128. The official NeuralOperator 2.0.0
`SpectralConv.forward` allocates a complex64 buffer regardless of the input dtype.
`spectral_conv_f64.py` copies that method under its MIT license and changes exactly
that allocation to preserve the input FFT dtype. The module hash pins the reviewed
upstream implementation. Explicit model conversion preserves complex values;
`Module.double()` alone would leave complex parameters at complex64.

`smoke.py` checks every floating intermediate during forward, backward and the
optimizer step, both PDE output contracts and exact checkpoint reload parity.
`training_smoke.py` exercises two tiny epochs through the actual training driver.
These fixtures are implementation tests, not physical accuracy evidence.

The cluster installation adds only previously absent distributions using pinned
requirements and `--no-deps`; existing JAX and NVIDIA distributions remain at their
original versions. Torch 2.11.0+cu128 declares older cuDNN/NCCL/NVSHMEM pins than
the preserved environment and an absent cuda-toolkit metapackage. `pip check`
therefore does not pass; its full output is retained alongside installation
provenance. Successful allocated-GPU numerical checks are required before this
specific runtime combination is used. No claim of satisfying the upstream pinned
dependency environment is made. The installation also retains pre-existing
unrelated missing-dependency findings.

## Running and retaining work

Use the absolute interpreter paths specified in repository `AGENTS.md`.
`stage.py` creates a content-hashed code bundle for one fixed Poisson worker.
Copy it directly into the corresponding paralab job directory; its data must be
generated on the cluster or copied from an already verified cluster generation.
Check `squeue` before and after submission. `worker.py` executes precision checks,
the tiny training smoke, then the three capacities in fresh sequential processes.
It exits after its fixed queue. Signals request an epoch-boundary checkpoint.
Training checkpoints are retained every epoch; continuation support is not yet
implemented, so an interrupted run must not be described as automatically resumed.

The original Poisson index files preserve absolute calibration provenance paths.
`copied-data-audit.json` records their relocation into the worker data directory;
model-facing case and reference sidecar hashes are unchanged. A checksummed
archive must be pulled and verified before deleting the exact completed remote
job directory. Use only the repository's explicit-ID cancellation script.

`collect_when_done.py` is a bounded local monitor for the owned Poisson FNO job.
It waits for Slurm completion, verifies the transferred archive and source hashes,
checks completed-model fields and checkpoint hashes independently with NumPy,
then preserves the archive in ordered 64-MiB Git-tracked parts before exact remote
cleanup. Reconstruct by concatenating the filenames in `archive-parts/manifest.json`
order and checking its aggregate SHA256 before extracting. Missing model results
remain explicit. A failed collector leaves the remote evidence intact; inspect
`checks/collection-monitor.log` and the canonical lab log for collection outcomes.

## Glossary

- **FNO:** Fourier neural operator, a network using learned Fourier-space layers.
- **FOM:** full-order numerical PDE solver.
- **ROM:** reduced-order model operating in a smaller representation.
- **DST:** discrete sine transform, used for the direct Poisson solver.
- **CG:** conjugate gradients, an iterative linear-system solver.
- **Case:** one independent physical input, including its full trajectory.
- **Validation:** development cases used to choose settings and checkpoints.
- **Intervals:** grid cells along an axis; there is one more nodal point.
- **Relative error:** field discrepancy divided by the specified reference scale;
  Poisson uses solution norm, Burgers uses the initial-field norm.
- **Case-maximum error:** maximum error over one case's requested output times.
- **Capacity:** hidden-channel count and Fourier modes retained by the network.
- **Epoch:** one pass through the training cases.
- **Checkpoint:** saved model, optimizer, normalization and random-generator state.
- **f64:** double-precision real arithmetic; complex128 is its complex counterpart.
- **Smoke:** a small test of implementation behavior, not a research result.
- **Hash:** a checksum identifying exact file content.
- **TFNO:** tensor-factorized FNO, reserved for the later efficient-baseline study.
- **Fixed-initial error:** field discrepancy divided by the norm of the supplied
  initial field, the normalisation the Burgers lane uses for every method.
- **Autoregressive rollout:** predicting each output time by feeding the previous
  prediction back in; not used here.
- **Burn-in:** untimed repetitions run before a timed block so the GPU clock has
  already ramped when measurement starts.
- **Complete query:** everything charged between the supplied input and the
  requested output, with nothing precomputed outside the timed region.
