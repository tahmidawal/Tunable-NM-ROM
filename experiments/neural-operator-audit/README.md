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
