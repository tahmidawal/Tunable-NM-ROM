# B8 native SNF expanded-data training

Prospective scientific training attempt: at most 120 allocated GPU minutes, one
GPU, within the shared maximum of two owned pending/running jobs. The Python
adapter caps preparation plus Julia execution at 6900 seconds, reserving time
for final evidence writes. The owner must back up source/config before submission
and retain completed, failed and timed-out artifacts with resource accounting.
No submission is authorized by the presence of this file alone.

Updated prerequisite after the retained transfer failure: `b8snftrainfix` must have passed the actual native
`makedata_SNF` / `train_SNF` fixture, seven native optimizer steps, GPU Float64
checks, development-shift normalization invariance and directional-gradient
check. Its result must have a matching local/remote verified retention manifest.
The driver verifies the pinned sources, qualified patches, Julia binary,
Manifest and Lux rule; it copies the qualified project and uses an isolated
writable depot over the qualified read-only lookup depots. No package resolution.

Invocation: `train_snf_expanded.py --out OWNED_OUTPUT --qualification-job
b8snftrainfix --timeout-seconds 6900`.

The scalar two-dimensional unit-square adapter uses the exact same 160 training
and 52 development trajectories and all 11 output times as the expanded CROM
run. Original and expanded spectral coefficient hashes are checked before
synthesis on the allocated GPU. Training coordinates are all uniform 128²
points, including boundaries. No coarse substitute, validation-to-training
leakage, or final-test fields. Little-endian Float64 binary arrays carry explicit
column-major shapes; Julia converts them into the native JLD2 x/u/t/mu format.
The source and normalization records are retained with hashes.

Offline hypernetwork metadata contains background and viscosity plus two
canonically ordered Gaussian slots, each amplitude, center x/y, tangent/normal
width and sine/cosine of twice the angle. An absent slot is zero. These 16
physical features use training-only means/scales; constant features use scale
one. Native training-only normalized time is the seventeenth input. Native
field/coordinate/time normalization also uses training data only. A later
online solve must extract the neural decoder and fit its code from the common
initial field. Physical metadata or hypernetwork predictions cannot serve as
privileged online initialization.

Native architecture: latent 32, hypernetwork hh=0 and width 64; sine decoder
hd=5 and width 128 (six sine hidden layers, counting the input layer). Native
Adam schedule, E=210, no warmup: seven requested stages of 30 epochs with native
development-based early stopping (default patience6 per stage). These are
requested epochs, not a guarantee that210epochs execute; retain actual logs
and native step counters. Hypernetwork L2=0.001, decoder weight decay
0.01, Lipschitz penalty zero. Seed 2026111991. Both training and validation
batch sizes are explicitly 262144, including callback evaluation batches.
Native loss remains normalized unweighted field MSE plus authored regularizers.
Native initialization retains rand32 bias values converted to Float64; qualified
training, parameters and gradients use Float64.

The data expansion contains 38,207,488 spatial observations. Float64 field,
coordinate and 17-input parameter arrays together require approximately 5.69
GiB on host; allow 10–20 GiB for copies/loaders within the 96 GiB allocation.
Only minibatches move to GPU. Six width-128 activations at batch 262144 require
about 1.5 GiB before backward-pass intermediates. The default native validation
batch would be the full training set, so the explicit batch limit is mandatory.

Retain every native stage checkpoint and final model, logs, normalization,
source provenance and incomplete status after failure/timeout. A scientific-only
observer additionally retains current and best-so-far models at epoch1, every5
epochs, stage end and early stop, using atomic JLD2 rename. These snapshots contain
model/STATS and progress metadata, not complete optimizer/RNG state: they support
explicit model-only warm starts, never exact resume. A timeout before the first
completed epoch may still leave no trained snapshot; a partial stage is not a
finished training run. No silent training-budget extension.

The observer is an exact recorded patch of the verified native src/train.jl,
separate from native-training and boundary qualification patches. It runs only
after the unchanged native callback and early-stopping-state update, before the
unchanged break decision. It never alters parameters, optimizer state, loss,
schedule, patience, random draws or native callback frequency. It copies current
and best models to host only when saving. Each epoch records batch-loop wall
time, callback-plus-selection wall time, elapsed epoch before retention,
checkpoint I/O wall time, actual native step count and both development losses.
These are host wall timings through existing synchronization, not isolated GPU
kernel timings. No artificial workload or extra CUDA synchronize is introduced.
All snapshots and the append-only epoch_progress.tsv are retained and hashed.

Successful execution establishes a trained candidate only. Independent
embedding projection and physical current-time relative-L2 scoring, actual
4096² reconstruction, and a verified complete embedding-space PDE solve remain
required before accuracy/runtime comparison. No representation admission, PDE
admission or CROM/tree superiority follows from the native training loss.

For the fair continuous-boundary comparison variant, the additional retained
qualification and --boundary-envelope flag in B8-SNF-CONTINUOUS-BOUNDARY.md are
mandatory. The unconstrained native-network control remains separately labelled.
The failed b8snftraincheck execution is retained under B8-SNF-FLOAT64-REPAIR.md.
