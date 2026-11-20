# B8 SNF flat GPU parameter-transfer repair

Prospective qualification retry only: one GPU, at most30 allocated GPU minutes,
with qualifier subprocess timeout1200seconds. Use a new owned job/output
namespace, preserve the prior attempt in full, back up source/config before
submission and charge all allocated time. No submission or dependency changes
are made by this configuration.

Retain failed b8snftraincheck, job7387185 (scheduler COMPLETED after1m49s,
scientific result passed=false, Julia subprocess returncode1). It failed before
native optimization: Lux.f64 reconstructed a GPU-backed ComponentArray through
ComponentArrays' CPU-vector append path, triggering forbidden scalar indexing.
This is an execution/precision-port failure, not evidence of poor SNF training.

Change only parameter transfer at both failing sites: the isolated native
train_model patch and the fixture's post-training return to GPU. Construct
`ComponentArray(CUDA.CuArray(Float64.(getdata(p))), getaxes(p))`, preserving the
original flat axes. Transfer model state separately through the existing device
adaptor and audit any floating GPU state arrays. The current Dense/tanh/sine
architecture has empty layer state. Do not run Lux.f64 over GPU ComponentArray
leaves and do not enable CUDA scalar indexing. The fixture explicitly verifies
that its transferred values and axes equal the CPU model values/axes.

The pinned source, Julia binary, Manifest, existing Lux compatibility rule and
isolated depot contract remain unchanged. The qualifier still creates a fresh
copy from the pinned source and records the exact updated patch and fixture
hashes. No package resolution, installation or shared-environment write.

Repeat the same tiny native makedata_SNF/train_SNF/FlatDecoder fixture: invariant
training-only normalization under development perturbation, seven native Adam
updates, finite GPU Float64 data/parameters/gradients/states, directional-gradient
agreement and two improving native Adam steps. Keep every failure and all logs.
Only the actual scientific fixture result determines success; scheduler exit
status alone does not.

A pass qualifies this repaired native training execution path only. Before
scientific SNF comparison training, separately register and qualify a continuous
boundary-conforming decoder wrapper with appropriate training and derivative
checks. Label that variant as a physical-problem port, retain the native-network
control, and do not mistake boundary-node clamping for an equivalent smooth
Dirichlet decoder. No PDE, accuracy, runtime or unchanged-publication reproduction
claim follows from this retry.
