# B8 SNF scientific observer qualification

Prospective small fixture: one GPU, maximum15allocated GPU minutes, subprocess
limit600seconds; new owned namespace, backed-up reviewed source, shared two-job
cap. Retain all failure/success outputs. No study fields, scientific training,
dependency changes or submissions occur as part of preparing this configuration.

Run `qualify_snf_boundary.py --out OWNED_OUTPUT --qualification-job
b8snftrainfix --scientific-progress --timeout-seconds 600`. Without the optional
flag, preserve the original boundary qualification behavior. The flagged mode
runs the identical tiny seven native stages with the already qualified boundary
wrapper, additionally applying the exact scientific-progress patch and helper
used by the scientific adapter. Record both hashes. The original passed
boundary qualification remains retained and valid for its original scope.

Require all existing training/boundary checks, exactly seven epoch-progress TSV
rows and seven periodic snapshots (one epoch per native stage). Verify stage,
epoch, batch and actual native update counters, finite/nonnegative wall timings,
and finite reported losses. Reload each snapshot and check model-only warm-start
metadata: no exact optimizer resume, no serialized optimizer/RNG state. Execute
both loaded current and best-so-far models on a tiny synthetic input and verify
Float64 finite outputs and boundary-wrapper type. This detects serialization,
name resolution and observer-runtime failures before a long scientific run.

Scientific training now additionally requires the passed retained observer
qualification, by default `--progress-qualification-job b8snfprogress`, with
matching native qualification, exact scientific source patch and helper hashes.
The continuous-boundary flag retains its separate boundary qualification gate.
Passing this observer fixture is neither an accuracy/PDE admission nor evidence
that the210requested training epochs fit the scientific allocation.
