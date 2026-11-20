# B8 SNF continuous physical boundary qualification

Prospective fixture only: one GPU, at most30allocated GPU minutes, subprocess
limit1200seconds. Submit only from backed-up reviewed source, in a new owned
namespace, within the shared two-job cap. No dependency changes. Retain all
failed fixtures. This file/implementation does not submit a job.

Prerequisite: retained passed b8snftrainfix7387243, including the actual seven
native optimizer stages, Float64 GPU checks, gradient check and improving native
Adam steps. The new qualifier verifies that retained result, fixture, patched
native source, Julia binary, Manifest and existing Lux rule. It creates another
isolated project/depot and records one exact boundary-source insertion.

Invoke `qualify_snf_boundary.py --out OWNED_OUTPUT --qualification-job
b8snftrainfix --timeout-seconds 1200`. Retain the resulting qualification and
its manifest before enabling the scientific adapter option.

The Lux B8BoundaryDecoder delegates setup directly to the native decoder,
preserving parameter names, flat ComponentArray axes, setup random draws and
state structure. Native FlatDecoder assembly, hypernetwork, optimizer,
hypernetwork L2 and decoder weight-decay indexing remain intact. The boundary
port currently requires Lipschitz penalty alpha=0, matching the registered
recipe; no bound for the wrapped spatial function is claimed.

Decoder inputs contain normalized x/y followed by latent variables. Recover
physical coordinates with training-only coordinate statistics. With
B=16*x*(1-x)*y*(1-y), native normalized output d, and training field mean mu and
scale sigma, output `B*d+(B-1)*mu/sigma`. Denormalization gives
`B*(mu+sigma*d)`. Multiplying d alone would leave a nonzero physical boundary
mean and is incorrect. The entire continuous expression participates in
training and differentiation. Do not divide targets by B, clamp only endpoint
samples, or multiply a finished native checkpoint and call that retraining.

The qualifier reuses the exact retained native seven-stage training fixture,
with this wrapper inserted before FlatDecoder construction. It retains the
normalization-invariance, full parameter-gradient and optimizer checks, then
checks GPU Float64 physical/normalized identity, all four boundaries/corners,
parameter/state identity against native setup, spatial gradient product rule,
Laplacian product rule, and latent central finite differences. Interior and
first-interior-distance1/16384probes are included. Boundary/value tolerance
1e-12; spatial gradient relative1e-10; Laplacian relative1e-9; latent finite
 difference relative1e-6. CUDA scalar indexing remains disabled. No study
solution fields are read by the fixture.

For later registered scientific training, use `train_snf_expanded.py
--qualification-job b8snftrainfix --boundary-envelope
--boundary-qualification-job b8snfboundary --out OWNED_OUTPUT`. The adapter
requires verified retained wrapper qualification, identical underlying training
qualification, exact wrapper hash and exact boundary patch. It copies the
wrapper with the checkpoint artifacts and labels the candidate a continuous
homogeneous-Dirichlet new-PDE port, not an unchanged published SNF reproduction.
The default without --boundary-envelope remains an explicitly unconstrained
native-network control and is not labelled a fair boundary-conforming PDE
comparator. Preserve both variants and unsuccessful recipes.

The scientific variant trains the wrapped model on the same160/52trajectories,
all11times and128²uniform samples, with the previously registered architecture,
native training schedule, statistics and budgets. Online initialization remains
field-only. Native losses alone establish neither physical representation
qualification nor a PDE accuracy/runtime comparison; independent projection,
actual4096²outputs and complete PDE solves remain outstanding gates.
