# B8 matched expanded-data global neural parent control

Register separate latent32 and latent64 attempts, each at most 120 allocated GPU
minutes and one GPU, subject to the shared two-job concurrency limit. Do not
launch until the retained cluster reverse-Jacobian profile passes numerical
parity and the implementation review is complete. Back up source/config first;
retain failed/incomplete attempts and account for the entire allocation. This
configuration does not authorize submissions or automatic compute extensions.

Run `train_stable_expanded.py --out OWNED_OUTPUT --latent 32 --qualification-job
b8implprofile`; the independent64-variable control changes only `--latent
64`. Both use seed2026111994 and exactly40000 updates. The retained wrapper and `output/jacobian/result.json` derivative result
must be backed by a locally/remotely verified retention manifest. A faster PDE
step is not required to establish the algebraic Jacobian prerequisite for
training; all three root-Jacobian and residual-Jacobian numerical parity checks are required.
The b8implprofile overall strict PDE trace gate failed because one accepted-update
count differs (5 versus4 near zero). This failure remains recorded; this
prospective training-only derivative prerequisite does not waive or claim the
PDE trace gate. No complete-solve speed claim follows. An additional cluster
pretraining forward/reverse derivative check covers the actual32/64 initial model.

Use hash-verified B7 repaired network/primary training codes and the verified
original coordinate cache. Append registered32 training and8 development
ridge/two-pulse trajectories from b8trainrefs/b8devrefs, giving160/52 trajectories
and1760/572 snapshot fields. Preserve all11times and128² Gauss quadrature; no
final-test fields. New training codes copy the nearest original training field's
code using the common16² encoder samples. Do not import the unfinished,
extrapolated new-family root fits. All network weights and snapshot embeddings
then train jointly. Existing B7 primary codes retain their initial values.

The model remains the B7 global nonlinear coordinate decoder: fixed spatial
Fourier features, four width128 tanh hidden layers, scalar output with the fixed
boundary envelope, and trainable additive latent shifts in every hidden layer.
It is not an output POD expansion or a local tree. For latent64, append32
modulation rows drawn from a seeded normal distribution at0.01 times the B7
modulation-weight standard deviation; append exactly zero extra code entries.
This algebraically preserves every initial32-code field and permits useful
initial derivatives in the added directions. Record numerical preservation.

Five explicit training strata receive equal0.2 probability: smooth, narrow
primary pulses (width<=0.05), broad primary pulses, ridge and two-pulse. Each
update chooses one stratum, eight distinct trajectories within it and uniformly
sampled output times. This intentionally changes trajectory weighting to an
equal-family objective; it does not change the within-field current-time
normalized physical squared-L2 loss. Per-snapshot Adam clocks/moments advance
only for selected rows. Network gradient clipping5, cosine Adam1e-4 to5e-6;
code Adam3e-3 to3e-4. No weight/code regularizer or relaxed fit criterion.

For the first38000updates, draw1024coordinates independently per field from a
fixed training-only mixture: half normalized physical quadrature weights and
half weighted initial parent residual energy. Use the existing spatial_sampling
helper and original w/p importance weights, without grid-size or self-normalized
correction. This is an unbiased estimate of the physical quadrature objective.
The last2000updates use four fields and all16384quadrature points, retaining the
same family replay. The proposal never sees development fields.

Every5000updates retain a pre-evaluation network/code/optimizer/RNG checkpoint
and evaluate the same36-field panel as the child: original development trajectory
indices0,3,13,20 plus all8expanded development trajectories, each at0,0.125,0.25.
Use nearest training code, interpolated training codes and zero starts with
60-step strict LM; finish the best unfinished start with120more LM steps. Use
S.code_jacobian only after the qualification gate. Record every start/status;
failed fits remain failures. Independent256² Gauss scoring synthesizes the
verified spectral reference and uses its current-time physical norm. Select by
worst independent panel error; unfinished counts and scoring changes are
separate explicit admission conditions, never hidden in the selection metric.

At the end refit all1760training embeddings against the selected fixed network
using full128²quadrature,60+120LM as needed. Retain partial refresh evidence,
all final fit statuses and the selected model checkpoint. Re-evaluate the same
panel with the refreshed training-code library and report it separately from
historical checkpoint metrics. No exact Adam continuation is claimed after this
code refresh. Keep all prior checkpoints and unsuccessful fits.

Strict fit thresholds remain stationarity1e-8 and physical update1e-10 (the
existing sufficiently-small-stationarity clause remains unchanged). The panel
numerical gate is worst error<=0.005, independent-grid scoring change<=0.0002,
and zero unfinished development fits. Passing it does not qualify the full
572-field development cohort or actual4096² output. No PDE admission, runtime
advantage, or proof of tree superiority is claimed. This control tests whether
training on the missing families and adding global neural variables address the
parent limitation before attributing failure or improvement to local children.

Runtime risk: repeated cold panel fits and the final all-training refresh can
consume the allocation. Retain incomplete results if the120-minute cap is hit;
do not omit the full refresh or change fit gates to label the attempt complete.

Pre-submission integrity repair: verify the retained profile SOURCE.sha256
manifest against its retention hash, then require the current neural/stable.py
hash to equal that qualified source entry. Retain the additional in-run
derivative check. Reject any nonfinite candidate codes, fit status or scored
error before selection reductions; retain the offending payload in a separate
failure checkpoint and stop. The refreshed final checkpoint stores refreshed
metrics matching its codes/evaluation, preserves pre_refresh_metrics and the
pre-refresh checkpoint hash, and retains both sets in the result record.
