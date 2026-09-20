# nmrom-baselines — is our NM-ROM better than other nonlinear-manifold ROMs?

Lane of the 2026-09-20 speed-and-accuracy campaign (`reports/2026-09-20-speed-accuracy-campaign-protocol.md`
on `main` is the binding contract). Worktree `worktrees/2026-09-20-nmrom-baselines`, branch
`exp/2026-09-20-nmrom-baselines` (fork `exp/2026-09-14-no-audit` @ `f3510e88`), cluster namespace
`/cluster/tufts/paralab/tawal01/nmrombase_20260920/`. This file was written before any GPU job.

## 1. Question

There is no evidence either way on whether the project's bank+head+correction NM-ROM is better than
published nonlinear-manifold ROMs. This lane implements, in JAX,

- **(A) Kim, Choi, Widemann, Zohdi 2022** (JCP 451:110841, arXiv 2009.11990): shallow masked (sparse)
  autoencoder, LSPG projection, Gauss–Newton, gappy-POD hyper-reduction with an active-path sub-network;
- **(B) Lee & Carlberg 2020** convolutional autoencoder + LSPG — *only if time permits; first thing cut.*

and compares them with the frozen checkpoint `sep_hfit_dense_mid_N256_dense.pkl` (K=16, R=512,
sha256 `18f0266a…d6b589`) on the shared Burgers 2D family at 128², 256², 512² intervals.

## 2. Gate (nothing from a baseline is used unless it passes)

Reproduce Kim et al. Section 6.2: vector 2D Burgers, $Re=10^4$, $\Omega=[0,1]^2$, $t\in[0,2]$, zero
Dirichlet, $u_0=v_0=\mu\sin 2\pi x\sin 2\pi y$ on $[0,\tfrac12]^2$, $n_x=n_y=60$ (3364 interior nodes per
component), backward differences for advection, central for diffusion, backward Euler $n_t=1500$;
training $\mu\in\{0.9,0.95,1.05,1.1\}$ (6004 snapshots, 10 % validation), test $\mu=1$; two autoencoders
(u and v), $n_s=5$ each, $M_1=6728$, $M_2=33730$ (which fixes $b=100,\ \delta b=10$ since
$M_2=(n-1)\delta b+b$), Adam lr $10^{-3}$, ×0.1 on a 10-epoch training-loss plateau, batch 240, ≤ 10 000
epochs, stop after 200 epochs without validation improvement, Kaiming (PyTorch default) initialisation.
Reference state $x_{ref}(\mu)=x_0(\mu)$, $\hat x_0=h(0)$; per-feature scaling to $[-1,1]$.

Published numbers: NM-LSPG "less than 1 %" maximum relative error at $n_s=5,n_{train}=4$ (their Fig. 14);
NM-LSPG-HR 0.93–0.98 % (Table 3, 55 residual basis / 58 samples best); LS-LSPG-HR 34–38 %.
Metric $\max_n \lVert\tilde x(t_n)-x(t_n)\rVert_2/\lVert x(t_n)\rVert_2$, larger of u and v.

**Pre-registered pass rule:** median over three training seeds of the NM-LSPG (no HR) error ≤ **1.5 %**
(published < 1 %, tolerance ×1.5 for unpublished details: activation choice, scaling, GN stopping rule,
f32 PyTorch vs JAX initialisation streams) **and** the LS-LSPG control at $n_s=5$ ≥ 10 % (published ≈ 35 %),
so a pass cannot come from an easy problem. HR gate (secondary, gates only the HR arm): NM-LSPG-HR at
55/58 ≤ 2 %. Paper-ambiguity variants allowed if the first attempt fails, at most three, all reported:
activation swish↔sigmoid, per-feature↔global scaling, f32↔f64 training. If none passes, the lane says so
and no Kim-baseline number is used in any comparison.

Unimplemented/unknown details, declared: their residual-snapshot source for $\Phi_r$ is not stated in
detail; we use the NM-LSPG residuals at converged steps on the four training parameters. Their timing is a
CPU NumPy/PyTorch implementation; ours is GPU JAX, so **their speed-ups are not a reproduction target**.

## 3. Comparison on the shared Burgers family (only after the gate)

Family (`experiments/mr-burgers2d/engines.py`, sha `820039e5…`): scalar
$u_t+u(u_x+u_y)=\nu\Delta u$, Gaussian bump IC with 5 descriptors, $\nu\in[0.01,0.1]$ log-uniform,
$T=0.25$, sign-upwind + backward Euler, $\Delta t=0.005$ (50 steps), zero Dirichlet, state = $(L-1)^2$ interior.
Split = the shared neural-operator protocol (`dataset_seed=20260914`, `pde_seed_code=22`): **128 train,
32 validation** parameter draws, regenerated from seed on the cluster and value-checked against the
read-only cache index (`pilot-data01`, index sha256 train `5333584b…`, validation `468b9e70…`).
The final cohort stays sealed. Inside train, cases 0–111 fit the autoencoder and cases 112–127 are the
**tuning subset** on which every hyper-parameter choice is made; validation cases are never used to choose.

Baseline training data (as the method prescribes): all 51 same-grid FOM states of each fit trajectory at the
target mesh (5712 snapshots), Newton 1e-9 / BiCGStab 1e-7.

**Metric:** fixed-initial relative error against the same-grid reference FOM (`fft_tight`: dt 0.005, Newton
1e-6, linear 1e-8), $\max$ over the five evolved output times $t\in\{.05,…,.25\}$ and over the 32 validation
cases ("worst evolved same-grid"); median, all-times value and projection (autoencode-only) error reported
beside it.

**Arms per mesh** (K = 8, 16, 32): Kim NM-LSPG; Kim NM-LSPG-HR (their online algorithm); POD-LSPG at the
same K (linear control — must be clearly worse than the nonlinear arms, or the family is too easy to
discriminate); our ROM q=0 (fast) and q=256 dense (accurate), K=16 only (one frozen checkpoint exists);
FOM `fft_tight` and `nt1e-4_dt005`. No coarse-grid/refined-reference arm: this lane's question is
reduced-vs-reduced on the same grid; FOM rows are printed so nobody reads a reduced-vs-reduced win as a
win over the full-order solve.

**Fair tuning (documented sweep, tuning subset only, at 128², K=16, then frozen for all meshes):**
reference state {IC, zero}; scaling {per-feature, global}; mask $(b,\delta b)$ ∈ {(100,10),(50,5),(200,20)};
encoder width $M_1$ ∈ {2n (published), 4096, 1024}; lr {1e-3, 3e-4}; plateau patience {10, 50}; GN cap {20};
HR (basis, samples) grid. Selected by worst evolved LSPG error on the tuning subset.
The published $M_1=2n$ dense encoder needs $2n^2$ weights: 5.2e8 at 128², **8.5e9 at 256² (34 GB f32 before
Adam state) and 1.4e11 at 512²** — the prescribed recipe stops fitting an 80 GB GPU at 256². That is a
reported finding; the capped-$M_1$ variant is what runs above 128², labelled as our adaptation.

**Timing:** one allocation per mesh; every arm, our ROM and the FOMs interleaved in a fresh random order
per (repetition, case); 0.25 s GPU burn-in before every invocation; `block_until_ready` around device-to-
device timing (supplied field on GPU → six dense fields on GPU); ≥ 5 retained repetitions × 6 cases; medians;
GPU model and UUID recorded. Peak memory = XLA memory analysis of each compiled query (arguments + outputs
+ temporaries) and process-level `peak_bytes_in_use` as a cross-check. Training time and the mesh at which a
baseline stops fitting/training within its wall budget are recorded per arm.

Speed mandate: applies to our ROM's arm, which other lanes own; here our ROM runs the b-panel production
query (`topfix.make_query`, vendored from `exp/2026-09-17-b-panel` @ `25434a27`) unchanged. Advantage not
implemented for the baselines is listed in the report (e.g. their CPU sub-network is tuned for sparse
sampling; we evaluate the same sub-network densely on GPU).

## 4. Controls that must fail on real data

- LS-LSPG at the gate must be ≥ 10 % (published 35 %).
- POD-LSPG at matched K on our family must be worse than NM arms; if not, say the family does not
  discriminate.
- Mask tables are tested against the paper's dense construction (`test_mask.py`).
- HR parity: with all rows sampled and a full-rank basis the HR path must reproduce NM-LSPG to 1e-8.
- NumPy audit recomputes every reported error from saved fields.

## 5. Stop rules and budget

≤ 2 running, ≤ 8 total GPU jobs: J1 gate; J2 tuning sweep 128²; J3–J5 final 128²/256²/512²; 3 spare.
Stop the Kim arm if the gate fails after three variants. Cut (B) unless J1–J5 finish with > 1 day left.
Missing the bar or losing to a baseline is reported as such.

## Glossary

- **NM-ROM** nonlinear-manifold reduced-order model: the state is a decoder output $x_{ref}+g(\hat x)$.
- **LSPG** least-squares Petrov–Galerkin: each time step minimises the full discrete residual norm over $\hat x$.
- **HR** hyper-reduction: the residual is evaluated only at sampled rows and fitted to a residual POD basis.
- **K / $n_s$** latent dimension. **$M_1,M_2$** encoder/decoder hidden widths. **$b,\delta b$** mask block width/shift.
- **Same-grid error** error against the full-order solve on the same mesh and time step.
- **Evolved times** the requested output times after $t=0$. **Tuning subset** 16 training cases held out from fitting.
- **FOM** full-order model. **POD** proper orthogonal decomposition (linear basis).
