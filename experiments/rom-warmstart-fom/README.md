# ROM-warm-started FOM — cost of an FOM-exact hybrid

**Question.** After the ROM reaches some tolerance, hand its solution to the FULL-ORDER
solver as an *initial guess* and finish to full accuracy. The answer is then FOM-exact by
construction, so the ROM's accuracy ceiling stops being a limitation and the only remaining
question is **cost**: what is the total, and how does it scale with `N`?

If the hybrid wins, every accuracy objection the reviewers raised — including *"a direct
solver beats you"* — is sidestepped, because the delivered field **is** the FOM's field.

Scope: **Poisson-2D** and **Burgers-2D**. Both arms reuse the frozen reference harnesses
(`poisson2d-rom-objective/`, `burgers2d-rom-latent-stepping/`); neither the ROM nor the PDE
operator is re-implemented here.

---

## TWO RISKS, STATED UP FRONT AND MEASURED

These are not caveats discovered afterwards; they are the two ways this experiment was
expected to fail, and both are measured rather than assumed.

**(a) Poisson: the ROM is not free, and at small `N` the FOM is very cheap.** The reference
cell measured the Poisson latent solve at 19.7–20.0 ms flat across a 289× DOF range while the
FOM CG went 5.6 → 96.0 ms. A hybrid that pays ~20 ms *before* it starts the FOM **cannot**
win below some resolution. Locating that crossover `N` — or establishing that there is none
inside the ladder — is a headline result here, not a footnote.

**(b) Burgers: the FOM already warm-starts.** `burgers2d_film`'s implicit solver starts each
Newton solve from `u_{n-1}`, which at `dt = 0.005` is an excellent guess. The ROM must beat
**that** bar. The win was expected to concentrate in the first steps and to be small or
negative overall. To keep the comparison honest a **third arm** is measured: linear
extrapolation `2u_{n-1} − u_{n-2}`, the classical trick a practitioner would reach for first.
The ROM is therefore compared against the best cheap alternative, not only the weakest one.

A third, related honesty measure was added on the Poisson side: the FD Poisson system on a
square with Dirichlet walls is **diagonalised exactly** by the same discrete sine basis the
ROM uses for its test modes, so an exact **direct** solve is available and is timed alongside
everything else. That is the strongest form of the reviewers' objection and it is reported,
not avoided.

---

## Design

### Poisson-2D (`wsf_poisson.py`)

```
total  =  t_pre  +  t_rom  +  t_decode  +  t_fom(from the ROM guess)
baseline                                =  the SAME CG from a ZERO start
```

| stage | what it is | charged to the hybrid? |
|---|---|---|
| `t_pre` | the per-query input projection `Λ^{-1} Φ_M^T f` (the mode table is a per-mesh constant, built offline and timed separately) | **yes** |
| `t_rom` | the jitted weak-form LM latent solve on the NNLS-EQ quadrature (`M=64` test modes, `m=256` meshfree points) | **yes** |
| `t_decode` | decoding the latent onto the FOM's interior grid — the array CG is handed | **yes** |
| `t_fom` | CG from that array to `tau_FOM` | **yes** |

- **ROM tolerance.** `rom_tau` is the **relative reduction of the weak-form objective from
  the initial guess**:

  > stop at the first *accepted* LM iterate with `V(z) ≤ rom_tau · V(z_0)`, where
  > `V(z) = ‖ Wl · (Φ_M^T (w_q ⊙ dec(z, pts))) − f_m ‖_2` and `z_0` is the initial latent
  > (the mean training latent). `rom_tau = 0` disables the test and the solver is then the
  > reference LM, stopping on its own relative-decrease / step-size / budget rules.

  This is the identical definition used by the sister `cost-to-tolerance` cell, so the two
  compose. A tolerance on `‖Au−f‖/‖f‖` is *unreachable*: at the weak-form solution that sits
  near 2e-1 while the field error is ~8e-3.
  Ladder: `0.5, 1e-1, 1e-2, 1e-3, 0`. The three middle values are the ones the sister
  `cost-to-tolerance` cell also runs; `0.5` probes the very loose end and `0` the
  reference solver's own stopping rules (reported as `ref. stops`, **not** as
  "converged" — the reference LM may stop on budget or lambda saturation).

- **`tau_FOM` ∈ {1e-6, 1e-8, 1e-10}**, on the relative discrete residual `‖Au−f‖_2/‖f‖_2`.

- **One CG kernel for both arms.** `wsf_util.make_cg` is a jitted, iteration-counting CG in
  which `x0` and `tau` are *runtime arguments*. The warm-started arm and the zero-start arm
  therefore execute the same compiled code with the same stopping test; the only difference
  is the value of `x0`. It is an **outer true-residual loop** around the textbook inner
  recursion: the returned iterate provably satisfies `‖b−Ax‖/‖b‖ ≤ tau` on the *recomputed*
  residual, not merely on the recursively updated one, because recursive-residual drift
  depends on the trajectory and would otherwise make "FOM-exact to tolerance" an
  initial-guess-dependent property. The testbed's own `jax.scipy.sparse.linalg.cg` cannot
  report an iteration count, so it is retained as the **correctness reference** (checked at
  every reported tolerance on several right-hand sides) and is additionally **timed with a
  runtime `x0` for both arms** as a baseline sensitivity check, rather than being the timed
  baseline itself.

- **What "FOM-exact" is checked against.** The correctness gate is *reference-free*: every
  row asserts that the delivered iterate's true relative residual is `≤ tau` in **both**
  arms. The reported `err_final` (against a reference solution computed by CG at 1e-13) is a
  secondary check, and at the tightest `tau` it is bounded below by the reference's own
  accuracy — the achievable relative residual of the reference grows with `N` (measured
  1.0e-13 at `N=128`, 5.7e-13 at `N=256`), an f64 floor of the FD operator, so the reference
  is only required to be 10x tighter than the tightest reported tolerance.

- **`N ∈ {32, 64, 128, 256, 512}`.** The coordinate decoder is meshfree, so the same `N=64`
  `K=8` hard-BC checkpoint is used at every `N` and the NNLS-EQ weights are **refit on each
  `N`'s grid** (as in the reference `followup/fu_timing.py`).

### Burgers-2D (`wsf_burgers.py`)

```
hybrid total = t_rom_ic + t_rom_rollout + t_decode + t_fom(from the ROM guesses)
baseline     = the SAME implicit chain warm-started from u_{n-1}
third arm    = the SAME implicit chain warm-started from 2u_{n-1} − u_{n-2}
```

- **One chain kernel for all three arms.** `make_chain` builds a single jitted 50-step
  backward-Euler chain whose only per-arm difference is a **traced** `guess_mode` integer.
  Newton stopping test, linear tolerance, operator, compilation and warm-up are therefore
  identical across arms by construction.
- **Newton stopping test:** `‖R(u, u_prev, ν)‖_2 ≤ tau · ‖u_prev‖_2` — the testbed's own
  convergence metric (`burgers2d_film.newton_step` reports exactly this ratio).
  `tau ∈ {1e-6, 1e-8, 1e-10}`.
- **Why a new Newton driver.** The testbed's Newton is a *fixed-length 8-iteration scan* and
  `jax.scipy.sparse.linalg.bicgstab` cannot report an iteration count, so neither can answer
  "how many iterations did the warm start save". The driver here imports the testbed's
  `residual` **verbatim** — the discrete operator is not re-implemented — and every `(N, tau)`
  asserts that the previous-step arm reproduces the testbed's own rollout.
- **BiCGStab NaN landmine.** Known from an earlier round: once the Newton residual reaches
  machine epsilon, BiCGStab's `rho`/`omega` inner products can underflow and return a NaN
  step. Here the Newton loop exits on its tolerance test before that can happen, and the
  counting BiCGStab still detects underflow of `rho`, `rhat^T v` or `t^T t` and any
  non-finite iterate, freezes on the last good state, and **reports** the occurrence
  (`bicgstab_breakdowns`, `newton_flags_nonzero`, `health_warning`). Breakdowns are counted,
  never dropped. The solver also includes the **alpha half-step convergence test**: without
  it, an exactly converged `s = r − alpha A p` gives `t·t = 0` and a naive implementation
  declares a breakdown and discards a converged iterate.

- **Solver-health gate.** A Burgers configuration is published only if every step of every
  arm met the Newton tolerance with finite arithmetic; otherwise the job aborts. A cheap
  *failed* warm solve must never be able to contribute a headline speedup.
- **`N ∈ {32, 64, 128, 256}`**, EQ weights refit per `N`, variant `lspg:eq256:weak64` with
  the hyper-reduced (EQ-node) cold start — the reference cell's headline configuration.

### Timing protocol (non-negotiable)

Every ladder point is measured **sequentially in one process on one GPU** — cross-`N` timings
from different GPUs are invalid and have burned this project before. Warm-up 2, median of 7,
`block_until_ready`, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, `jax_backend=gpu` asserted
before any work. One job for Poisson, one for Burgers, each in its own cluster directory.

### What is deliberately *not* varied

- **`dt`.** The Burgers testbed's `dt = 0.005` is hard-coded in `burgers2d_film`. The
  previous-step guess gets worse as `dt` grows, so the ROM's advantage should grow with `dt` —
  but changing `dt` changes the FOM, the training snapshots and the ROM together, which is a
  different experiment. Everything here is at the testbed's `dt`; the per-step curves show
  where the win would have to come from.
- **Preconditioning.** The Poisson CG is unpreconditioned, as in the testbed. A
  preconditioner would shrink the iteration count in *both* arms and therefore shrink the
  absolute saving a warm start can buy.

---

## Files

```
wsf_util.py       timing protocol, provenance stamp, the counting CG + its reference check
wsf_poisson.py    the Poisson arm
wsf_burgers.py    the Burgers arm (counting BiCGStab + the 3-arm implicit chain)
wsf_summarize.py  runs/hybrid_points.json (the flat output schema) + SUMMARY_TABLES.md
wsf_figs.py       the figures (reports-pipeline style, PNG + PDF)
wsf_style.py      a copy of the frozen reports figure style
cluster/          make_cell.sh / launch.sh / pull.sh  (namespace wsfom/, one job per dir)
in/               the two checkpoints (git-ignored; sha256 recorded below)
runs/             pulled cluster output, logs, and the generated hybrid_points.json
```

## Jobs

Eleven Slurm jobs, each in its own directory under `/cluster/tufts/paralab/tawal01/wsfom/`,
all on `--gres=gpu:a100:1`, `-p gpu`, submitted simultaneously.

| job dir | role | what it produces |
|---|---|---|
| `wsp_n{32,64,128,256,512}` | panel | Poisson: EQ refit at that mesh, ROM accuracy, CG iteration counts from both starts, solver cross-checks, and the within-mesh cost breakdown |
| `wsb_n{32,64,128,256}` | panel | Burgers: EQ refit, ROM rollout accuracy, per-step Newton and BiCGStab counts for all three arms, NaN-guard checks |
| `wsp_cons` | consolidated | Poisson: the whole `(rom_tau x N)` grid and the pure-FOM baseline at every mesh, **sequentially in one job on one GPU** |
| `wsb_cons` | consolidated | Burgers: the whole 50-step rollout from all three starts at every mesh, **sequentially in one job on one GPU** |

**Which numbers come from where.** Iteration counts, accuracy and solver-health fields are
hardware-independent and are taken from the fanned-out panels; table P5 checks that the panel
and consolidated runs agree on them exactly. **Every cross-`N` wall-clock number — the
headline total-time figure, the crossover result and the Burgers cost curve — comes only from
the consolidated runs**, and `wsf_summarize.select_consolidated` enforces that by pooling
consolidated rows by (source file, Slurm job id, GPU, commit, harness source hash) and using
a single group. Within one panel, the cost breakdown is valid because one panel is one job on
one GPU.

<!-- RESULTS -->

---

## Provenance

### Inputs (git-ignored because of their size; sha256 recorded here)

| file | sha256 | origin |
|---|---|---|
| `in/autodec_K8_N64_hbc_stages.pkl` | `45b5ff291216981a…` | `poisson2d-rom-objective/runs/hbc_K8/` — the hard-BC `K=8`, `N=64` FiLM auto-decoder the reference cell's headline used |
| `in/blat_ad_N64_K8.pkl` | `aa07cd4a1471c59a…` | `exp/2026-08-16-burgers2d-rom-latent-stepping` `runs/ad_n64_k8/` — the `K=8`, `N=64` Burgers auto-decoder + POD basis |

### Reference harness sources imported unchanged

| file | sha256 |
|---|---|
| `poisson2d-rom-objective/pro_common.py` | `044d1c3aaf4727bc…` |
| `poisson2d-rom-objective/followup/fu_eq.py` | `92717e93c8c04c93…` |
| `burgers2d-rom-latent-stepping/blat_common.py` | `4fecfe2f87a25327…` |
| `burgers2d-rom-latent-stepping/followup/fu_common.py` | `35c5693d05cdad62…` |

plus `ms_parametric.py` / `ms_autodecoder.py` (from `exp/2026-08-14-multistage-precision`) and
`burgers2d_film.py` (from `exp/2026-08-14-burgers2d-coord-rom`), staged into the job's `deps/`
by `cluster/make_cell.sh` and checksummed into `MANIFEST.sha256` before every `scp`.

Every result row additionally carries `commit`, a **sha256 of every `wsf_*.py`** (so a dirty
tree cannot publish two different codes under one commit hash), `gpu`, `gpu_kind`,
`jax_backend`, `slurm_job_id`, `seed`, and `run_role`.

<!-- PROVENANCE -->
