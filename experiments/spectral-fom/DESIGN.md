# spectral-fom — design and pre-registration

Written 2026-09-23 before any cluster result of this lane.

**Purpose.** The paper compares NM-ROM only with named iterative FOMs (CG, CN–CG, Newton–BiCGStab). The
spectral / fast-transform FOMs are kept out of the paper for now. This lane measures and stores them, so the
authors can decide later. Nothing from here goes into `paper/`.

## Solvers

All solvers solve the paper's own discrete system on the paper's own grid.

- **Poisson square (2D) and Poisson cube (3D).** Zero Dirichlet boundaries, with the 5-/7-point Laplacian scaled by
  $n^2$. The solver is $u = S\,(S f S / \Lambda)\,S$, where $S$ is the orthonormal DST-I and $\Lambda$ holds the discrete
  eigenvalues $\sum_a 4n^2\sin^2(\pi k_a/2n)$. This is an exact solve of the same discrete system. Two
  implementations are timed:
  - `dst_fft`: an odd extension followed by a real FFT of length $2(N+1)$;
  - `dst_mm`: a dense sine matrix applied by GEMM.

  The spectral FOM is the faster of the two at each mesh. This is disclosed as a best-of-two choice.
- **Heat 2D/3D.** Diagonalised by the same DST, with two arms:
  - `modal_cn`: Crank–Nicolson applied exactly in modal space at the paper's $\Delta t$. The per-mode factor is
    $g^{s}$ with $g = (1-\tfrac12\Delta t\nu\lambda)/(1+\tfrac12\Delta t\nu\lambda)$. It is the exact CN solution,
    that is, CN–CG with the linear solve done to round-off.
  - `modal_exp`: exact propagation $e^{-\nu t\lambda}$. This is the heat lane's same-grid truth, so its error is
    round-off by construction.
- **Burgers 2D.** Specified separately, in section B below, before any Burgers timing.
- **L-shape.** No fast transform diagonalises the Dirichlet Laplacian on the non-rectangular domain. There is
  no spectral arm.
- **NS 3D.** The lane's own CNAB2 FOM is already Fourier pseudo-spectral. Its timings are recorded from
  `ns3d-shift-head` and not rerun.

## ROM settings

The accurate and fast settings are taken from the owning lanes' committed selections. Each is re-run here with
that lane's committed code, staged byte-for-byte from the pinned commit and recorded in `PROVENANCE.json`. This
lane never chooses them.

- Poisson 2D: `exp/2026-09-23-poisson-bank-knob` @ 06546331. Accurate is `R512_linear`, fast is
  `R128_linear`, on the 12 development sources.
- Cube: `exp/2026-09-23-poisson-bank-knob-3d` @ d2c775a9, `frozen-N32/64.json`. Accurate is `R128_linear`,
  fast is `R64_linear`, on the final cohort (seed 920499, 64 cases).

## Validation (untimed, in the job, before timing)

- **Spectral vs SciPy truth.** The spectral field is compared with the SciPy `dstn` same-grid truth on every
  case. The limit is $10^{-10}$ relative.
- **The paper's CG converges to the spectral field.** On case 0, CG runs at rtol $10^{-6}$, $10^{-8}$ and
  $10^{-10}$. Every run must converge. Its distance to the spectral field must decrease strictly, and the last
  distance must be $\le 10^{-8}$.
- **Control that must fail.** The same DST solve with the continuum eigenvalues $(\pi k)^2$ must exceed
  $10^{-10}$.

## Timing (`spec_timing.py`)

There is one allocation per mesh, and meshes run sequentially inside one job.

**Phases.** A–B–A, in this order:
1. ROM phase A1;
2. spectral phase B;
3. ROM phase A2, identical to A1.

Each phase runs 10 retained repetitions over every case, in randomised order within each case. Between phases
there is a sync, a 5 s cooldown and a fixed 2 s dummy matmul kernel. Before every invocation there is 0.1 s of
burn-in with the same kernel, `gc.collect()`, and the GPU UUID guard. The garbage collector is disabled during
the timed call.

**What is timed.** GPU time is `fused_device_seconds`: from after the synchronised host-to-device input copy
until `block_until_ready`. This is the owning lane's scope.

**Timing statistics.**
- ROM: the median over A1 ∪ A2.
- Spectral: the median over B.

**Gates** (limit 1.10):
- **Drift:** the ROM median in A2 divided by the median in A1 must lie in $[1/1.1,\ 1.1]$.
- **Within-phase neighbour:** the median after the slowest-median other subject, divided by the median after the
  fastest-median other subject, must be $\le 1.10$.
  - With two subjects in a phase, the comparison is "after the other" vs "after itself", oriented slow over fast.
  - Each side needs at least 3 samples.

Also required: determinism (byte-identical fields on every repetition), and the independent NumPy audit
(`sp_audit_np.py`). The audit uses a dense sine-matrix truth, not an FFT. It must detect two controls: a
swapped-case truth and a perturbed recorded error.

## Reported quantities

**Error** follows each problem's paper convention: relative L2 against the same-grid discrete truth, worst over
the cohort.

**Ratio** = spectral FOM ms / our ms. A ratio below 1 means the spectral solver is faster.

**Choosing the spectral FOM.** For exact solvers (Poisson, heat `modal_exp`), it is simply the faster variant.
For solvers with their own error (heat `modal_cn`, the Burgers ladder), it is the fastest spectral setting whose
worst error is $\le$ our arm's worst error. The tight/exact spectral arm is also reported.

**Timing rows that fail a gate** are reported with the failure beside them and are not called gate-clean.
