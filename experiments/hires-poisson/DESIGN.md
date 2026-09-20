# hires-poisson — pre-registration (2026-09-20)

Lane of the speed-and-accuracy campaign; binding contract:
`reports/2026-09-20-speed-accuracy-campaign-protocol.md` (on `main`). Worktree
`worktrees/2026-09-20-hires-poisson`, branch `exp/2026-09-20-hires-poisson`, forked from
`exp/2026-09-20-paper-p3d` @ `9b7c638b`. Cluster namespace
`/cluster/tufts/paralab/tawal01/hires_p_20260920/`, one directory per job. This file contains no
results.

## Question

Does the NM-ROM's speedup over conjugate gradient keep growing past the meshes measured so far,
**at high accuracy**, when the frozen $R{=}512/K{=}32$ Poisson 2D checkpoint (trained at 255
intervals) is transferred without retraining to $2048^2$ and $4096^2$? The main unknown, answered
by the first job: does the *corrected* accuracy (q > 0) survive transfer above $1024^2$.

Orientation from earlier audited panels (not claims of this lane): $1024^2$ q=0 3.15 % / q=256
0.96 % worst same-grid, ROM ≈ 4 ms device and mesh-flat; bank floor 0.74 %.

## Model and transfer (nothing is trained in this lane)

Decoder $u(x;z,y) = G(x)\,[\,h(z) + C_q y\,]$ with bank $G:\Omega\to\mathbb{R}^{R}$ (frozen
feature network evaluated at the new mesh's nodes), head $h:\mathbb{R}^{K}\to\mathbb{R}^{R}$,
nested correction directions $C_q\in\mathbb{R}^{R\times q}$ built once in coefficient space at the
training mesh by the unchanged `plin_core.extend_basis`. The query solves the row-scaled weak
least-squares problem

$$\min_{z,y}\ \big\| \Lambda^{-1}\Phi^{\top}\!\big(A\,G\,[h(z)+C_q y] - f\big) \big\|_2
 \;=\; \min_{z,y} \| B\,[h(z)+C_q y] - f_m \|_2 ,\qquad B=\Phi^{\top} G\in\mathbb{R}^{M\times R},$$

over the $M$ lowest discrete sine modes $\Phi$ of the five-point operator, with $y$ eliminated
exactly (thin QR of $BC_q$) and $z$ found by the unchanged damped LM kernel
(`kernel_solver.make_lm_kernel`, budget 300, normalised-gradient stop $10^{-6}$), started from the
nearest cached training code. Test-count rule `m4`: $M$ requested $= 4(K+q)$. The top rung
$q=R$ is the rank-$R$ linear reduced model solved directly by a triangular solve.

What is new is only how the mesh-sized bank is held: row chunks, never one $(n-1)^2\times R$
array ($68.7$ GB in f64 at $4096^2$). $B$ for every $M$ is a gather from one chunk-accumulated
contraction $T_{abr}=\sum_{x,y}S_{xa}G_{xyr}S_{yb}$; the bank rank/metric comes from a TSQR of
the chunk $R$ factors; the projection floor from chunk-accumulated $G^\top u$.

## Arms (every job, one allocation, one physical GPU)

| family | subjects |
|---|---|
| NM-ROM fast | `rom_q0_*` |
| NM-ROM accurate | `rom_q{64,128,256}_*`, headline accurate arm fixed in advance: **q = 256** |
| linear top rung | `rom_q512_linear_*` (labelled: a linear ROM on the learned bank) |
| named FOM | unpreconditioned CG, zero start, f64, relative residual $10^{-2}$ (`cg_0.01`) — the comparator of the orientation numbers |
| tighter FOM | CG at $10^{-3},10^{-4},10^{-6}$; "fastest tested FOM with error $\le$ the ROM's" is selected from all same-mesh CG rows by worst physical error |
| direct control | `dst_direct` (FFT-based DST-I solve) — a labelled control, not the named FOM |
| coarse-grid FOM | DST and CG($10^{-2},10^{-4}$) on $n_c\in\{64,128,256,512\}$ intervals from the point-sampled supplied source + bilinear interpolation to the requested nodes (interpolation charged); matched-accuracy pick = fastest coarse arm whose worst **physical** error (vs the 2×-refined reference) $\le$ the ROM arm's |

ROM variants (same numerics up to the decode): `retained` = the unchanged single-bank
`correction_core` kernel with its diagnostics inside the timed region (the unoptimised baseline,
run wherever it fits); `lean64` = same query, diagnostics moved to an untimed post-query check,
chunked f64 decode; `lean32` = `lean64` with the bank stored and applied in f32 (field returned
f64). POD-LSPG is out of scope here (it was timed at $1024^2$ in p-linear; snapshots at $4096^2$
are a separate cost); that is a stated omission, not a claim.

## Cohort, errors, timing

- Sources: the twelve **opened development** sources of p-linear/p-bank-head (seeds 7090703 × 6,
  7090732 × 6), regenerated from seed on the cluster. Nothing is tuned on them in this lane: the
  checkpoint, directions, rule `m4`, LM settings and headline arm q=256 are all fixed above. These
  are development sources, not a sealed cohort; reported as such.
- Errors: relative $L^2$ over all nodes against (a) the same-grid five-point truth and (b) the
  $2n$-interval five-point solution subsampled ("physical"). Both from SciPy DST on the host.
  Headline accuracy = worst same-grid error over the 12 sources (the protocol bar); matched-
  accuracy selections use physical error because coarse arms are only meaningful against a
  refined reference.
- Timing: host f64 source in → host f64 nodal field out for every subject; input copy, device
  work and output copy in one synchronised interval; randomised subject order per case; 0.1 s GPU
  burn-in before each invocation; **5 retained repetitions × 12 sources**, medians; physical GPU
  UUID read through the CUDA driver before and after every invocation (outside the timed region).
  Device-only medians are reported beside totals because at $\ge 2048^2$ the identical
  host↔device copies are a large share of every subject.
- CG at $10^{-6}$ on $4096^2$ may be restricted to fewer sources (`slow_subject_cases`), never
  fewer than 5 repetitions; it cannot be the named comparator and any restriction is printed.

## Pass bar (protocol, pre-registered)

At the largest mesh reached: accurate arm (q=256) worst same-grid error $\le 1\,\%$ (stretch
$0.5\,\%$) **and** $S=T_{\rm CG\,1e\text{-}2}/T_{\rm ROM}\ge 5$ on median total time in the same
job. Reported beside it: the fast arm; S against the fastest CG with error ≤ the ROM's; S against
the matched-accuracy coarse-grid arm; S against DST. The bank floor (0.74 % at $1024^2$) makes the
0.5 % stretch unreachable with this checkpoint unless the floor drops under transfer; that is
stated now. A miss is a result.

**Expected and stated in advance:** the coarse-grid and DST controls are likely to be faster than
the ROM at matched accuracy (a 1 %-accurate answer does not need a $4096^2$ solve). If so, the
lane reports "speedup over same-mesh CG grows with mesh; the ROM does not beat a coarse solve at
its own accuracy" — never softened.

## Gates and controls

1. Preflight `jax_backend=gpu` (exit 42), x64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, one visible GPU.
2. Bank numerical rank $=R$ (TSQR), correction block full rank, training/dev disjoint.
3. **Parity** (per q, all 12 sources): `lean64` vs `retained` field $\le 10^{-12}$ relative and
   identical integers (LM attempts, accepted, Jacobians, exit reason, start index). `lean32` vs
   f64: identical integers and field $\le 10^{-4}$ relative — the f32 tolerance was set from the
   local N=64 smoke, which measured $6\times10^{-8}$ (q=0) to $4\times10^{-5}$ (q=R) and FAILED a
   first-guess $10^{-5}$ limit, so the gate is known to be able to fail on real data. `lean32` is a
   labelled reduced-precision-decode arm. At $4096^2$ the retained kernel and possibly the f64 bank
   do not fit; parity is then carried by the $2048^2$ job of the same commit and said so.
4. Every lean solve stationary (full and reduced normalised gradient $\le 10^{-6}$), computed
   untimed; CG rows must report `cg_converged`.
5. Independent audit `hp_audit_np.py` (NumPy/SciPy only, imports no JAX/driver), run on the
   cluster as a second process because the fields are tens of GB: regenerates sources from seed
   with its own rng code, solves references, **verifies them by the five-point stencil residual**,
   recomputes both errors for every saved field and compares with every recorded row
   ($\le 10^{-9}$ rel + $10^{-11}$), checks hashes/boundaries/UUID, derives the table. Negative
   control done locally: a recorded error perturbed by $10^{-4}$ relative makes it fail. Strided
   subsamples are pulled; full fields are deleted on the cluster.

## Profile → fix → re-measure loop

`SPEED-LOG.md` records hypothesis, change, same-job before/after, parity, kept/reverted. Queue:
(a) diagnostics out of the timed kernel [`retained`→`lean64`]; (b) f32 decode; (c) q=R direct
triangular solve; then, from the measured breakdown (input / device / output medians and LM
attempt counts): DST-based projection vs thin sine products at large $n$, fewer/larger chunks,
device-only vs host-copy share, f32 output option for all subjects alike, accuracy levers within
the frozen checkpoint (larger $M$, q between 256 and 512).

## Work order and budget (≤ 2 running, ≤ 8 total GPU jobs)

1. `hp2048` — $2048^2$, H200, 240 G. Answers the transfer question. 2. `hp4096` — $4096^2$
(f32 bank if f64 + working set does not fit; decided from job 1's memory/timing). 3. L-shaped
$1024^2$, $2048^2$ (code from `worktrees/2026-09-17-lshape`; own design amendment before
submission). 4. 3D $128^3$ (code from `experiments/paper-p3d`; own amendment). 5–8. re-measure
jobs from the speed loop. $256^3$ only if all else is done.

## Stop rules

Stop a mesh family when the bar verdict is measured and the speed-loop queue has no untested item
expected to move the verdict; stop the lane at 8 jobs. A failed gate stops the job's numbers from
being reported until explained in an amendment.

## Amendments

(none yet)

**A1 (2026-09-20, before any GPU job) — independent Codex audit applied.** Codex CLI
(gpt-6-astra, read-only; its sandbox could not execute commands here, so the files were inlined
into the prompt) returned eight findings; record in `checks/codex-design-audit.md`. Applied:
(1) gates now control acceptance — the driver writes `COMPLETE` only if assembly, parity (with a
coverage count), solver validity (with coverage) and determinism all pass, and the audit's
`passed` is the conjunction of its gates, never a constant; (2) the f32 parity limit in
`config-2048.json` is the pre-registered $10^{-4}$ (the first commit still carried $10^{-5}$);
(3) the audit now computes the protocol selections and the bar verdict itself, on the full
12-source cohort only — a subject restricted to fewer sources is ineligible as a comparator;
(4) an **assembly gate** compares the chunked build against the parent `core.assemble` at 64
intervals in every job (operator $\le10^{-11}$, bank $\le10^{-13}$, chunked projection floor vs a
direct NumPy least-squares floor $\le10^{-7}$) — the first version of this gate demanded bitwise
bank equality and FAILED on the smoke ($1.3\times10^{-15}$, different evaluation chunking), so it
can fail; (5) the untimed diagnostics restore every retained check (correction-recovery backward
error, residual reconstruction, projected-Jacobian rank, LM backward error), run for every
variant, and the linear rung gets a normal-equation stationarity check; (8) chunk consumers are
synchronised before the next chunk is built. Not changed, stated instead: (7) the parent CG
kernel evaluates one extra true-residual stencil inside its timed region (one operator
application against $\ge 87$ iterations, < 1.2 % of CG device time, in the ROM's favour); the
parent's audited CG is kept unmodified and the bias is declared here. At $4096^2$ the retained
baseline cannot fit (Codex: 171.7 GB with all three copies); parity there is carried by the
$2048^2$ job of the same kernels, as already stated in gate 3.

**A2 (2026-09-20, before the 3D job) — 3D arm of the lane, `hp3d_solve.py`.** Same question at
$128^3$ for the accepted Poisson3D checkpoint of `experiments/paper-p3d`
(`runs/final08/checkpoints/{bank,head_K16}.pkl`; trained at $32^3$, $R{=}128$, $K{=}16$, 512
sine tests, three nearest-code starts, LM budget 160, gradient stop $10^{-6}$ — all unchanged). One
job carries $64^3$ (the mesh the orientation number 0.263 % / 1.33× comes from, as the in-job
anchor) and $128^3$. Cohort: the first 12 sources of paper-p3d's **development** (validation-seed
920411) draw; the accepted final cohort is not reused and nothing is tuned. Arms per mesh:
`rom_q{0,32,96}` in variants `retained` (the unchanged `poisson.engine`, in-kernel diagnostic
included), `lean64` (diagnostic removed from the timed kernel), `leandst64` (additionally the
dense $M\times n^3$ source projection replaced by one DST-I + gather; identity
$(\phi^\top f/n^3)_k = \mathrm{dst3}(f)_k/n^{3/2}$), `leandst32` (f32 decode, labelled),
`leandst1` (one start instead of three — a different solver setting, labelled, not a parity arm);
`rom_q128_linear`; CG $10^{-2}$ (named), $10^{-4}$, $10^{-6}$ (paper-p3d's plain CG without history,
cap $40n$ iterations); `dst_direct` control; coarse DST / CG($10^{-2},10^{-4}$) on
$n_c\in\{16,32,64\}$ with trilinear zero-boundary interpolation (charged). Headline arms fixed now:
accurate = `rom_q96_leandst64`, fast = `rom_q0_leandst64`. Parity limits: `lean64` $10^{-12}$,
`leandst64` $10^{-10}$ (FFT vs matmul round-off through the LM), `leandst32` $10^{-4}$, identical LM
integers; local $16^3$ smoke measured $8\times10^{-16}$, $1\times10^{-15}$, $8\times10^{-8}$. Audit:
`hp3d_audit_np.py` (own seven-point stencil-residual check of the SciPy references). 5 reps × 12
sources, same timing contract as 2D but on interior arrays (paper-p3d's scope).
