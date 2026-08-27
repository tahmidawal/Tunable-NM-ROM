# Running notes — 1D Burgers node screening + Poisson quadrature-free (exp/2026-08-27-b1d-poissonqf)

User-directed screening experiment, started 2026-08-27 evening with explicit
go ("start now") and explicit parallelize-for-speed instruction. Two parts,
one worktree (`worktrees/2026-08-27-b1d-poissonqf`, branch
`exp/2026-08-27-b1d-poissonqf`, cut from `exp/2026-08-27-nodes-mm` — the
superset of the codesign node-learning machinery), one cluster namespace
`/cluster/tufts/paralab/tawal01/b1dqf/`, one directory per job.

## Part A — 1D Burgers frozen-decoder node learning at N=128/256/512

New self-contained 1D testbed (`b1d_common.py` + `sep_b1d_screen.py`; nothing
1D existed in this repo — verified by a full-tree sweep). Discretization
mirrors the 2D testbed one dimension down: sign-upwind advection, centered
diffusion, backward Euler dt=0.005 x 50, Newton with dense interior Jacobian
(direct solve; no BiCGStab landmines), truth residual gate 1e-8. Weak form on
M sine modes = exact 1D Dirichlet-Laplacian eigenvectors; all linear terms
exact through `A = Phi^T G` (exlin rule); advection-only NNLS node fits.

Sizes, fixed across N (the project recipe M=4K, r=4K, scaled so the generous
budget is legal at N=128 with 126 interior points): **K=8, R=32, M=32**;
budgets m = 32 (=M, tight), 16 (=M/2, added arm — the N=64 smoke showed m=M
binds only mildly in 1D, so the halved budget makes the screening decisive
either way), and generous m = min(4M, n_i) = 126/128/128 at N=128/256/512
(the N=128 generous arm uses every interior point and its NNLS fit saturates
— flagged in the report). Decoder trained normally in-job (40k steps,
512 train trajectories, 8192-snapshot pick, seed 0), then completely frozen.
Six certified arms per N: oracle (full grid), NNLS/learned at m=M,
NNLS/learned at m=M/2, NNLS at generous. Node learning = arm-n recipe:
positions only, sampled-advection + sampled-gradient losses (frozen
teacher/denominators), NNLS weight re-solve every 500 of 2000 steps,
LR 3e-3, sigmoid box + min-separation. 8 fresh test trajectories
(TEST_SEED=1), rollouts with the r5 LM rule; per-arm burn-in + 3 timed
repetitions per trajectory, accuracy read from the timed invocation.

Gates, all passed in the N=64 local smokes (2 runs, exit 0): data residual
1.7e-13; gate E (eigen identity) 1.9e-13; gate F (weak ops == Phi^T FOM
residual) 1.8e-15; gate C (continuous node machinery at grid init) 1.9e-14;
gate D (FD vs autodiff, kink-free window) 2.0e-8.

## Part B — Poisson 2D quadrature-free exact residual at N=128/256/512

The designed-never-run cell from 2026-08-26 (`(Λ⊙ΦᵀG)h − Φᵀf`, no EQ at
all). Driver `sep_poisson_qf.py` (built by a parallel subagent, spec + review
by the coordinator): three residual paths through the SAME incumbent
`lm_tau_poisson` solver — FULL (all interior nodes), EQ (incumbent NNLS
m=4M=256), QF (whole residual = one precomputed (M,R) matrix B = ΦᵀG; no
sample points). Frozen decoders inherited from the sepdec scaling arms
(N128 K16 R96, N256 K16 R64, N512 K16 R64 nff128; sha256 + provenance in
`runs/inherited_qf/README.md`); no training anywhere. Truth = in-job CG at
1e-13 from the recorded seeds; held-out + fresh-seed cohorts (16 each,
historical fresh seeds per N). Gates: S (source term bitwise), F/E (cached
banks == meshfree decoder), Q (QF == FULL to machine precision at 32 random
states AND at every solve solution). Balanced timing (forward/reversed
sweeps), GPU burn-in, all raw reps persisted; setup-cost split recorded (EQ
NNLS fit vs QF B-matmul — the QF path removes the NNLS fit entirely).

N=64 smoke (local, gpu): gate Q 9.0e-16 resid / 1.1e-15 grad (random),
0 fails at solutions; EQ b_resid 4.7e-2 at solutions vs QF ~3e-14; QF the
fastest path (10.3 ms vs FULL 14.7 / EQ 12.4); EQ setup 151 s NNLS vs QF
4.9 s B build.

## 2026-08-27 — submission

Six jobs, all A100, gpu partition, one dir each under `b1dqf/`, squeue
checked before/after each submit, manifests checked in-job:

- 2967085 `b1d_n128`, 2967087 `b1d_n256`, 2967089 `b1d_n512`
- 2967090 `qf_n128`, 2967093 `qf_n256`, 2967094 `qf_n512`

Code committed at `89a14ab` (1D testbed) and `b29593e` (QF driver +
inherited checkpoints) on the branch before staging (stage records the
commit). Monitor armed on the queue.
